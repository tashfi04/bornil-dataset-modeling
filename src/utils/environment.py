"""Check that every installed package matches environment_lock.txt.

Every reported run has to come from the same environment, so a script stops
before doing any work when a package was upgraded, removed or added. The lock
is the full package list of the Kaggle image the runs use, written by

    pip list --format=freeze > environment_lock.txt

requirements.txt lists only the packages the code imports, at the same
versions. To add a package without changing any installed one, install it with
`pip install <package> -c environment_lock.txt` and regenerate the lock. To move
to a new environment, regenerate the lock, update requirements.txt and
PYTHON_VERSION to match, and re-run the smoke and overfit checks.

Training, evaluation and anything else that runs the model check every
package. A data-preparation script checks only Python and the packages that can
change what it writes (check_environment(packages=...)), so it also runs on
Kaggle's CPU image, which lacks the CUDA packages. A script whose output is
only printed or looked at does not check at all.
"""
import os
import re
import platform
import subprocess
from importlib import metadata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCK_PATH = os.path.join(_REPO_ROOT, "environment_lock.txt")
REQUIREMENTS_PATH = os.path.join(_REPO_ROOT, "requirements.txt")

# pip does not record the interpreter, so it is checked here instead
PYTHON_VERSION = "3.13"

# Enough to see what changed without burying the error under hundreds of lines
_MAX_LISTED = 20

# Each provides cv2, and which one Python imports depends on sys.path order
OPENCV = ('opencv-python', 'opencv-python-headless', 'opencv-contrib-python')


def _canonical(name):
    """Package names as pip compares them: case and -_. are not significant."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_versions(path):
    """Read the `name==version` lines of a requirements-style file."""
    versions = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '==' not in line:
                raise RuntimeError(
                    f"{path}: '{line}' is not an exact version (name==version)")
            name, version = (part.strip() for part in line.split('==', 1))
            versions[_canonical(name)] = version
    return versions


def installed_packages():
    """Every installed package and its version, the same set `pip list` shows."""
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata['Name']
        # The first copy on sys.path is the one Python imports
        if name and _canonical(name) not in packages:
            packages[_canonical(name)] = dist.version
    return packages


def gpu_summary():
    """GPU names and driver versions for the log, or None without nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return '; '.join(gpus) or None


def _differences(reference, installed, reference_name, allow_added=False):
    """Describe each package that is missing, extra, or at another version."""
    lines = []
    for name in sorted(set(reference) | set(installed)):
        want, have = reference.get(name), installed.get(name)
        if want == have:
            continue
        if have is None:
            lines.append(f"{name} {want} is not installed")
        elif want is None:
            if not allow_added:
                lines.append(f"{name} {have} is installed but not in {reference_name}")
        else:
            lines.append(f"{name} {have} ({reference_name} has {want})")
    return lines


def _scoped_differences(locked, installed, packages):
    """Differences for just `packages`.

    Each entry is a package name, or a tuple of alternatives such as OPENCV: at
    least one of them must be installed, and every installed one must match.
    """
    lines = []
    for entry in packages:
        group = (entry,) if isinstance(entry, str) else tuple(entry)
        present = [_canonical(name) for name in group if _canonical(name) in installed]
        if not present:
            lines.append(f"{' or '.join(group)} is not installed")
        for name in present:
            want, have = locked.get(name), installed[name]
            if want is None:
                lines.append(f"{name} {have} is installed but not in the lock")
            elif have != want:
                lines.append(f"{name} {have} (the lock has {want})")
    return lines


def _listed(lines):
    shown = lines[:_MAX_LISTED]
    if len(lines) > _MAX_LISTED:
        shown.append(f"... and {len(lines) - _MAX_LISTED} more")
    return "\n  " + "\n  ".join(shown)


def check_environment(packages=None):
    """Raise unless Python and the installed packages match the lock.

    By default every installed package must match. A data-preparation script
    passes `packages`, the names (or OPENCV-style tuples of alternatives) whose
    versions can change what it writes, and then only those and Python are
    checked. () checks Python alone, for a script that uses only the standard
    library and this repository.

    Returns what a checkpoint records about the environment.
    """
    locked = _read_versions(LOCK_PATH)

    # requirements.txt is what gets installed elsewhere, so it must agree with
    # the lock or the two describe different environments
    required = _read_versions(REQUIREMENTS_PATH)
    disagree = []
    for name, version in required.items():
        # A version without a local label covers every build of that release,
        # the way pip treats it, so torch==2.11.0 covers 2.11.0+cu128
        lock_version = locked.get(name)
        if lock_version is not None and '+' not in version:
            lock_version = lock_version.split('+', 1)[0]
        if lock_version != version:
            disagree.append(f"{name}: requirements.txt has {version}, "
                            f"environment_lock.txt has {locked.get(name)}")
    if disagree:
        raise RuntimeError(
            "requirements.txt and environment_lock.txt disagree:" + _listed(disagree))

    problems = []
    python = platform.python_version()
    if not (python == PYTHON_VERSION or python.startswith(PYTHON_VERSION + '.')):
        problems.append(f"python {python} (expected {PYTHON_VERSION})")
    installed = installed_packages()
    if packages is None:
        problems += _differences(locked, installed, "the lock")
    else:
        problems += _scoped_differences(locked, installed, packages)

    if problems:
        scope = ("The installed packages do not match" if packages is None
                 else "The packages this script depends on do not match")
        raise RuntimeError(
            f"{scope} environment_lock.txt:"
            + _listed(problems)
            + "\nUse the Kaggle notebook pinned to the environment the lock was "
              "taken from. If the change is intended, regenerate the lock and "
              "re-run the smoke and overfit checks.")
    return {'python': python, 'packages': installed}


def describe_environment(environment):
    """One line for the log: Python, the packages the code imports, and the GPU."""
    packages = environment['packages']
    direct = ', '.join(f"{name} {packages[name]}"
                       for name in _read_versions(REQUIREMENTS_PATH))
    return (f"python {environment['python']}, {direct} "
            f"(all {len(packages)} packages match environment_lock.txt); "
            f"GPU: {gpu_summary() or 'none found'}")


def checkpoint_environment_mismatch(checkpoint, current, path):
    """Explain how a checkpoint's environment differs from this one, or return None.

    Packages installed since the checkpoint was written are allowed, so adding
    one (with `pip install -c environment_lock.txt`) keeps earlier checkpoints
    usable. A package that changed version or was removed is a mismatch.
    """
    saved = checkpoint.get('environment')
    if saved is None:
        return (f"{path} has no record of the environment it was trained in. "
                f"It predates environment_lock.txt.")
    lines = []
    if saved.get('python') != current['python']:
        lines.append(f"python {current['python']} (the checkpoint has {saved.get('python')})")
    lines += _differences(saved.get('packages', {}), current['packages'],
                          "the checkpoint", allow_added=True)
    if lines:
        return f"{path} was trained in a different environment:" + _listed(lines)
    return None
