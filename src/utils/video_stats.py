"""Per-video frame counts, scanned once and shared by every model.

Decoding all 17,988 clips takes a couple of hours, and the result depends only on
the videos themselves - not on the model, tokenizer, or CTC limits. So the scan
is done once into a stats file and every later validation reads that instead.

The counts come from an actual decode. cv2.CAP_PROP_FRAME_COUNT is unreliable on
these webm files because they lack duration metadata.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

STATS_VERSION = 1

# Workers are separate processes already
cv2.setNumThreads(0)


def count_frames(video_path):
    """Decode a video and return its true frame count."""
    cap = cv2.VideoCapture(video_path)
    frames = 0
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frames += 1
    finally:
        cap.release()
    return frames


def _scan_one(task):
    recording, video_path = task
    if not os.path.exists(video_path):
        return {'recording': recording, 'status': 'missing'}
    try:
        frames = count_frames(video_path)
    except Exception as exc:
        return {'recording': recording, 'status': 'undecodable',
                'reason': f'{type(exc).__name__}: {exc}'}
    if frames == 0:
        return {'recording': recording, 'status': 'undecodable',
                'reason': 'decoded zero frames'}
    return {'recording': recording, 'status': 'ok', 'frames': frames}


def scan_videos(rows, workers=1, csv_path=None, progress=None):
    """Decode every video in `rows` and collect frame counts.

    rows: iterable of (recording, video_path)
    progress: optional callable wrapping the results iterator (e.g. tqdm)
    """
    tasks = list(rows)
    frames = {}
    missing = []
    undecodable = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_one, task) for task in tasks]
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, total=len(futures))
        for fut in iterator:
            res = fut.result()
            if res['status'] == 'ok':
                frames[res['recording']] = res['frames']
            elif res['status'] == 'missing':
                missing.append(res['recording'])
            else:
                undecodable[res['recording']] = res['reason']

    return {
        'version': STATS_VERSION,
        'csv_path': csv_path,
        'scanned': len(tasks),
        'frames': frames,
        'missing': sorted(missing),
        'undecodable': undecodable,
    }


def save_video_stats(path, stats):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_video_stats(path):
    """Return the stats file at `path`, or None if it is absent or unusable."""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    if stats.get('version') != STATS_VERSION:
        return None
    if 'frames' not in stats:
        return None
    return stats


def covered_recordings(stats):
    """Every recording the scan reached, whatever the verdict."""
    return (set(stats['frames'])
            | set(stats.get('missing', []))
            | set(stats.get('undecodable', {})))


def summarize(frame_counts):
    """Distribution summary for a list of frame counts."""
    counts = sorted(frame_counts)
    n = len(counts)
    if n == 0:
        return None

    def pct(p):
        return counts[min(n - 1, int(round((n - 1) * p / 100)))]

    return {
        'count': n,
        'min': counts[0],
        'max': counts[-1],
        'mean': sum(counts) / n,
        'percentiles': {p: pct(p) for p in (5, 25, 50, 75, 90, 95, 99)},
    }
