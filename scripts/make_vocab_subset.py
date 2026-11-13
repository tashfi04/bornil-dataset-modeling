"""Build a subset whose val and test sentences use words its train sentences repeat.

Every sentence in the dataset is different, so a model can only generalise by
learning signs that recur across sentences. This builds a subset for that:

1. A target vocabulary: words found in at least --vocab-min-sentences different
   sentences of the full official train split.
2. Train: sentences from the official train split, chosen greedily so that
   each target word appears in up to --target-repeats different training
   sentences. Only the vocabulary guides this choice, and it comes from train;
   val and test play no part in it.
3. Val and test: sentences from the official val and test splits in which
   every word appears in at least --val-min-repeats of the chosen training
   sentences (--max-unknown lets a share of words fall short), sampled at
   random.

Only clips on the approved list (valid_samples_path) are used, so every chosen
clip is trained on. No sentence appears twice in the subset or in two splits.
Rows keep their official split; the loader assigns splits from
train_val_test_split.json as always.

The report prints counts only, never sentence text. With --compare it puts the
same numbers for an older subset CSV alongside.

    python scripts/make_vocab_subset.py --out path/to/subset_vocab.csv
    python scripts/make_vocab_subset.py --compare path/to/other_subset.csv
"""
import os
import sys
import csv
import json
import heapq
import random
import argparse
import unicodedata
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.environment import check_environment
from src.utils.text_utils import normalize_text

SPLITS = ('train', 'val', 'test')
# The cache stores at most this many frames per clip (the subset config's num_frames)
CACHED_FRAMES = 160


def words_of(normalized):
    """Whitespace words with punctuation removed, so "না।" and "না" are one word."""
    words = []
    for word in normalized.split(' '):
        bare = ''.join(c for c in word if unicodedata.category(c)[0] not in 'PS')
        if bare:
            words.append(bare)
    return words


def load_rows(csv_path):
    csv.field_size_limit(10 ** 7)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def candidates(rows, split, approved):
    """Approved rows, one per sentence, and only sentences confined to one split.

    Returns {split: [entry]} where entry holds the row, its normalized words and
    its official split, plus counts of what was left out and why.
    """
    by_sentence = defaultdict(list)
    skipped = Counter()
    for row in rows:
        recording = row['recording']
        if recording not in approved:
            skipped['not approved'] += 1
            continue
        if recording not in split:
            raise RuntimeError(f"{recording} is not in the split file")
        normalized = normalize_text(row['text'])
        by_sentence[normalized].append((row, split[recording]))

    pool = {name: [] for name in SPLITS}
    for normalized, entries in by_sentence.items():
        if len({s for _, s in entries}) > 1:
            skipped['sentence in more than one split'] += len(entries)
            continue
        row, name = entries[0]
        skipped['repeat recording of a sentence'] += len(entries) - 1
        words = words_of(normalized)
        if words:
            pool[name].append({'row': row, 'words': words, 'split': name})
    return pool, skipped


def unknown_share(words, vocab):
    return sum(1 for w in words if w not in vocab) / len(words)


def pick_train(pool, vocab, size, target_repeats, min_share, rng):
    """Greedy choice maximising sum over target words of min(sentences with it, target).

    That objective has diminishing returns, so lazy greedy is exact: a stale
    heap entry only ever overstates a sentence's gain.
    """
    eligible = [e for e in pool if 1 - unknown_share(e['words'], vocab) >= min_share]
    counts = Counter()

    def gain(entry):
        return sum(1 for w in set(entry['words']) if w in vocab and counts[w] < target_repeats)

    heap = [(-gain(e), rng.random(), i) for i, e in enumerate(eligible)]
    heapq.heapify(heap)
    chosen, used = [], set()
    while heap and len(chosen) < size:
        negative, tie, i = heapq.heappop(heap)
        current = gain(eligible[i])
        if current != -negative:
            heapq.heappush(heap, (-current, tie, i))
            continue
        if current == 0:
            break
        chosen.append(eligible[i])
        used.add(i)
        for w in set(eligible[i]['words']):
            if w in vocab:
                counts[w] += 1
    # Every target word reached its target: fill up with random eligible sentences
    rest = [e for i, e in enumerate(eligible) if i not in used]
    rng.shuffle(rest)
    chosen.extend(rest[:size - len(chosen)])
    return chosen, len(eligible)


def report(label, subset, frames, lengths, vocab):
    """Coverage numbers for one subset, as a dict of printable values."""
    by = {name: [e for e in subset if e['split'] == name] for name in SPLITS}
    train_df = Counter(w for e in by['train'] for w in set(e['words']))
    out = {'label': label}
    for name in SPLITS:
        out[f'{name} clips'] = len(by[name])
    for name in ('val', 'test'):
        tokens = [w for e in by[name] for w in e['words']]
        if not tokens:
            continue
        for k in (1, 5, 10, 20):
            seen = sum(1 for w in tokens if train_df[w] >= k)
            out[f'{name} words in >={k} train sentences'] = f'{100 * seen / len(tokens):.1f}%'
        every = sum(1 for e in by[name] if all(train_df[w] >= 10 for w in e['words']))
        out[f'{name} sentences, every word in >=10 train sentences'] = \
            f'{every}/{len(by[name])} ({100 * every / max(1, len(by[name])):.1f}%)'
    out['words in >=20 train sentences'] = sum(1 for v in train_df.values() if v >= 20)
    out['target words in >=20 train sentences'] = sum(
        1 for w in vocab if train_df[w] >= 20)
    for name in SPLITS:
        if by[name]:
            out[f'{name} mean words per sentence'] = \
                f"{sum(len(e['words']) for e in by[name]) / len(by[name]):.1f}"
            out[f'{name} mean BPE tokens'] = \
                f"{sum(lengths[e['row']['recording']] for e in by[name]) / len(by[name]):.1f}"
    out['frames to cache (<=160 per clip)'] = f"{sum(min(frames[e['row']['recording']], CACHED_FRAMES) for e in subset):,}"
    return out


def subset_from_csv(path, split, approved):
    """An existing subset CSV as entries, approved rows only (what training used)."""
    _, rows = load_rows(path)
    entries = []
    for row in rows:
        if row['recording'] in approved:
            words = words_of(normalize_text(row['text']))
            entries.append({'row': row, 'words': words, 'split': split[row['recording']]})
    return entries


def run(full_csv, split_path, approved_path, out_path, compare_path, args):
    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    with open(approved_path, 'r', encoding='utf-8') as f:
        approved_file = json.load(f)
    approved = set(approved_file['valid'])
    frames, lengths = approved_file['frames'], approved_file['target_lengths']

    fieldnames, rows = load_rows(full_csv)
    pool, skipped = candidates(rows, split, approved)
    print(f"{len(rows):,} rows in {full_csv}; usable sentences: "
          + ", ".join(f"{name} {len(pool[name]):,}" for name in SPLITS))
    print("  left out: " + ", ".join(f"{reason} {n:,}" for reason, n in skipped.items()))

    sentence_counts = Counter(w for e in pool['train'] for w in set(e['words']))
    vocab = {w for w, n in sentence_counts.items() if n >= args.vocab_min_sentences}
    print(f"\nTarget vocabulary: {len(vocab):,} words in at least {args.vocab_min_sentences} "
          f"of the {len(pool['train']):,} train sentences")

    rng = random.Random(args.seed)
    train, eligible_train = pick_train(pool['train'], vocab, args.train, args.target_repeats,
                                       args.train_min_share, rng)
    if len(train) < args.train:
        sys.exit(f"Only {eligible_train} train sentences have at least {args.train_min_share:.0%} "
                 f"of their words in the vocabulary; {args.train} are needed. Lower "
                 f"--train-min-share.")
    print(f"Train: {args.train:,} of {eligible_train:,} sentences with at least "
          f"{args.train_min_share:.0%} of their words in the vocabulary, up to "
          f"{args.target_repeats} sentences per target word")

    # In how many chosen training sentences each word appears
    repeats = Counter(w for e in train for w in set(e['words']))

    def short_share(entry):
        return sum(1 for w in entry['words'] if repeats[w] < args.val_min_repeats) / len(entry['words'])

    print(f"Val/test sentences eligible, by the share of words allowed in fewer than "
          f"{args.val_min_repeats} training sentences:")
    for limit in (0.0, 0.1, 0.2, 0.34):
        counts = [sum(1 for e in pool[name] if short_share(e) <= limit) for name in ('val', 'test')]
        print(f"  <= {limit:.0%}: val {counts[0]:,}, test {counts[1]:,}")

    chosen = list(train)
    for name, size in (('val', args.val), ('test', args.test)):
        eligible = [e for e in pool[name] if short_share(e) <= args.max_unknown]
        if len(eligible) < size:
            sys.exit(f"Only {len(eligible)} {name} sentences have at most {args.max_unknown:.0%} "
                     f"of their words in fewer than {args.val_min_repeats} training sentences; "
                     f"{size} are needed. Raise --max-unknown or lower --val-min-repeats.")
        chosen.extend(rng.sample(eligible, size))

    columns = [report('new subset', chosen, frames, lengths, vocab)]
    if compare_path:
        old = subset_from_csv(compare_path, split, approved)
        columns.append(report('compared subset', old, frames, lengths, vocab))
        overlap = len({e['row']['recording'] for e in chosen} & {e['row']['recording'] for e in old})
        print(f"\nClips in both subsets: {overlap:,}")

    print("\nCoverage (word counts are by sentence: in how many train sentences a word appears)")
    keys = [k for k in columns[0] if k != 'label']
    width = max(len(k) for k in keys)
    print(f"  {'':<{width}}  " + "  ".join(f"{c['label']:>16}" for c in columns))
    for key in keys:
        print(f"  {key:<{width}}  " + "  ".join(f"{str(c.get(key, '-')):>16}" for c in columns))

    if out_path:
        keep = {e['row']['recording'] for e in chosen}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if row['recording'] in keep)
        print(f"\nWrote {len(keep):,} rows to {out_path}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build a vocabulary-based subset")
    parser.add_argument('--csv', default=None, help="Full dataset CSV (defaults to the subset config's)")
    parser.add_argument('--out', default=None, help="Where to write the subset CSV (omit to only report)")
    parser.add_argument('--compare', default=None, help="An older subset CSV to report alongside")
    parser.add_argument('--train', type=int, default=2286)
    # Sized to fit how many val/test sentences pass --val-min-repeats
    parser.add_argument('--val', type=int, default=160)
    parser.add_argument('--test', type=int, default=140)
    parser.add_argument('--vocab-min-sentences', type=int, default=15,
                        help="A target word appears in at least this many full-train sentences")
    parser.add_argument('--val-min-repeats', type=int, default=10,
                        help="Each val/test word must appear in this many chosen training sentences")
    parser.add_argument('--max-unknown', type=float, default=0.0,
                        help="Share of a val/test sentence's words allowed to fall short of that")
    parser.add_argument('--train-min-share', type=float, default=0.75,
                        help="Share of a train sentence's words that must be in the vocabulary")
    parser.add_argument('--target-repeats', type=int, default=20,
                        help="Train sentences wanted per target word; more adds nothing to the score")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Standard library and this repository only, so only Python's version matters
    check_environment(packages=())

    from configs.subset_vivit_ctc_config import config
    run(args.csv or config.csv_path, config.train_val_test_split_path, config.valid_samples_path,
        args.out, args.compare, args)


if __name__ == "__main__":
    main()
