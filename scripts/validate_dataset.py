"""Decide which samples a given model can train on, and record the verdict.

The expensive part - decoding every video to get its true frame count - is done
once by scripts/scan_videos.py and cached. This script reads that cache and
applies the cheap, model-specific checks, so it finishes in seconds and can be
re-run freely for each model or config change.

A sample is rejected when:
  - the video file is missing, or could not be decoded
  - the text is empty after normalization
  - the target is longer than `max_bpe_tokens`
  - the target needs more CTC steps than the model produces (adjacent duplicate
    tokens each need a separating blank)

    python scripts/scan_videos.py                      # once, ~2 h
    python scripts/validate_dataset.py --model vivit   # seconds
    python scripts/validate_dataset.py --model cnn_bilstm
"""
import os
import sys
import json
import argparse

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_utils import (
    normalize_text, text_to_int, text_to_bpe_ids, load_bpe_tokenizer,
)
from src.utils.ctc_limits import ctc_time_steps, target_token_limit, required_ctc_steps
from src.utils.video_stats import load_video_stats, summarize


def target_ids(text, tokenization, tokenizer, char_to_id):
    if tokenization == 'bpe':
        return text_to_bpe_ids(text, tokenizer)
    return text_to_int(text, char_to_id)


def main():
    parser = argparse.ArgumentParser(description="Validate the dataset before training")
    parser.add_argument('--model', choices=['vivit', 'cnn_bilstm'], default='vivit',
                        help="Which model's config to validate against")
    parser.add_argument('--out', default=None,
                        help="Where to write the valid-sample list "
                             "(defaults to config.valid_samples_path)")
    parser.add_argument('--stats', default=None,
                        help="Video stats file (defaults to config.video_stats_path)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only check the first N rows (debugging)")
    args = parser.parse_args()

    if args.model == 'vivit':
        from configs.vivit_ctc_config import config
    else:
        from configs.cnn_bilstm_ctc_config import config

    stats_path = args.stats or config.video_stats_path
    stats = load_video_stats(stats_path)
    if stats is None:
        print(f"No usable video stats at {stats_path}.")
        print("Run this first (it decodes every video once, ~2 h):")
        print("    python scripts/scan_videos.py")
        sys.exit(1)

    if stats.get('csv_path') and stats['csv_path'] != config.csv_path:
        print(f"WARNING: stats were scanned from {stats['csv_path']} "
              f"but this config reads {config.csv_path}")

    tokenization = getattr(config, 'tokenization_type', 'character')
    tokenizer = None
    char_to_id = None
    if tokenization == 'bpe':
        tokenizer = load_bpe_tokenizer(config.bpe_tokenizer_path)
    else:
        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            char_to_id = json.load(f)['char_to_id']

    ctc_steps = ctc_time_steps(config)
    token_limit = target_token_limit(config)

    df = pd.read_csv(config.csv_path)
    if args.limit:
        df = df.iloc[:args.limit]

    print("=== DATASET VALIDATION ===")
    print(f"Model: {args.model}  tokenization: {tokenization}")
    print(f"CTC time steps: {ctc_steps}   target token limit: {token_limit}")
    print(f"Rows to check: {len(df)}")
    print(f"Using video stats: {stats_path} ({len(stats['frames'])} decoded)")

    frames_by_recording = stats['frames']
    missing_set = set(stats.get('missing', []))
    undecodable = stats.get('undecodable', {})

    valid = []
    rejected = {}
    target_lengths = {}
    unscanned = []

    for _, row in df.iterrows():
        recording = row['recording']

        if recording in missing_set:
            rejected[recording] = 'missing_file'
            continue
        if recording in undecodable:
            rejected[recording] = f'undecodable ({undecodable[recording]})'
            continue
        if recording not in frames_by_recording:
            unscanned.append(recording)
            rejected[recording] = 'not_scanned'
            continue

        text = normalize_text(str(row['text']))
        if not text.strip():
            rejected[recording] = 'empty_text'
            continue

        ids = target_ids(str(row['text']), tokenization, tokenizer, char_to_id)
        if len(ids) == 0:
            rejected[recording] = 'zero_length_target'
            continue
        if len(ids) > token_limit:
            rejected[recording] = f'target_too_long ({len(ids)} > {token_limit})'
            continue

        # Adjacent duplicate tokens each need a blank between them, so a target
        # can be short enough yet still unalignable
        needed = required_ctc_steps(ids)
        if needed > ctc_steps:
            rejected[recording] = (f'needs_blank_separators '
                                   f'({needed} steps needed > {ctc_steps})')
            continue

        target_lengths[recording] = len(ids)
        valid.append(recording)

    print(f"\n=== RESULTS ===")
    print(f"Valid samples: {len(valid)} / {len(df)} ({100*len(valid)/len(df):.2f}%)")
    print(f"Rejected: {len(rejected)}")

    reasons = {}
    for reason in rejected.values():
        key = reason.split(' (')[0]
        reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    if unscanned:
        print(f"\nWARNING: {len(unscanned)} recordings are not in the stats file. "
              f"Run scripts/scan_videos.py again to cover them.")

    kept_frames = [frames_by_recording[r] for r in valid]
    info = summarize(kept_frames)
    if info:
        compressed = getattr(config, 'compressed_frames', None)
        print(f"\nFrame counts over the {info['count']} valid videos:")
        print(f"  min={info['min']} max={info['max']} mean={info['mean']:.0f}")
        print("  " + "  ".join(f"p{p}={v}" for p, v in info['percentiles'].items()))

        below_read = sum(1 for f in kept_frames if f < config.num_frames)
        print(f"  shorter than num_frames ({config.num_frames}): {below_read} "
              f"({100*below_read/info['count']:.1f}%) - resampled over their real "
              f"length, not zero-padded")
        if compressed:
            below_compressed = sum(1 for f in kept_frames if f < compressed)
            print(f"  shorter than compressed_frames ({compressed}): "
                  f"{below_compressed} ({100*below_compressed/info['count']:.1f}%) "
                  f"- stretched in time")

        # Fewer frames than target tokens is a data quality question, not a CTC error
        starved = [r for r in valid if frames_by_recording[r] < target_lengths[r]]
        if starved:
            print(f"  WARNING: {len(starved)} videos have fewer frames than target "
                  f"tokens; check these recordings for truncation")
            for recording in starved[:5]:
                print(f"    {recording}: {frames_by_recording[recording]} frames, "
                      f"{target_lengths[recording]} tokens")

    out_path = args.out or config.valid_samples_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        'model': args.model,
        'tokenization': tokenization,
        'ctc_steps': ctc_steps,
        'token_limit': token_limit,
        'num_frames': config.num_frames,
        'frame_size': list(config.frame_size),
        'total_rows': len(df),
        'video_stats_path': stats_path,
        'valid': sorted(valid),
        'rejected': rejected,
        'frames': {r: frames_by_recording[r] for r in valid},
        'target_lengths': target_lengths,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(valid)} valid recordings to {out_path}")
    print("Set `valid_samples_path` in your config to train on exactly this list.")


if __name__ == "__main__":
    main()
