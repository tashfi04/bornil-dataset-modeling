"""Validate every sample once and record which ones are safe to train on.

The dataset is static, so this runs ahead of training and writes its verdict to
config.valid_samples_path. The data loader then trains only on the recordings
listed there, which keeps the training run itself free of surprises.

Each sample is rejected for any of:
  - the video file is missing
  - the video cannot be decoded, or decodes to zero frames
  - the text is empty after normalization
  - the target is longer than the model's CTC time axis

Frame counts come from an actual decode. cv2.CAP_PROP_FRAME_COUNT is unreliable
on these webm files because they lack duration metadata.

    python scripts/validate_dataset.py --model vivit
    python scripts/validate_dataset.py --model cnn_bilstm
    python scripts/validate_dataset.py --model vivit --quick   # skip decoding
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_utils import (
    normalize_text, text_to_int, text_to_bpe_ids, load_bpe_tokenizer,
)
from src.utils.ctc_limits import ctc_time_steps, target_token_limit, required_ctc_steps

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


def check_one(task):
    """Worker: decide whether a single recording is usable."""
    recording, video_path, quick = task

    if not os.path.exists(video_path):
        return {'recording': recording, 'ok': False, 'reason': 'missing_file'}

    if quick:
        return {'recording': recording, 'ok': True, 'frames': None}

    try:
        frames = count_frames(video_path)
    except Exception as exc:
        return {'recording': recording, 'ok': False,
                'reason': f'undecodable ({type(exc).__name__}: {exc})'}

    if frames == 0:
        return {'recording': recording, 'ok': False, 'reason': 'zero_frames'}

    return {'recording': recording, 'ok': True, 'frames': frames}


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
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--quick', action='store_true',
                        help="Skip decoding; only check existence and text length")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only check the first N rows (debugging)")
    args = parser.parse_args()

    if args.model == 'vivit':
        from configs.vivit_ctc_config import config
    else:
        from configs.cnn_bilstm_ctc_config import config

    tokenization = getattr(config, 'tokenization_type', 'character')
    tokenizer = None
    char_to_id = None
    if tokenization == 'bpe':
        tokenizer = load_bpe_tokenizer(config.bpe_tokenizer_path)
    else:
        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            char_to_id = json.load(f)['char_to_id']

    # The CTC time axis bounds how many target tokens a sample may have
    ctc_steps = ctc_time_steps(config)
    token_limit = target_token_limit(config)

    df = pd.read_csv(config.csv_path)
    if args.limit:
        df = df.iloc[:args.limit]

    print("=== DATASET VALIDATION ===")
    print(f"Model: {args.model}  tokenization: {tokenization}")
    print(f"CTC time steps: {ctc_steps}   target token limit: {token_limit}")
    print(f"Rows to check: {len(df)}   workers: {args.workers}")
    if args.quick:
        print("QUICK MODE: videos are not decoded, so corrupt files will not be caught")

    # Text checks are cheap, so do them before spending time on decoding
    rejected = {}
    target_lengths = {}
    to_decode = []
    for _, row in df.iterrows():
        recording = row['recording']
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
        video_path = os.path.join(config.chunk_base_path, row['chunk_path'], recording)
        to_decode.append((recording, video_path, args.quick))

    print(f"\nPassed text checks: {len(to_decode)}   rejected: {len(rejected)}")

    valid = []
    frames_by_recording = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_one, t) for t in to_decode]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Checking videos"):
            res = fut.result()
            if res['ok']:
                valid.append(res['recording'])
                if res.get('frames'):
                    frames_by_recording[res['recording']] = res['frames']
            else:
                rejected[res['recording']] = res['reason']
    elapsed = time.time() - started
    frame_counts = sorted(frames_by_recording.values())

    compressed = getattr(config, 'compressed_frames', None)

    print(f"\n=== RESULTS ({elapsed/60:.1f} min) ===")
    print(f"Valid samples: {len(valid)} / {len(df)} ({100*len(valid)/len(df):.2f}%)")
    print(f"Rejected: {len(rejected)}")

    reasons = {}
    for reason in rejected.values():
        key = reason.split(' (')[0]
        reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    if frame_counts:
        n = len(frame_counts)

        def pct(p):
            return frame_counts[min(n - 1, int(round((n - 1) * p / 100)))]

        print(f"\nDecoded frame counts over {n} videos:")
        print(f"  min={frame_counts[0]} max={frame_counts[-1]} "
              f"mean={sum(frame_counts)/n:.0f}")
        print("  " + "  ".join(f"p{p}={pct(p)}" for p in (5, 25, 50, 75, 90, 95, 99)))

        below_read = sum(1 for f in frame_counts if f < config.num_frames)
        print(f"  shorter than num_frames ({config.num_frames}): {below_read} "
              f"({100*below_read/n:.1f}%) - these are stretched, not padded")
        if compressed:
            below_compressed = sum(1 for f in frame_counts if f < compressed)
            print(f"  shorter than compressed_frames ({compressed}): "
                  f"{below_compressed} ({100*below_compressed/n:.1f}%) - upsampled in time")

        # Fewer frames than target tokens means less than one frame per token,
        # which is a data quality question rather than a CTC error
        starved = [r for r, f in frames_by_recording.items()
                   if f < target_lengths.get(r, 0)]
        if starved:
            print(f"  WARNING: {len(starved)} videos have fewer frames than target "
                  f"tokens; check these recordings for truncation")

    out_path = args.out or config.valid_samples_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        'model': args.model,
        'tokenization': tokenization,
        'ctc_steps': ctc_steps,
        'token_limit': token_limit,
        'num_frames': config.num_frames,
        'frame_size': list(config.frame_size),
        'quick': args.quick,
        'total_rows': len(df),
        'valid': sorted(valid),
        'rejected': rejected,
        # Kept so the frame distribution can be analysed without decoding again
        'frames': frames_by_recording,
        'target_lengths': {r: target_lengths[r] for r in valid if r in target_lengths},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(valid)} valid recordings to {out_path}")
    print("Set `valid_samples_path` in your config to train on exactly this list.")


if __name__ == "__main__":
    main()
