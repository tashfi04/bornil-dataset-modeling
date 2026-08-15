"""Decode each video once, sample and resize its frames, and cache them as JPEG.

The source clips are 400-1200 frame webm files that must be decoded
sequentially, which dominates epoch time. Caching the sampled frames removes
that cost from training.

Start with a dry run to check the projected disk cost before writing anything:

    python scripts/preprocess_cache_frames.py --dry-run --sample 200

Then, if the projection fits your storage budget:

    python scripts/preprocess_cache_frames.py --out /kaggle/working/frame_cache

Tune size with --frames / --size / --quality. Defaults follow the ViViT config.
"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.vivit_ctc_config import config
from src.data_loader import sample_frames
from src.utils.frame_cache import cache_path, encode_frames

# Each worker is already a separate process, so keep OpenCV single-threaded
cv2.setNumThreads(0)


def read_and_sample(video_path, num_frames, size, strategy, segments):
    """Decode a video, sample to num_frames, resize to size. Returns RGB frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (size[1], size[0]))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    return sample_frames(frames, num_frames, strategy, segments), len(frames)


def process_one(task):
    """Worker: decode one video and (optionally) write its cache file."""
    (video_path, out_path, num_frames, size, strategy, segments,
     quality, dry_run) = task

    started = time.time()
    try:
        frames, raw_count = read_and_sample(video_path, num_frames, size, strategy, segments)
        blob = encode_frames(frames, quality=quality)
        if not dry_run:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(blob)
        return {
            'ok': True, 'bytes': len(blob), 'frames': len(frames),
            'raw_frames': raw_count, 'seconds': time.time() - started,
            'path': video_path,
        }
    except Exception as exc:
        return {'ok': False, 'error': f"{type(exc).__name__}: {exc}",
                'path': video_path, 'seconds': time.time() - started}


def human(num_bytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(description="Cache pre-sampled video frames")
    parser.add_argument('--out', default=os.path.join(config.repo_root, 'data', 'frame_cache'),
                        help="Cache root directory")
    parser.add_argument('--frames', type=int, default=config.num_frames,
                        help="Frames to keep per video")
    parser.add_argument('--size', type=int, default=config.frame_size[0],
                        help="Square frame size in pixels")
    parser.add_argument('--quality', type=int, default=90, help="JPEG quality (1-100)")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--dry-run', action='store_true',
                        help="Decode a sample and project total size WITHOUT writing")
    parser.add_argument('--sample', type=int, default=200,
                        help="Videos to sample in a dry run")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only process the first N videos (debugging)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-encode clips that are already cached")
    args = parser.parse_args()

    size = (args.size, args.size)
    strategy = getattr(config, 'sampling_strategy', 'uniform')
    segments = getattr(config, 'sampling_segments', 3)

    df = pd.read_csv(config.csv_path)
    print(f"Dataset rows: {len(df)}")

    rows = df
    if args.limit:
        rows = rows.iloc[:args.limit]
    if args.dry_run:
        n = min(args.sample, len(rows))
        rows = rows.sample(n=n, random_state=config.random_seed)
        print(f"DRY RUN on {n} randomly sampled videos - nothing will be written")

    print(f"Frames/video: {args.frames}   size: {size}   JPEG quality: {args.quality}")
    print(f"Sampling: {strategy} ({segments} segments)   workers: {args.workers}")
    if not args.dry_run:
        print(f"Writing to: {args.out}")

    tasks = []
    skipped = 0
    for _, row in rows.iterrows():
        video_path = os.path.join(config.chunk_base_path, row['chunk_path'], row['recording'])
        out_path = cache_path(args.out, row['chunk_path'], row['recording'])
        if not args.dry_run and not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        tasks.append((video_path, out_path, args.frames, size, strategy,
                      segments, args.quality, args.dry_run))

    if skipped:
        print(f"Skipping {skipped} already-cached clips (use --overwrite to redo)")
    if not tasks:
        print("Nothing to do.")
        return

    results = []
    failures = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching"):
            res = fut.result()
            if res['ok']:
                results.append(res)
            else:
                failures.append(res)
    elapsed = time.time() - started

    if not results:
        print("\nAll videos failed. First few errors:")
        for f in failures[:5]:
            print(f"  {f['path']}: {f['error']}")
        sys.exit(1)

    total_bytes = sum(r['bytes'] for r in results)
    mean_bytes = total_bytes / len(results)
    mean_raw = sum(r['raw_frames'] for r in results) / len(results)
    mean_secs = sum(r['seconds'] for r in results) / len(results)

    print(f"\n=== Results ({len(results)} ok, {len(failures)} failed) ===")
    print(f"Mean raw frames per video: {mean_raw:.0f}")
    print(f"Mean cached size per video: {human(mean_bytes)}")
    print(f"Mean decode+encode time per video: {mean_secs:.2f}s")
    print(f"Wall clock for this run: {elapsed/60:.1f} min")

    if args.dry_run:
        projected = mean_bytes * len(df)
        # Wall clock scales with the worker pool, not the per-video cost
        projected_secs = mean_secs * len(df) / max(1, args.workers)
        print(f"\n--- PROJECTION for all {len(df)} videos ---")
        print(f"Projected cache size: {human(projected)}")
        print(f"Projected preprocessing time: {projected_secs/3600:.1f} h "
              f"at {args.workers} workers")
        print("\nIf that size does not fit your budget, retry with a smaller")
        print("--size, fewer --frames, or a lower --quality (e.g. 80).")
    else:
        print(f"\nCache written to: {args.out}")
        print("Point training at it by setting `cached_frames_path` in your config.")

    if failures:
        print(f"\n{len(failures)} failures; first few:")
        for f in failures[:10]:
            print(f"  {f['path']}: {f['error']}")


if __name__ == "__main__":
    main()
