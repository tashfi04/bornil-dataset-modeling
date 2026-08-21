"""Decode every video once and record its frame count.

This is the expensive step (a couple of hours for the full dataset). The result
depends only on the videos, so it is written to config.video_stats_path and
reused by scripts/validate_dataset.py for every model, which then finishes in
seconds instead of repeating the decode.

    python scripts/scan_videos.py                 # scan anything not yet scanned
    python scripts/scan_videos.py --rescan        # start over
    python scripts/scan_videos.py --limit 200     # quick trial
"""
import os
import sys
import time
import argparse

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.utils.video_stats import (
    covered_recordings, load_video_stats, save_video_stats, scan_videos, summarize,
)


def main():
    parser = argparse.ArgumentParser(description="Scan videos for frame counts")
    parser.add_argument('--out', default=None,
                        help="Stats file (defaults to config.video_stats_path)")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--rescan', action='store_true',
                        help="Ignore any existing stats and scan everything again")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only scan the first N rows")
    args = parser.parse_args()

    out_path = args.out or config.video_stats_path
    df = pd.read_csv(config.csv_path)
    if args.limit:
        df = df.iloc[:args.limit]

    existing = None if args.rescan else load_video_stats(out_path)
    already = covered_recordings(existing) if existing else set()

    rows = []
    for _, row in df.iterrows():
        recording = row['recording']
        if recording in already:
            continue
        rows.append((recording,
                     os.path.join(config.chunk_base_path, row['chunk_path'], recording)))

    print("=== VIDEO SCAN ===")
    print(f"Rows in CSV: {len(df)}")
    print(f"Already scanned: {len(already)}")
    print(f"To scan now: {len(rows)}   workers: {args.workers}")
    print(f"Stats file: {out_path}")

    if not rows:
        print("\nNothing to scan; stats file is already complete.")
        stats = existing
    else:
        started = time.time()
        fresh = scan_videos(rows, workers=args.workers, csv_path=config.csv_path,
                            progress=lambda it, total: tqdm(it, total=total, desc="Scanning"))
        elapsed = time.time() - started

        # Merge into anything already scanned so an interrupted run can resume
        stats = fresh
        if existing:
            merged_frames = dict(existing['frames'])
            merged_frames.update(fresh['frames'])
            merged_undecodable = dict(existing.get('undecodable', {}))
            merged_undecodable.update(fresh['undecodable'])
            stats = {
                'version': fresh['version'],
                'csv_path': config.csv_path,
                'scanned': len(merged_frames) + len(merged_undecodable),
                'frames': merged_frames,
                'missing': sorted(set(existing.get('missing', [])) | set(fresh['missing'])),
                'undecodable': merged_undecodable,
            }

        save_video_stats(out_path, stats)
        print(f"\nScanned {len(rows)} videos in {elapsed/60:.1f} min")

    print("\n=== RESULTS ===")
    print(f"Decoded OK: {len(stats['frames'])}")
    print(f"Missing files: {len(stats.get('missing', []))}")
    print(f"Undecodable: {len(stats.get('undecodable', {}))}")

    info = summarize(stats['frames'].values())
    if info:
        print(f"\nFrame counts over {info['count']} videos:")
        print(f"  min={info['min']} max={info['max']} mean={info['mean']:.0f}")
        print("  " + "  ".join(f"p{p}={v}" for p, v in info['percentiles'].items()))

    print(f"\nWrote {out_path}")
    print("scripts/validate_dataset.py will now reuse this instead of decoding.")


if __name__ == "__main__":
    main()
