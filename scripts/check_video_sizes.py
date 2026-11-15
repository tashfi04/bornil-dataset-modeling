"""What sizes and shapes the source videos have, before any resizing.

The loader and the frame cache resize every frame straight to a square, so a
wide video is squeezed sideways and a tall one is stretched. This reads the
first frame of a random sample of videos and reports how mixed they are and
how much each shape is distorted by that resize. Counts only; nothing is
saved. It needs no GPU.

    python scripts/check_video_sizes.py --config subset --csv path/to/subset.csv
    python scripts/check_video_sizes.py --config prod --clips 500
"""
import os
import sys
import random
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import pandas as pd


def first_frame_size(path):
    """(width, height) of a video's first decoded frame, or None if it cannot be read."""
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    return frame.shape[1], frame.shape[0]


def shape_name(width, height):
    ratio = width / height
    if ratio > 1.05:
        return 'wide (landscape)'
    if ratio < 0.95:
        return 'tall (portrait)'
    return 'square'


def percentiles(values, points=(5, 25, 50, 75, 95)):
    values = sorted(values)
    return '  '.join(
        f"p{p}={values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]:g}" for p in points)


def main():
    parser = argparse.ArgumentParser(description="Report source video sizes")
    parser.add_argument('--config', choices=['subset', 'test', 'prod', 'overfit'], default='subset')
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--clips', type=int, default=300, help="Videos to sample")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.config == 'subset':
        from configs.subset_vivit_ctc_config import config
    elif args.config == 'test':
        from configs.test_vivit_ctc_config import config
    elif args.config == 'overfit':
        from configs.overfit_vivit_ctc_config import config
    else:
        from configs.vivit_ctc_config import config

    df = pd.read_csv(args.csv or config.csv_path)
    rows = list(df[['chunk_path', 'recording']].itertuples(index=False))
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.clips]

    sizes, unreadable = [], 0
    for chunk, recording in rows:
        size = first_frame_size(os.path.join(config.chunk_base_path, chunk, recording))
        if size is None:
            unreadable += 1
        else:
            sizes.append(size)
    if not sizes:
        sys.exit(f"None of the {len(rows)} sampled videos under {config.chunk_base_path} could be read.")

    target = config.frame_size[0]
    print(f"{len(sizes)} videos read ({unreadable} unreadable), sampled from {len(df):,} rows\n")

    print("Most common sizes (width x height):")
    for (width, height), count in Counter(sizes).most_common(10):
        print(f"  {width:>5} x {height:<5} {count:>4}  ({100 * count / len(sizes):.0f}%)")
    print(f"  distinct sizes: {len(set(sizes))}")

    print("\nShape:")
    shapes = Counter(shape_name(w, h) for w, h in sizes)
    for name, count in shapes.most_common():
        print(f"  {name:<18} {count:>4}  ({100 * count / len(sizes):.0f}%)")

    print(f"\nShorter side, pixels:        {percentiles([min(w, h) for w, h in sizes])}")
    print(f"Shrink factor to {target} px:     "
          f"{percentiles([round(max(w, h) / target, 1) for w, h in sizes])}")

    # Resizing (w, h) straight to a square scales width by target/w and height
    # by target/h, so shapes come out h/w times as wide as they should be
    print(f"\nWidth distortion after the square resize (1.00 = undistorted, below 1 = squeezed "
          f"sideways, above 1 = stretched sideways):")
    print(f"  {percentiles([round(h / w, 2) for w, h in sizes])}")
    for name in shapes:
        group = [round(h / w, 2) for w, h in sizes if shape_name(w, h) == name]
        print(f"  {name:<18} {percentiles(group, (5, 50, 95))}")

    print("\n=== Reading it ===")
    print("If nearly all videos share one shape, the square resize distorts them all")
    print("alike and the model can adapt. If wide and tall videos are mixed, the same")
    print("hand shape looks different depending on the device, and resizing with the")
    print("shape kept (padding the short side) is worth doing at the next cache build.")
    print("A large shrink factor with OpenCV's default filter also loses fine detail")
    print("such as fingers; area averaging avoids that.")


if __name__ == "__main__":
    main()
