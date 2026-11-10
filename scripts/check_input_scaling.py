"""Which pixel scaling the pretrained ViViT responds to, on our own clips.

The Kinetics-400 checkpoint was trained on one input scaling, and two
candidates fit what is published about it (src/utils/input_scaling.py). This
passes the same clips through the complete pretrained model, Kinetics
classifier included, at each candidate and at the 0..1 that every run before
2026-11-09 used. Nothing is trained or saved.

Every clip shows one person signing to a camera, so at the scaling the model
was trained on its predictions should be confident and mostly agree across
clips. At a wrong scaling they should be less sure and more scattered.

Everything except the scaling is the same for every candidate: 32 frames
spread over the whole clip (pretraining used 32 consecutive frames at stride
2), and frames resized to a square, as our pipeline does, rather than
center-cropped. So absolute confidence is lower than on Kinetics; only the
comparison between scalings matters.

    python scripts/check_input_scaling.py --config subset --csv /kaggle/working/subset.csv
"""
import os
import sys
import json
import math
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import VivitForVideoClassification

from src.utils.environment import check_environment
from src.utils.frame_cache import cache_path, read_cached_frames
from src.utils.input_scaling import scale_frames

CANDIDATES = ('zero_one', 'minus_one_one', 'hf_vivit')


def load_clip(path, num_frames, size):
    """`num_frames` frames spread evenly over a cached clip, as (T, C, H, W) in 0..1."""
    frames = read_cached_frames(path)
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    picked = []
    for i in indices:
        frame = frames[i]
        if frame.shape[:2] != (size, size):
            frame = cv2.resize(frame, (size, size))
        picked.append(frame)
    clip = np.stack(picked).astype(np.float32) / 255.0      # (T, H, W, C), RGB
    return torch.from_numpy(clip).permute(0, 3, 1, 2)       # (T, C, H, W)


def classify(model, clips, scaling, batch_size, device):
    """Softmax over the 400 Kinetics classes for every clip, as (clips, 400)."""
    probs = []
    for start in range(0, len(clips), batch_size):
        batch = torch.stack(clips[start:start + batch_size]).to(device)  # (B, T, C, H, W)
        batch = scale_frames(batch, scaling, channel_dim=2)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(pixel_values=batch).logits
        probs.append(logits.float().softmax(dim=-1).cpu())
    return torch.cat(probs)


def main():
    parser = argparse.ArgumentParser(description="Compare input scalings on the pretrained ViViT")
    parser.add_argument('--config', choices=['subset', 'test', 'prod', 'overfit'], default='subset')
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--clips', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    check_environment()

    if args.config == 'subset':
        from configs.subset_vivit_ctc_config import config
    elif args.config == 'test':
        from configs.test_vivit_ctc_config import config
    elif args.config == 'overfit':
        from configs.overfit_vivit_ctc_config import config
    else:
        from configs.vivit_ctc_config import config
    if args.csv:
        config.csv_path = args.csv

    cache_root = getattr(config, 'cached_frames_path', None)
    if not cache_root:
        sys.exit(f"The {args.config} config has no frame cache (cached_frames_path); "
                 f"this check reads clips from one.")

    df = pd.read_csv(config.csv_path)
    with open(config.train_val_test_split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    rows = df[df['recording'].map(split) == args.split]
    paths = []
    for _, row in rows.iterrows():
        path = cache_path(cache_root, row['chunk_path'], row['recording'])
        if os.path.exists(path):
            paths.append(path)
        if len(paths) == args.clips:
            break
    if not paths:
        sys.exit(f"No {args.split} clip of {config.csv_path} is in the cache at {cache_root}.")

    device = config.device
    model_name = getattr(config, 'vivit_model_name', 'google/vivit-b-16x2-kinetics400')
    model = VivitForVideoClassification.from_pretrained(model_name).to(device).eval()
    num_frames, height, width = model.config.video_size
    if height != width:
        sys.exit(f"{model_name} expects {height}x{width} frames; this check assumes square ones.")
    num_classes = model.config.num_labels

    clips = [load_clip(path, num_frames, height) for path in paths]
    print(f"{len(clips)} {args.split} clips, {num_frames} frames each at {height}x{width}, "
          f"through the full {model_name} with its {num_classes}-class head\n")

    probs = {name: classify(model, clips, name, args.batch_size, device) for name in CANDIDATES}

    max_entropy = math.log(num_classes)
    top1 = {}
    print(f"{'scaling':<15} {'pixel range':<12} {'mean top-1 prob':>16} {'mean entropy':>14} "
          f"{'distinct top-1':>15} {'most common top-1':>24}")
    ranges = {'zero_one': '0..1', 'minus_one_one': '-1..1', 'hf_vivit': '-3..1'}
    for name in CANDIDATES:
        p = probs[name]
        best, best_index = p.max(dim=1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
        counts = Counter(best_index.tolist())
        cls, n = counts.most_common(1)[0]
        top1[name] = best_index
        print(f"{name:<15} {ranges[name]:<12} {best.mean().item():>16.3f} "
              f"{entropy.mean().item() / max_entropy:>13.0%} {len(counts):>15} "
              f"{f'class {cls} in {n}/{len(clips)}':>24}")
    print(f"(entropy is shown as a share of the maximum, ln {num_classes}: lower means surer)")

    # Follow the class the surest scaling picks most often through every scaling
    surest = max(CANDIDATES, key=lambda name: probs[name].max(dim=1).values.mean().item())
    cls = Counter(top1[surest].tolist()).most_common(1)[0][0]
    label = model.config.id2label.get(cls, str(cls))
    print(f"\nClass {cls} ({label}), the most common top-1 at {surest}: mean probability "
          + ", ".join(f"{name} {probs[name][:, cls].mean().item():.3f}" for name in CANDIDATES))

    print("\nSame top-1 class for the same clip:")
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1:]:
            same = (top1[a] == top1[b]).float().mean().item()
            print(f"  {a} vs {b}: {same:.0%} of clips")

    print("\n=== Reading it ===")
    print("The scaling the model was trained on should give the highest top-1")
    print("probability, the lowest entropy and the fewest distinct top-1 classes")
    print("(the clips all show the same activity). If 0..1 is clearly worse than the")
    print("best candidate, the old scaling was hurting the pretrained features. If all")
    print("three look alike, the scaling mattered little.")


if __name__ == "__main__":
    main()
