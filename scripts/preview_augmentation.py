"""Save augmented training clips as images, to check the augmentation by eye.

Each image shows one clip: a row of evenly spaced frames as recorded, then the
same frame positions after several random augmentations. Look for crops that
cut off a hand or the face, and for lighting changes that hide the hands. The
time warp shows as the augmented rows reaching a sign a little earlier or later.

    python scripts/preview_augmentation.py --config subset --csv path/to/subset.csv
    python scripts/preview_augmentation.py --clips 8 --draws 4

Images go to <model_output_dir>/augmentation_preview/.
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from src.utils.environment import check_environment


def frame_strip(video, columns):
    """`columns` evenly spaced frames of a (C, T, H, W) clip side by side, as BGR uint8."""
    picks = np.linspace(0, video.size(1) - 1, columns).round().astype(int)
    frames = video[:, picks].permute(1, 2, 3, 0).numpy()     # (columns, H, W, C)
    strip = np.concatenate(list(frames), axis=1)
    return cv2.cvtColor((strip * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Preview training-clip augmentation")
    parser.add_argument('--model', choices=['vivit', 'cnn_bilstm'], default='vivit')
    parser.add_argument('--config', choices=['prod', 'test', 'subset'], default='prod')
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--no-cache', action='store_true',
                        help="Decode the videos instead of reading the frame cache")
    parser.add_argument('--clips', type=int, default=6, help="Clips to preview")
    parser.add_argument('--draws', type=int, default=3, help="Augmented versions of each clip")
    parser.add_argument('--columns', type=int, default=8, help="Frames shown per row")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    check_environment()

    if args.model == 'vivit':
        if args.config == 'test':
            from configs.test_vivit_ctc_config import config
        elif args.config == 'subset':
            from configs.subset_vivit_ctc_config import config
        else:
            from configs.vivit_ctc_config import config
    else:
        if args.config == 'subset':
            parser.error("CNN-BiLSTM has only the prod and test configs")
        if args.config == 'test':
            from configs.test_cnn_bilstm_ctc_config import config
        else:
            from configs.cnn_bilstm_ctc_config import config
    if not getattr(config, 'augment', False):
        print(f"The {args.model} {args.config} config trains without augmentation unless "
              f"its script is given --augment true; previewing its augment_* settings anyway.")
    if args.csv:
        config.csv_path = args.csv
    if args.no_cache:
        config.cached_frames_path = None

    from src.data_loader import get_data_loaders
    from src.utils.augmentation import check_augmentation

    check_augmentation(config)
    torch.manual_seed(args.seed)
    train_loader, _, _ = get_data_loaders(config)
    dataset = train_loader.dataset

    out_dir = os.path.join(config.model_output_dir, 'augmentation_preview')
    os.makedirs(out_dir, exist_ok=True)
    for index in torch.randperm(len(dataset))[:args.clips].tolist():
        dataset.augment = False
        recorded = dataset[index]
        dataset.augment = True
        rows = [frame_strip(recorded['video'], args.columns)]
        rows += [frame_strip(dataset[index]['video'], args.columns) for _ in range(args.draws)]

        name = os.path.splitext(os.path.basename(recorded['video_path']))[0]
        path = os.path.join(out_dir, f"{name}.jpg")
        if not cv2.imwrite(path, np.concatenate(rows, axis=0)):
            raise RuntimeError(f"Could not write {path}")
        print(f"Wrote {path}: top row as recorded, then {args.draws} augmented versions")


if __name__ == "__main__":
    main()
