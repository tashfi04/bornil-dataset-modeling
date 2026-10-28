"""Train the ViViT-CTC model on the ~2,000-sentence subset.

Run scripts/train_bpe_tokenizer.py first; this script expects the tokenizer at
config.bpe_tokenizer_path to already exist. With the default config, build the
frame cache first:

    python scripts/preprocess_cache_frames.py --config subset --csv path/to/subset.csv
    python scripts/subset_vivit_ctc.py --csv path/to/subset.csv
    python scripts/subset_vivit_ctc.py --csv path/to/subset.csv --no-cache
    python scripts/subset_vivit_ctc.py --csv path/to/subset.csv --frozen-backbone
"""
import os
import sys
import argparse

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.subset_vivit_ctc_config import config, use_frozen_backbone
from src.training.vivit_trainer import ViViTTrainer
from src.data_loader import check_dataset_health
from src.utils.run_info import print_run_summary


def main():
    parser = argparse.ArgumentParser(description="Train ViViT-CTC on the subset")
    parser.add_argument('--csv', default=None,
                        help="Subset CSV (defaults to the config's csv_path)")
    parser.add_argument('--no-cache', action='store_true',
                        help="Decode the videos instead of reading the frame cache")
    parser.add_argument('--frozen-backbone', action='store_true',
                        help="Use the prod config's frozen-backbone recipe; outputs "
                             "go to vivit_ctc_subset_frozen")
    args = parser.parse_args()

    if args.csv:
        config.csv_path = args.csv
    if args.no_cache:
        config.cached_frames_path = None
    if args.frozen_backbone:
        use_frozen_backbone(config)

    print_run_summary(config, "ViViT-CTC Subset Training")

    if not check_dataset_health(config):
        print("Dataset health check failed! Fix paths before training.")
        exit(1)
    print("Dataset health check passed!")

    trainer = ViViTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
