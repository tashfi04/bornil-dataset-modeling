"""Train the ViViT-CTC model with the reduced Kaggle test config.

Run scripts/train_bpe_tokenizer.py first; this script expects the tokenizer at
config.bpe_tokenizer_path to already exist.
"""
import os
import sys

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.test_vivit_ctc_config import config
from src.training.vivit_trainer import ViViTTrainer
from src.data_loader import check_dataset_health
from src.utils.run_info import print_run_summary


def main():
    print_run_summary(config, "ViViT-CTC Training (Kaggle test)")

    if not check_dataset_health(config):
        print("Dataset health check failed! Fix paths before training.")
        exit(1)
    print("Dataset health check passed!")

    trainer = ViViTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
