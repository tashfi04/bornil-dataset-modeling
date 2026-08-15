"""Train the CNN-BiLSTM-CTC baseline with the reduced Kaggle test config.

Run scripts/preprocess_text.py first; this script expects the character
vocabulary at config.vocab_path to already exist.
"""
import os
import sys

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.test_cnn_bilstm_ctc_config import config
from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer
from src.data_loader import check_dataset_health
from src.utils.run_info import print_run_summary


def main():
    print_run_summary(config, "CNN-BiLSTM-CTC Training (Kaggle test)")

    if not check_dataset_health(config):
        print("Dataset health check failed! Fix paths before training.")
        exit(1)
    print("Dataset health check passed!")

    trainer = CNNBiLSTMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
