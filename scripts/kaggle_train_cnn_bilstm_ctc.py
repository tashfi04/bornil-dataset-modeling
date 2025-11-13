import os
import sys

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.test_cnn_bilstm_ctc_config import config  # Use test config
from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer
from src.data_loader import check_dataset_health

def main():
    print("=== CNN-BiLSTM-CTC Training ===")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Frames per video: {config.num_frames}")
    print(f"Epochs: {config.num_epochs}")

    if not check_dataset_health(config):
        print("Dataset health check failed! Fix paths before training.")
        exit(1)
    else:
        print("Dataset health check passed!")

    trainer = CNNBiLSTMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
