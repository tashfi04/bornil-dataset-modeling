import os
import sys

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.test_cnn_lstm_ctc_config import config  # Use test config
from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer

def main():
    print("=== QUICK TEST RUN ===")
    print(f"Batch size: {config.batch_size}")
    print(f"Max frames: {config.max_frames}")
    print(f"Epochs: {config.num_epochs}")

    trainer = CNNBiLSTMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
