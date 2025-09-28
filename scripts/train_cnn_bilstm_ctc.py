import os
import sys

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.cnn_lstm_ctc_config import config
from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer

def main():
    trainer = CNNBiLSTMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
