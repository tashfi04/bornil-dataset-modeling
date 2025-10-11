import os
import sys
import GPUtil

GPUtil.showUtilization()

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.cnn_bilstm_ctc_config import config
from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer
from src.data_loader import check_dataset_health

def main():
    if not check_dataset_health(config):
        print("❌ Dataset health check failed! Fix paths before training.")
        exit(1)
    else:
        print("✅ Dataset health check passed!")

    trainer = CNNBiLSTMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
