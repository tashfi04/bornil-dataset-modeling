import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from src.utils.text_utils import build_vocab_from_csv
from configs.base_config import config

if __name__ == "__main__":
    build_vocab_from_csv(config.csv_path, config.vocab_path)
    print("Vocabulary building complete!")
