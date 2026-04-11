import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.utils.text_utils import train_bpe_tokenizer

if __name__ == "__main__":
    print("Training BPE Tokenizer")
    output_path = os.path.join(config.repo_root, "data", "bpe_tokenizer.json")
    vocab_size = config.bpe_vocab_size
    train_bpe_tokenizer(config.csv_path, output_path, vocab_size=vocab_size)
