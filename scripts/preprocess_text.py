import sys
import os
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from src.utils.text_utils import build_vocab_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the character vocabulary")
    parser.add_argument("--config", choices=["prod", "test"], default="prod",
                        help="test reads the dataset from the Kaggle test config's paths")
    args = parser.parse_args()

    if args.config == "test":
        from configs.test_vivit_ctc_config import config
    else:
        from configs.base_config import config

    build_vocab_from_csv(config.csv_path, config.vocab_path)
    print("Vocabulary building complete!")
