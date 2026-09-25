import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_utils import train_bpe_tokenizer, load_bpe_tokenizer, normalize_text
from src.utils.ctc_limits import VIVIT_TUBELET_FRAMES

def tokenize(config, vocab_size):
    print("Training BPE Tokenizer")

    # Train tokenizer
    output_path = config.bpe_tokenizer_path

    train_bpe_tokenizer(config.csv_path, output_path, vocab_size=vocab_size)

    # Load the trained tokenizer to analyze token counts
    tokenizer = load_bpe_tokenizer(output_path)

    # Load and normalize all texts
    df = pd.read_csv(config.csv_path)
    texts = df['text'].astype(str).apply(normalize_text).tolist()

    # Calculate token counts per sentence
    token_counts = []
    for text in texts:
        encoded = tokenizer.encode(text)
        # Add 1 to account for blank shift (even though we don't shift here, the number of tokens is the same)
        token_counts.append(len(encoded.ids))


    print("\n=== BPE Tokenization Statistics ===")
    print(f"Vocab size: {vocab_size}")
    print(f"Number of sentences analyzed: {len(token_counts)}")
    print(f"Min tokens per sentence: {min(token_counts)}")
    print(f"Max tokens per sentence: {max(token_counts)}")
    print(f"Mean tokens: {np.mean(token_counts):.2f}")
    print(f"Median tokens: {np.median(token_counts):.2f}")
    print(f"95th percentile: {np.percentile(token_counts, 95):.0f}")
    print(f"99th percentile: {np.percentile(token_counts, 99):.0f}")

    # Percentiles and counts
    total_samples = len(token_counts)
    p50 = np.percentile(token_counts, 50)
    p75 = np.percentile(token_counts, 75)
    p90 = np.percentile(token_counts, 90)
    p95 = np.percentile(token_counts, 95)
    p99 = np.percentile(token_counts, 99)

    print(f"\nPercentiles:")
    print(f"  50th: {p50:.0f} tokens (covers {int(0.5*total_samples)} sentences)")
    print(f"  75th: {p75:.0f} tokens (covers {int(0.75*total_samples)} sentences)")
    print(f"  90th: {p90:.0f} tokens (covers {int(0.9*total_samples)} sentences)")
    print(f"  95th: {p95:.0f} tokens (covers {int(0.95*total_samples)} sentences)")
    print(f"  99th: {p99:.0f} tokens (covers {int(0.99*total_samples)} sentences)")

    # Compare with original character lengths
    char_lengths = df['text'].str.len().tolist()
    avg_compression = np.mean(char_lengths) / np.mean(token_counts)
    print(f"\nCompression ratio (chars → BPE tokens): {avg_compression:.2f}x")

    # CTC needs at least one output step per target token, and ViViT emits
    # compressed_frames // tubelet steps, so the token limit that covers a given
    # share of sentences fixes how many frames ViViT must be given
    print(f"\nCTC time budget (ViViT emits compressed_frames // {VIVIT_TUBELET_FRAMES} steps):")
    for share in (90, 95, 99):
        tokens = int(np.ceil(np.percentile(token_counts, share)))
        print(f"  {share}% of sentences: max_bpe_tokens = {tokens}, "
              f"compressed_frames >= {tokens * VIVIT_TUBELET_FRAMES}")

    limit = config.max_bpe_tokens
    kept = sum(1 for count in token_counts if count <= limit)
    print(f"  Configured max_bpe_tokens = {limit} keeps {kept} sentences "
          f"({100 * kept / total_samples:.2f}%)")
    print("  Adjacent repeated tokens need extra steps, so the exact count comes "
          "from scripts/validate_dataset.py")

def main():
    parser = argparse.ArgumentParser(description="Script for BPE tokenization")
    parser.add_argument("--config", choices=["prod", "test"], default="prod",
                        help="test reads the dataset from the Kaggle test config's paths")
    parser.add_argument("--bpe_vocab_size", type=int, default=None,
                        help="Size of the BPE vocabulary (defaults to the config's)")
    args = parser.parse_args()

    if args.config == "test":
        from configs.test_vivit_ctc_config import config
    else:
        from configs.base_config import config

    tokenize(config, args.bpe_vocab_size or config.bpe_vocab_size)

if __name__ == "__main__":
    main()
