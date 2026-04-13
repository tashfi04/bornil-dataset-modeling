import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.utils.text_utils import train_bpe_tokenizer, load_bpe_tokenizer, normalize_text

if __name__ == "__main__":
    print("Training BPE Tokenizer")

    # Train tokenizer
    output_path = config.bpe_tokenizer_path
    vocab_size = config.bpe_vocab_size
    train_bpe_tokenizer(config.csv_path, output_path, vocab_size=vocab_size)

    # Load the trained tokenizer to analyze token counts
    tokenizer = load_bpe_tokenizer(output_path)

    # Load and normalize all texts
    import pandas as pd
    df = pd.read_csv(config.csv_path)
    texts = df['text'].astype(str).apply(normalize_text).tolist()

    # Calculate token counts per sentence
    token_counts = []
    for text in texts:
        encoded = tokenizer.encode(text)
        # Add 1 to account for blank shift (even though we don't shift here, the number of tokens is the same)
        token_counts.append(len(encoded.ids))


    print("\n=== BPE Tokenization Statistics ===")
    print(f"Number of sentences analyzed: {len(token_counts)}")
    print(f"Min tokens per sentence: {min(token_counts)}")
    print(f"Max tokens per sentence: {max(token_counts)}")
    print(f"Mean tokens: {np.mean(token_counts):.2f}")
    print(f"Median tokens: {np.median(token_counts):.2f}")
    print(f"95th percentile: {np.percentile(token_counts, 95):.0f}")
    print(f"99th percentile: {np.percentile(token_counts, 99):.0f}")

    # Compare with original character lengths
    char_lengths = df['text'].str.len().tolist()
    avg_compression = np.mean(char_lengths) / np.mean(token_counts)
    print(f"\nCompression ratio (chars → BPE tokens): {avg_compression:.2f}x")

    # Recommendation for target_frames in ViViT
    p95_tokens = np.percentile(token_counts, 95)
    print(f"Set ViViT target_frames = {int(p95_tokens)} (95th percentile of BPE tokens)")
    print(f"This ensures CTC can handle {int(p95_tokens)} output steps for 95% of sentences")
    