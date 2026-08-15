import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.base_config import config

def generate_and_save_splits():
    """Generate and save train/val/test splits for reuse"""

    # Load the dataset
    df = pd.read_csv(config.csv_path)
    print(f"Total samples: {len(df)}")

    # Calculate text lengths
    df['text_length'] = df['text'].str.len()

    # Since we have unique sentences, we can't stratify by exact text
    # Instead, we use text length bins

    # We'll use fewer bins to ensure minimum samples
    n_bins = min(10, len(df) // 4)  # Ensure at least 4 samples per bin

    # Use quantile-based binning for balanced bins
    df['length_bin'] = pd.qcut(df['text_length'], q=n_bins, duplicates='drop')

    # Check bin distribution
    print(f"Using {len(df['length_bin'].unique())} bins")
    print("Bin distribution:")
    print(df['length_bin'].value_counts().sort_index())

    # Initialize splitter
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.test_ratio, random_state=config.random_seed)

    # First split: train_val vs test
    train_val_idx, test_idx = next(sss.split(df, df['length_bin']))
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Second split: train vs val
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio/(1-config.test_ratio),
                                   random_state=config.random_seed)
    train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df['length_bin']))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # Create split mapping
    split_mapping = {}
    for _, row in train_df.iterrows():
        split_mapping[row['recording']] = 'train'
    for _, row in val_df.iterrows():
        split_mapping[row['recording']] = 'val'
    for _, row in test_df.iterrows():
        split_mapping[row['recording']] = 'test'

    # Save the split
    split_file = os.path.join(config.repo_root, 'data', 'train_val_test_split.json')
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_mapping, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"\nFinal split sizes:")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Split saved to: {split_file}")

    # Verify text length distributions
    print("\nText length statistics by split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        avg_len = split_df['text_length'].mean()
        std_len = split_df['text_length'].std()
        print(f"{split_name}: {avg_len:.2f} ± {std_len:.2f} chars (min: {split_df['text_length'].min()}, max: {split_df['text_length'].max()})")

    # Verify we have the same unique recording IDs
    original_ids = set(df['recording'].tolist())
    split_ids = set(split_mapping.keys())
    assert original_ids == split_ids, "Some samples were lost in splitting!"
    print(f"\nAll {len(original_ids)} samples accounted for in splits")

if __name__ == "__main__":
    generate_and_save_splits()
