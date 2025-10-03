import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.base_config import config

def generate_and_save_splits():
    """Generate and save train/val/test splits for reproducible experiments"""
    
    # Load the dataset
    df = pd.read_csv(config.csv_path)
    print(f"Total samples: {len(df)}")
    
    # Create a unique identifier for each sample (using recording filename)
    # This ensures we can reliably map back to the same splits
    sample_ids = df['recording'].tolist()
    
    # Stratified split by text length (better than random)
    # This ensures similar sentence length distributions across splits
    df['text_length'] = df['text'].str.len()

    # Since we have unique sentences, we can't stratify by exact text
    # Instead, we use text length bins, but handle the case where bins might have few samples
    
    # Create more robust bins - use fewer bins for better distribution
    n_bins = min(5, len(df) // 20)  # Ensure at least ~20 samples per bin
    df['length_bin'] = pd.cut(df['text_length'], bins=n_bins)
    
    # Check bin counts
    bin_counts = df['length_bin'].value_counts()
    print("Text length bin distribution:")
    for bin_val, count in bin_counts.items():
        print(f"  {bin_val}: {count} samples")

    # First split: separate test set
    # Use stratification if possible, otherwise fall back to random
    try:
        train_val_df, test_df = train_test_split(
            df, 
            test_size=config.test_ratio,
            random_state=config.random_seed,
            stratify=df['length_bin']
        )
        print("Used stratified split for test set")
    except ValueError as e:
        print(f"Stratification failed: {e}. Using random split.")
        train_val_df, test_df = train_test_split(
            df, 
            test_size=config.test_ratio,
            random_state=config.random_seed
        )

    # Second split: separate validation set
    train_val_df['length_bin'] = pd.cut(train_val_df['text_length'], bins=n_bins)
    
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=config.val_ratio/(1-config.test_ratio),
            random_state=config.random_seed,
            stratify=train_val_df['length_bin']
        )
        print("Used stratified split for validation set")
    except ValueError as e:
        print(f"Stratification failed: {e}. Using random split.")
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=config.val_ratio/(1-config.test_ratio),
            random_state=config.random_seed
        )
    
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
    print(f"\n✓ All {len(original_ids)} samples accounted for in splits")

if __name__ == "__main__":
    generate_and_save_splits()
