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
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=config.test_ratio,
        random_state=config.random_seed,
        stratify=pd.cut(df['text_length'], bins=5)  # Stratify by text length
    )
    
    # Second split: separate validation set
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.val_ratio/(1-config.test_ratio),
        random_state=config.random_seed,
        stratify=pd.cut(train_val_df['text_length'], bins=5)
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
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples") 
    print(f"Test set: {len(test_df)} samples")
    print(f"Split saved to: {split_file}")
    
    # Verify stratification
    print("\nText length statistics:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        avg_len = split_df['text_length'].mean()
        print(f"{split_name}: avg text length = {avg_len:.2f}")

if __name__ == "__main__":
    generate_and_save_splits()
