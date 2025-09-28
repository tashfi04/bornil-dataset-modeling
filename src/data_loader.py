import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

from src.utils.text_utils import text_to_int

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BdSLDataset(Dataset):
    def __init__(self, df, config, mode='train'):
        """
        Args:
            df: DataFrame with video metadata
            config: Configuration object (from base_config or model-specific config)
            mode: 'train', 'val', or 'test'
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.config = config  # Store the config object
        
        # Load vocabulary
        with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char_to_id = vocab['char_to_id']
        
        logger.info(f"Initialized {mode} dataset with {len(self.df)} samples")
        logger.info(f"Using config: {self.config.model_type if hasattr(self.config, 'model_type') else 'base'}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            text_label = row['text']
            video_filename = row['recording']
            chunk_folder = row['chunk_path']
            
            # Build full video path using config
            full_video_path = os.path.join(self.config.chunk_base_path, 
                                          chunk_folder, video_filename)
            
            # Load and preprocess video using config parameters
            video = self.load_video_frames(full_video_path)
            
            # Convert text to integer sequence
            text_seq = text_to_int(text_label, self.char_to_id)
            
            return {
                'video': torch.FloatTensor(video),
                'text': text_label,
                'text_seq': torch.LongTensor(text_seq),
                'video_path': full_video_path
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the batch
            return self.__getitem__((idx + 1) % len(self.df))
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames using config parameters"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Use config parameters for resizing
                frame = cv2.resize(frame, (self.config.frame_size[1], self.config.frame_size[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Sample fixed number of frames using config parameter
        if len(frames) > self.config.num_frames:
            indices = np.linspace(0, len(frames)-1, self.config.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.config.num_frames:
            # Pad with last frame
            frames.extend([frames[-1]] * (self.config.num_frames - len(frames)))
        
        # Normalize and reshape to (C, T, H, W)
        frames = np.array(frames) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
        
        return frames

def get_data_loaders(config):
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration object containing all necessary parameters
    """
    try:
        df = pd.read_csv(config.csv_path)
        logger.info(f"Loaded dataset with {len(df)} total samples")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise
    
    # Use config parameters for splitting
    train_df, test_df = train_test_split(
        df, test_size=config.test_ratio, 
        random_state=config.random_seed,
        stratify=df['text'] if len(df['text'].unique()) > 1 else None
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.val_ratio/(1-config.test_ratio), 
        random_state=config.random_seed
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets with config
    train_dataset = BdSLDataset(train_df, config, 'train')
    val_dataset = BdSLDataset(val_df, config, 'val')
    test_dataset = BdSLDataset(test_df, config, 'test')
    
    # Use config parameters for DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """Custom collate function to handle variable length text sequences"""
    # Filter out None values from failed samples
    batch = [b for b in batch if b is not None]
    
    videos = torch.stack([item['video'] for item in batch])
    text_seqs = [item['text_seq'] for item in batch]
    text_labels = [item['text'] for item in batch]
    video_paths = [item['video_path'] for item in batch]
    
    # Get lengths for CTC loss
    video_lengths = torch.LongTensor([videos.size(1)] * len(batch))  # Use actual sequence length
    text_lengths = torch.LongTensor([len(seq) for seq in text_seqs])
    
    # Pad text sequences
    padded_text_seqs = torch.nn.utils.rnn.pad_sequence(text_seqs, batch_first=True)
    
    return {
        'videos': videos,
        'text_seqs': padded_text_seqs,
        'text_labels': text_labels,
        'video_paths': video_paths,
        'video_lengths': video_lengths,
        'text_lengths': text_lengths
    }
