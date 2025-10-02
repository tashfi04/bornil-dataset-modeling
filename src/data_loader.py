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
        original_idx = idx
        max_attempts = 3
        
        for attempt in range(max_attempts):
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
                logger.warning(f"Error loading sample {idx} (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:  # Last attempt failed
                    logger.error(f"Failed to load sample {original_idx} after {max_attempts} attempts")
                    # Return a dummy sample that won't break training
                    dummy_video = torch.zeros((3, self.config.num_frames, 
                                            self.config.frame_size[0], self.config.frame_size[1]))
                    dummy_text = ""
                    dummy_seq = torch.LongTensor([0])  # Blank token
                    return {
                        'video': dummy_video,
                        'text': dummy_text,
                        'text_seq': dummy_seq,
                        'video_path': 'failed_to_load'
                    }
                # Try next sample
                idx = (idx + 1) % len(self.df)
    
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

        # Only do minimal sampling if video is too long (to prevent memory issues)
        max_frames = getattr(self.config, 'max_frames', 300)  # Set a reasonable maximum
        if len(frames) > max_frames:
            # Sample evenly but keep more temporal information
            indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            logger.info(f"Video too long ({len(frames)} frames), sampled to {max_frames}")
        
        # Padding for shorter videos is handled in collate_fn
        # Normalize and reshape
        frames = np.array(frames) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        
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

    # Load fixed splits
    split_file = config.train_val_test_split_path
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            f"Please run scripts/generate_splits.py first."
        )

    with open(split_file, 'r', encoding='utf-8') as f:
        split_mapping = json.load(f)

    # Split the data based on saved mapping
    train_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'train'])]
    val_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'val'])]
    test_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'test'])]
    
    logger.info(f"Using fixed splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
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
    """Custom collate function to handle variable length videos and text sequences"""
    # Filter out None values from failed samples
    batch = [b for b in batch if b is not None]

    # Sort batch by video length (descending) for packed sequences
    batch.sort(key=lambda x: x['video'].size(1), reverse=True)
    
    videos = [item['video'] for item in batch]
    text_seqs = [item['text_seq'] for item in batch]
    text_labels = [item['text'] for item in batch]
    video_paths = [item['video_path'] for item in batch]

    # Get actual video lengths (number of frames)
    video_lengths = torch.LongTensor([video.size(1) for video in videos])

    # Pad videos to maximum length in this batch
    padded_videos = torch.nn.utils.rnn.pad_sequence(
        [video.permute(1, 0, 2, 3) for video in videos],  # (T, C, H, W) for padding
        batch_first=True
    ).permute(0, 2, 1, 3, 4)  # Back to (B, C, T, H, W)
    
    # Get lengths for CTC loss
    text_lengths = torch.LongTensor([len(seq) for seq in text_seqs])
    
    # Pad text sequences
    padded_text_seqs = torch.nn.utils.rnn.pad_sequence(text_seqs, batch_first=True)
    
    return {
        'videos': padded_videos,
        'text_seqs': padded_text_seqs,
        'text_labels': text_labels,
        'video_paths': video_paths,
        'video_lengths': video_lengths, # Actual lengths before padding
        'text_lengths': text_lengths
    }
