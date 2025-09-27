import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from configs.base_config import config
from src.utils.text_utils import text_to_int

class BdSLDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        
        # Load vocabulary
        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char_to_id = vocab['char_to_id']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_label = row['text']
        video_filename = row['recording']
        chunk_folder = row['chunk_path']
        
        # Build full video path
        full_video_path = os.path.join(config.chunk_base_path, 
                                      chunk_folder, video_filename)
        
        # Load and preprocess video
        video = self.load_video_frames(full_video_path)
        
        # Convert text to integer sequence
        text_seq = text_to_int(text_label, self.char_to_id)
        
        return {
            'video': torch.FloatTensor(video),
            'text': text_label,
            'text_seq': torch.IntTensor(text_seq),
            'video_path': full_video_path
        }
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize and convert to RGB
                frame = cv2.resize(frame, (config.frame_size[1], config.frame_size[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        
        # Sample fixed number of frames
        if len(frames) > config.num_frames:
            indices = np.linspace(0, len(frames)-1, config.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < config.num_frames:
            # Pad with last frame
            frames.extend([frames[-1]] * (config.num_frames - len(frames)))
        
        # Normalize and reshape to (C, T, H, W)
        frames = np.array(frames) / 255.0  # Normalize to [0, 1]
        frames = np.transpose(frames, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
        
        return frames

def get_data_loaders():
    """Create train, validation, and test data loaders"""
    df = pd.read_csv(config.csv_path)
    
    # Train/val/test split
    train_df, test_df = train_test_split(df, test_size=config.test_ratio, 
                                        random_state=config.random_seed)
    train_df, val_df = train_test_split(train_df, test_size=config.val_ratio/(1-config.test_ratio), 
                                       random_state=config.random_seed)
    
    # Create datasets
    train_dataset = BdSLDataset(train_df, 'train')
    val_dataset = BdSLDataset(val_df, 'val')
    test_dataset = BdSLDataset(test_df, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers,
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers,
                           collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers,
                            collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """Custom collate function to handle variable length text sequences"""
    videos = torch.stack([item['video'] for item in batch])
    text_seqs = [item['text_seq'] for item in batch]
    text_labels = [item['text'] for item in batch]
    video_paths = [item['video_path'] for item in batch]
    
    # Get lengths for CTC loss
    video_lengths = torch.IntTensor([config.num_frames] * len(batch))
    text_lengths = torch.IntTensor([len(seq) for seq in text_seqs])
    
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
