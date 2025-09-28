import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from configs.base_config import config
from src.data_loader import get_data_loaders
from src.models.cnn_lstm_ctc import CNNBiLSTMCTC
from src.utils.text_utils import int_to_text

def train_model():
    # Load data
    train_loader, val_loader, _ = get_data_loaders()
    
    # Load vocabulary for decoder
    with open(config.vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_char = vocab['id_to_char']
    num_classes = len(vocab['char_to_id'])
    
    # Model
    model = CNNBiLSTMCTC(num_classes).to(config.device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0)  # Blank token is at index 0
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]"):
            videos = batch['videos'].to(config.device)
            text_seqs = batch['text_seqs'].to(config.device)
            video_lengths = batch['video_lengths'].to(config.device)
            text_lengths = batch['text_lengths'].to(config.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(videos)
            
            # Calculate loss
            loss = criterion(
                outputs.permute(1, 0, 2),  # (T, B, C)
                text_seqs,
                video_lengths,
                text_lengths
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                videos = batch['videos'].to(config.device)
                text_seqs = batch['text_seqs'].to(config.device)
                video_lengths = batch['video_lengths'].to(config.device)
                text_lengths = batch['text_lengths'].to(config.device)
                
                outputs = model(videos)
                loss = criterion(
                    outputs.permute(1, 0, 2),
                    text_seqs,
                    video_lengths,
                    text_lengths
                )
                val_loss += loss.item()
        
        # Print progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    train_model()
