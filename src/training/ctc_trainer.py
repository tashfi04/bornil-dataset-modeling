import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.utils.metrics import calculate_all_metrics

from src.training.base_trainer import BaseTrainer

class CTCTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
    def setup_data(self):
        """Setup data loaders for CTC training"""
        from src.data_loader import get_data_loaders
        
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.config)
        
        with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.num_classes = len(self.vocab['char_to_id'])
        
        self.logger.info(f"Vocabulary size: {self.num_classes}")
        self.logger.info(f"Video parameters: {self.config.num_frames} frames, {self.config.frame_size} resolution")
        
    def setup_model(self):
        """Setup CTC model - to be implemented by specific model trainers"""
        # This will be overridden by model-specific trainers
        self.model = None
        
    def setup_optimizer(self):
        """Setup optimizer for CTC training"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(trainable_params, lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            videos = batch['videos'].to(self.config.device)
            text_targets = batch['text_targets'].to(self.config.device)  # 1D targets
            video_lengths = batch['video_lengths'].to(self.config.device)
            text_lengths = batch['text_lengths'].to(self.config.device)
            
            # Forward pass with video lengths
            self.optimizer.zero_grad()
            outputs = self.model(videos, video_lengths)
            
            # CTC loss calculation
            loss = self.criterion(
                outputs.permute(1, 0, 2),   # (T, B, C)
                text_targets,               # 1D concatenated targets
                video_lengths,              # Use actual video lengths
                text_lengths
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch:03d} [Val]')
            for batch in pbar:
                videos = batch['videos'].to(self.config.device)
                text_targets = batch['text_targets'].to(self.config.device)  # 1D targets
                video_lengths = batch['video_lengths'].to(self.config.device)
                text_lengths = batch['text_lengths'].to(self.config.device)
                
                outputs = self.model(videos, video_lengths)
                loss = self.criterion(
                    outputs.permute(1, 0, 2),
                    text_targets,   # 1D concatenated targets
                    video_lengths,
                    text_lengths
                )
                total_loss += loss.item()
                pbar.set_postfix({'ValLoss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        filename = f'checkpoint_epoch_{epoch:03d}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        cnn_unfrozen = False
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning Rate: {current_lr:.2e}")

            # Calculate metrics (less frequently to save time)
            metrics = self.calculate_wer_cer(epoch)
            if metrics is not None:
                self.logger.info(
                    f"Metrics - WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}, "
                    f"Exact Match: {metrics['exact_match_accuracy']:.4f}, "
                    f"Token Accuracy: {metrics['token_accuracy']:.4f}"
                )
            
            # SMART UNFREEZING: Unfreeze CNN after validation loss plateaus
            if (not cnn_unfrozen and 
                patience_counter >= 3 and  # After 3 epochs without improvement
                self.config.freeze_cnn_initially):
                
                self.logger.info("Unfreezing CNN backbone for fine-tuning")
                self.model.unfreeze_cnn()
                
                # Reset optimizer with lower learning rate for fine-tuning
                self.optimizer = Adam(
                    [{'params': self.model.cnn.parameters(), 'lr': self.config.learning_rate / 10},
                    {'params': self.model.lstm.parameters()},
                    {'params': self.model.classifier.parameters()}],
                    lr=self.config.learning_rate
                )
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
                cnn_unfrozen = True
                patience_counter = 0  # Reset patience after unfreezing
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, val_loss, is_best)
            self.logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                break

    def calculate_wer_cer(self, epoch):
        """Calculate comprehensive evaluation metrics"""
        if epoch % 5 != 0:  # Calculate every 5 epochs to save time
            return None
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                videos = batch['videos'].to(self.config.device)
                video_lengths = batch['video_lengths'].to(self.config.device)
                text_seqs = batch['text_seqs']
                text_labels = batch['text_labels']
                
                outputs = self.model(videos, video_lengths)
                outputs = outputs.permute(1, 0, 2)  # (T, B, C) for CTC
                
                # Greedy decoding
                _, max_indices = torch.max(outputs, dim=2)
                max_indices = max_indices.transpose(0, 1).cpu().numpy()  # (B, T)

                # Get video lengths for decoding
                video_lengths_np = batch['video_lengths'].cpu().numpy()
                
                for i in range(len(max_indices)):
                    # Only decode up to actual video length (ignore padding)
                    seq_len = video_lengths_np[i]
                    sequence = max_indices[i][:seq_len]  # Only take real frames

                    # Remove blanks and collapse repeats
                    decoded = []
                    previous = None
                    for idx in sequence:
                        if idx != 0 and idx != previous:  # 0 is blank token
                            decoded.append(idx)
                        previous = idx
                    
                    # Convert to text
                    predicted_text = ''.join([self.vocab['id_to_char'][str(idx)] for idx in decoded])
                    all_predictions.append(predicted_text)
                    all_targets.append(text_labels[i])
        
        # Use utils metrics function
        metrics = calculate_all_metrics(all_targets, all_predictions)
        
        return metrics
