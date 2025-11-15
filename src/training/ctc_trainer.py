import os
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

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.accumulation_count = 0
        
    def setup_data(self):
        """Setup data loaders for CTC training"""
        from src.data_loader import get_data_loaders
        
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.config)
        
        with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.num_classes = len(self.vocab['char_to_id'])
        
        self.logger.info(f"Vocabulary size: {self.num_classes}")
        self.logger.info(f"Video parameters: variable length (num_frames={self.config.num_frames}), {self.config.frame_size} resolution")
        
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
        """Train for one epoch with optional gradient accumulation"""
        self.model.train()
        total_loss = 0
        accumulated_loss = 0

        # Track dummy samples for this epoch
        dummy_samples_this_epoch = 0
        total_samples_this_epoch = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d} [Train]')

        # Reset gradients at start of epoch
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Count dummy samples in this batch
            batch_dummy_samples = sum(1 for path in batch['video_paths'] if path == 'failed_to_load')
            dummy_samples_this_epoch += batch_dummy_samples
            total_samples_this_epoch += len(batch['video_paths'])

            if batch_dummy_samples > 0:
                self.logger.warning(f"Batch {batch_idx}: {batch_dummy_samples}/{len(batch['video_paths'])} are dummy samples!")

            # Move entire batch to GPU once
            batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            videos = batch['videos']
            text_targets = batch['text_targets']  # 1D targets
            video_lengths = batch['video_lengths']
            text_lengths = batch['text_lengths']

            # DEBUG: Length Checking
            invalid_samples = (video_lengths < text_lengths).sum().item()
            if invalid_samples > 0:
                self.logger.warning(f"Invalid Samples: Found {invalid_samples} samples with video_length < text_length")
                self.logger.warning(f"Video lengths: {video_lengths.tolist()}")
                self.logger.warning(f"Text lengths: {text_lengths.tolist()}")

            # Forward pass with video lengths
            outputs = self.model(videos, video_lengths)
            
            # CTC loss calculation
            loss = self.criterion(
                outputs,                    # (T, B, C)
                text_targets,               # 1D concatenated targets
                video_lengths,              # Use actual video lengths
                text_lengths
            )

            # Normalize loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()

            accumulated_loss += loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1)
            self.accumulation_count += 1

            # Only step optimizer and clip gradients when we've accumulated enough or at the end of epoch
            if (self.accumulation_count % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                
                # Clip gradients
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{accumulated_loss:.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Accum': f'{self.accumulation_count}/{self.gradient_accumulation_steps}'
                })
                
                total_loss += accumulated_loss
                accumulated_loss = 0
                self.accumulation_count = 0

        # Log dummy sample summary for the epoch
        if dummy_samples_this_epoch > 0:
            dummy_percentage = (dummy_samples_this_epoch / total_samples_this_epoch) * 100
            self.logger.warning(f"Epoch {epoch}: {dummy_samples_this_epoch}/{total_samples_this_epoch} ({dummy_percentage:.1f}%) were dummy samples!")
            if dummy_percentage > 50:
                self.logger.error("More than 50% dummy samples! Training will not be effective!")
        else:
            self.logger.info(f"Epoch {epoch}: All samples loaded successfully")

        # Calculate average loss
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        # Track dummy samples for validation
        dummy_samples_this_epoch = 0
        total_samples_this_epoch = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch:03d} [Val]')
            for batch in pbar:
                # Count dummy samples in this batch
                batch_dummy_samples = sum(1 for path in batch['video_paths'] if path == 'failed_to_load')
                dummy_samples_this_epoch += batch_dummy_samples
                total_samples_this_epoch += len(batch['video_paths'])

                # OPTIMIZATION: Move entire batch to GPU once
                batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                videos = batch['videos']
                text_targets = batch['text_targets']  # 1D targets
                video_lengths = batch['video_lengths']
                text_lengths = batch['text_lengths']
                
                outputs = self.model(videos, video_lengths)
                loss = self.criterion(
                    outputs,
                    text_targets,   # 1D concatenated targets
                    video_lengths,
                    text_lengths
                )
                total_loss += loss.item()
                pbar.set_postfix({'ValLoss': f'{loss.item():.4f}'})

        # Log dummy sample summary for validation
        if dummy_samples_this_epoch > 0:
            dummy_percentage = (dummy_samples_this_epoch / total_samples_this_epoch) * 100
            self.logger.warning(f"Validation Epoch {epoch}: {dummy_samples_this_epoch}/{total_samples_this_epoch} ({dummy_percentage:.1f}%) were dummy samples!")

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
            else:
                self.logger.info("Metrics calculation skipped this epoch (runs every 5 epochs)")
            
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
                # Move input to LSTM device immediately
                batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                videos = batch['videos']
                video_lengths = batch['video_lengths']
                text_labels = batch['text_labels']
                
                outputs = self.model(videos, video_lengths)

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
