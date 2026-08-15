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
        # True for schedulers that advance every optimizer step (e.g. OneCycleLR)
        # rather than once per epoch. setup_optimizer() may override this.
        self.step_scheduler_per_batch = False

        # Set before setup_optimizer(), which subclasses use to size the schedule
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.accumulation_count = 0

        # Mixed precision for the forward pass; the CTC loss stays in fp32
        self.use_amp = bool(getattr(config, 'use_amp', False)) and torch.cuda.is_available()
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        except (AttributeError, TypeError):  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.setup_data()
        self.setup_model()
        self.setup_optimizer()

    def setup_data(self):
        """Setup data loaders for CTC training"""
        from src.data_loader import get_data_loaders

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.config)

        # num_classes must match the tokenization the dataset uses. Both schemes
        # reserve id 0 for the CTC blank.
        self.tokenization_type = getattr(self.config, 'tokenization_type', 'character')
        if self.tokenization_type == 'bpe':
            from src.utils.text_utils import load_bpe_tokenizer

            self.vocab = None
            self.tokenizer = load_bpe_tokenizer(self.config.bpe_tokenizer_path)
            # text_to_bpe_ids shifts ids by +1, so ids span 1..vocab_size
            self.num_classes = self.tokenizer.get_vocab_size() + 1

            # The tokenizer is trained by a separate script, so it can fall out of
            # step with the config it is being used under
            configured = getattr(self.config, 'bpe_vocab_size', None)
            if configured is not None and self.tokenizer.get_vocab_size() != configured:
                self.logger.warning(
                    f"Tokenizer at {self.config.bpe_tokenizer_path} has vocab size "
                    f"{self.tokenizer.get_vocab_size()} but config.bpe_vocab_size is "
                    f"{configured}. Re-run scripts/train_bpe_tokenizer.py "
                    f"--bpe_vocab_size {configured} if this is unintended."
                )
        else:
            with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            self.tokenizer = None
            self.num_classes = len(self.vocab['char_to_id'])

        self.logger.info(f"Tokenization: {self.tokenization_type}")
        self.logger.info(f"Vocabulary size (num_classes incl. blank): {self.num_classes}")
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
        total_samples_this_epoch = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d} [Train]')

        # Reset gradients at start of epoch
        self.optimizer.zero_grad()
        self.accumulation_count = 0

        for batch_idx, batch in enumerate(pbar):
            total_samples_this_epoch += len(batch['video_paths'])

            # Move the whole batch to the GPU in one go
            batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            videos = batch['videos']
            text_targets = batch['text_targets']  # 1D targets
            video_lengths = batch['video_lengths']
            text_lengths = batch['text_lengths']

            assert text_targets.numel() == text_lengths.sum().item(), f"Mismatch: text_targets has {text_targets.numel()} elements but text_lengths sum to {text_lengths.sum().item()}"

            with self._autocast():
                outputs = self.model(videos, video_lengths)

            input_lengths = self.get_input_lengths(outputs, video_lengths)

            # zero_infinity=True silently zeroes samples where the target is
            # longer than the input, so report them instead of losing them
            invalid_samples = (input_lengths < text_lengths).sum().item()
            if invalid_samples > 0:
                self.logger.warning(f"Invalid Samples: Found {invalid_samples} samples with input_length < text_length (these contribute no gradient)")
                self.logger.warning(f"Input lengths: {input_lengths.tolist()}")
                self.logger.warning(f"Text lengths: {text_lengths.tolist()}")

            # Computed in fp32 for numerical stability
            loss = self.criterion(
                outputs.float(),            # (T, B, C)
                text_targets,               # 1D concatenated targets
                input_lengths,
                text_lengths
            )

            # Normalize loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            accumulated_loss += loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1)
            self.accumulation_count += 1

            # Only step optimizer and clip gradients when we've accumulated enough or at the end of epoch
            if (self.accumulation_count % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(self.train_loader)):

                # Unscale first so grad_clip applies to unscaled gradients
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Advance per-step schedulers alongside the optimizer
                if self.step_scheduler_per_batch and self.scheduler is not None:
                    self.scheduler.step()

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

        self.logger.info(f"Epoch {epoch}: trained on {total_samples_this_epoch} samples")

        # Calculate average loss
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return avg_loss

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_samples_this_epoch = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch:03d} [Val]')
            for batch in pbar:
                total_samples_this_epoch += len(batch['video_paths'])

                # Move the whole batch to the GPU in one go
                batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                videos = batch['videos']
                text_targets = batch['text_targets']  # 1D targets
                video_lengths = batch['video_lengths']
                text_lengths = batch['text_lengths']

                with self._autocast():
                    outputs = self.model(videos, video_lengths)
                loss = self.criterion(
                    outputs.float(),
                    text_targets,   # 1D concatenated targets
                    self.get_input_lengths(outputs, video_lengths),
                    text_lengths
                )
                total_loss += loss.item()
                pbar.set_postfix({'ValLoss': f'{loss.item():.4f}'})

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""

        model_state_dict = getattr(self.model, 'module', self.model).state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
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
        backbone_unfrozen = False

        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.config.num_epochs}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Per-step schedulers already advanced inside train_epoch
            if not self.step_scheduler_per_batch and self.scheduler is not None:
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
                interval = max(1, getattr(self.config, 'metrics_interval', 5))
                self.logger.info(f"Metrics calculation skipped this epoch (runs every {interval} epochs)")

            # Unfreeze the pretrained backbone once validation loss plateaus
            if (not backbone_unfrozen and
                patience_counter >= 3 and  # After 3 epochs without improvement
                self._backbone_frozen_initially()):

                if self._unfreeze_backbone():
                    backbone_unfrozen = True
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

    def _autocast(self):
        """Mixed-precision context for the forward pass; a no-op when disabled."""
        return torch.amp.autocast('cuda', enabled=self.use_amp)

    def get_input_lengths(self, outputs, video_lengths):
        """CTC `input_lengths` for this model.

        Assumes the model preserves its time axis frame-for-frame, as CNN-BiLSTM
        does, so the real per-video frame counts apply. Models that resample time
        must override this.
        """
        return video_lengths

    def _backbone_frozen_initially(self):
        """Whether the run started with a frozen backbone.

        CNN-BiLSTM uses `freeze_cnn_initially`, ViViT uses `freeze_backbone`.
        """
        return bool(getattr(self.config, 'freeze_cnn_initially', False)
                    or getattr(self.config, 'freeze_backbone', False))

    def _unfreeze_backbone(self):
        """Unfreeze the backbone and rebuild the optimizer.

        The backbone fine-tunes at a tenth of the head's learning rate. Handles
        both CNN-BiLSTM (.cnn) and ViViT (.vivit). Returns True if it happened.
        """
        base_model = getattr(self.model, 'module', self.model)

        if hasattr(base_model, 'unfreeze_cnn') and hasattr(base_model, 'cnn'):
            base_model.unfreeze_cnn()
            backbone = base_model.cnn
            name = 'CNN'
        elif hasattr(base_model, 'unfreeze_backbone') and hasattr(base_model, 'vivit'):
            base_model.unfreeze_backbone()
            backbone = base_model.vivit
            name = 'ViViT'
        else:
            self.logger.warning("Model exposes no known unfreeze helper; skipping unfreeze")
            return False

        self.logger.info(f"Unfreezing {name} backbone for fine-tuning")

        # Separate backbone from head without naming specific submodules
        backbone_ids = {id(p) for p in backbone.parameters()}
        head_params = [p for p in base_model.parameters() if id(p) not in backbone_ids]

        self.optimizer = Adam(
            [{'params': list(backbone.parameters()), 'lr': self.config.learning_rate / 10},
             {'params': head_params}],
            lr=self.config.learning_rate
        )
        # A per-step schedule does not carry over to the rebuilt optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.step_scheduler_per_batch = False
        return True

    def _ids_to_text(self, ids):
        """Decode CTC-collapsed ids back to text for whichever tokenization is active"""
        if self.tokenization_type == 'bpe':
            from src.utils.text_utils import bpe_ids_to_text

            return bpe_ids_to_text([int(i) for i in ids], self.tokenizer)
        return ''.join(self.vocab['id_to_char'][str(int(idx))] for idx in ids)

    def calculate_wer_cer(self, epoch):
        """Calculate comprehensive evaluation metrics"""
        interval = max(1, getattr(self.config, 'metrics_interval', 5))
        if epoch % interval != 0:
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

                with self._autocast():
                    outputs = self.model(videos, video_lengths)

                # Greedy decoding
                _, max_indices = torch.max(outputs, dim=2)
                max_indices = max_indices.transpose(0, 1).cpu().numpy()  # (B, T)

                # Decode only over the model's real CTC time steps
                input_lengths_np = self.get_input_lengths(outputs, video_lengths).cpu().numpy()

                for i in range(len(max_indices)):
                    seq_len = input_lengths_np[i]
                    sequence = max_indices[i][:seq_len]  # Only take real frames

                    # Remove blanks and collapse repeats
                    decoded = []
                    previous = None
                    for idx in sequence:
                        if idx != 0 and idx != previous:  # 0 is blank token
                            decoded.append(idx)
                        previous = idx

                    # Convert to text using whichever tokenization is active
                    predicted_text = self._ids_to_text(decoded)
                    all_predictions.append(predicted_text)
                    all_targets.append(text_labels[i])

        # Use utils metrics function
        metrics = calculate_all_metrics(all_targets, all_predictions)

        return metrics
