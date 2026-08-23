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

        # Optional caps for smoke runs, so the whole loop can be exercised without
        # decoding the entire dataset
        self.train_batches = self._capped(len(self.train_loader), 'max_train_batches')
        self.val_batches = self._capped(len(self.val_loader), 'max_val_batches')
        if self.train_batches < len(self.train_loader) or self.val_batches < len(self.val_loader):
            self.logger.warning(
                f"Batch caps active: {self.train_batches}/{len(self.train_loader)} train, "
                f"{self.val_batches}/{len(self.val_loader)} val batches per epoch. "
                f"Set max_train_batches/max_val_batches to None for a full pass."
            )

        self.logger.info(f"Tokenization: {self.tokenization_type}")
        self.logger.info(f"Vocabulary size (num_classes incl. blank): {self.num_classes}")
        self.logger.info(f"Video parameters: variable length (num_frames={self.config.num_frames}), {self.config.frame_size} resolution")

    def _capped(self, total, config_key):
        """Batch count for one epoch, honouring an optional cap."""
        cap = getattr(self.config, config_key, None)
        return min(total, cap) if cap else total

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
            if batch_idx >= self.train_batches:
                break
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
                outputs = self.model(videos, video_lengths)   # (B, T, C)

            input_lengths = self.get_input_lengths(outputs, video_lengths)
            self._check_ctc_lengths(input_lengths, text_lengths)

            # CTC wants (T, B, C), and fp32 for numerical stability
            loss = self.criterion(
                outputs.permute(1, 0, 2).float(),
                text_targets,               # 1D concatenated targets
                input_lengths,
                text_lengths
            )

            self._check_ctc_loss(loss, batch_idx)

            # Normalize loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            accumulated_loss += loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1)
            self.accumulation_count += 1

            # Only step optimizer and clip gradients when we've accumulated enough or at the end of epoch
            if (self.accumulation_count % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == self.train_batches):

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

                # accumulated_loss is the sum over this window, so divide by the
                # batches in it to stay comparable with the validation loss
                window_batches = max(1, self.accumulation_count)
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{accumulated_loss / window_batches:.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Accum': f'{self.accumulation_count}/{self.gradient_accumulation_steps}'
                })

                total_loss += accumulated_loss
                accumulated_loss = 0
                self.accumulation_count = 0

        self.logger.info(f"Epoch {epoch}: trained on {total_samples_this_epoch} samples")

        # Calculate average loss
        avg_loss = total_loss / self.train_batches if self.train_batches > 0 else 0

        return avg_loss

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_samples_this_epoch = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch:03d} [Val]')
            batches_run = 0
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= self.val_batches:
                    break
                batches_run += 1
                total_samples_this_epoch += len(batch['video_paths'])

                # Move the whole batch to the GPU in one go
                batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                videos = batch['videos']
                text_targets = batch['text_targets']  # 1D targets
                video_lengths = batch['video_lengths']
                text_lengths = batch['text_lengths']

                with self._autocast():
                    outputs = self.model(videos, video_lengths)   # (B, T, C)

                input_lengths = self.get_input_lengths(outputs, video_lengths)
                self._check_ctc_lengths(input_lengths, text_lengths)

                loss = self.criterion(
                    outputs.permute(1, 0, 2).float(),
                    text_targets,   # 1D concatenated targets
                    input_lengths,
                    text_lengths
                )
                total_loss += loss.item()
                pbar.set_postfix({'ValLoss': f'{loss.item():.4f}'})

        return total_loss / batches_run if batches_run else float('inf')

    def save_checkpoint(self, epoch, val_loss, is_best=False, training_state=None):
        """Write a checkpoint that training can be resumed from.

        `last_checkpoint.pth` is overwritten each epoch so an interrupted run can
        continue. Per-epoch copies are opt-in via `keep_epoch_checkpoints`,
        because each one is several hundred MB and they add up fast.
        """
        base_model = getattr(self.model, 'module', self.model)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_type': getattr(self.config, 'model_type', None),
            'num_classes': self.num_classes,
            'training_state': training_state or {},
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'last_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        if getattr(self.config, 'keep_epoch_checkpoints', False):
            torch.save(checkpoint, os.path.join(
                self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth'))

    def _resume_checkpoint_path(self):
        """An explicit `resume_from` wins; otherwise pick up last_checkpoint.pth."""
        explicit = getattr(self.config, 'resume_from', None)
        if explicit:
            if not os.path.exists(explicit):
                raise FileNotFoundError(
                    f"resume_from is set to {explicit}, which does not exist"
                )
            return explicit
        if not getattr(self.config, 'auto_resume', True):
            return None
        candidate = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        return candidate if os.path.exists(candidate) else None

    def load_checkpoint(self, path):
        """Restore model, optimizer, schedule and loop state from `path`."""
        self.logger.info(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.config.device)

        saved_type = checkpoint.get('model_type')
        current_type = getattr(self.config, 'model_type', None)
        if saved_type and current_type and saved_type != current_type:
            raise RuntimeError(
                f"Checkpoint was written by model_type={saved_type} but this run is "
                f"{current_type}. Set auto_resume=False or point resume_from elsewhere."
            )
        saved_classes = checkpoint.get('num_classes')
        if saved_classes is not None and saved_classes != self.num_classes:
            raise RuntimeError(
                f"Checkpoint has {saved_classes} output classes but this run has "
                f"{self.num_classes}; the tokenizer or vocabulary changed."
            )

        state = dict(checkpoint.get('training_state') or {})

        # Unfreezing rebuilds the optimizer with different param groups, so it has
        # to be replayed before the saved optimizer state will load.
        if state.get('backbone_unfrozen'):
            self.logger.info("Checkpoint was taken after unfreezing; replaying that")
            self._unfreeze_backbone()

        base_model = getattr(self.model, 'module', self.model)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as exc:
                self.logger.warning(
                    f"Could not restore the LR schedule ({exc}); it starts over. "
                    f"This happens when num_epochs or the dataset size changed."
                )
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        state['start_epoch'] = checkpoint['epoch'] + 1
        return state

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        backbone_unfrozen = False
        start_epoch = 1

        resume_path = self._resume_checkpoint_path()
        if resume_path:
            state = self.load_checkpoint(resume_path)
            start_epoch = state.get('start_epoch', 1)
            best_val_loss = state.get('best_val_loss', float('inf'))
            patience_counter = state.get('patience_counter', 0)
            backbone_unfrozen = state.get('backbone_unfrozen', False)
            self.logger.info(
                f"Resumed at epoch {start_epoch}/{self.config.num_epochs} "
                f"(best val loss {best_val_loss:.4f}, patience {patience_counter}, "
                f"backbone_unfrozen={backbone_unfrozen})"
            )
            if start_epoch > self.config.num_epochs:
                self.logger.info(
                    "This checkpoint already completed num_epochs; nothing to do. "
                    "Raise num_epochs to keep training."
                )
                return
        else:
            self.logger.info("Starting from scratch (no checkpoint to resume)")

        for epoch in range(start_epoch, self.config.num_epochs + 1):
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

            self.save_checkpoint(epoch, val_loss, is_best, training_state={
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'backbone_unfrozen': backbone_unfrozen,
            })
            self.logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                break

    def _autocast(self):
        """Mixed-precision context for the forward pass; a no-op when disabled."""
        return torch.amp.autocast('cuda', enabled=self.use_amp)

    def get_input_lengths(self, outputs, video_lengths):
        """CTC `input_lengths` for this model, given (B, T, C) outputs.

        Assumes the model preserves its time axis frame-for-frame, as CNN-BiLSTM
        does, so the real per-video frame counts apply. Models that resample time
        must override this.
        """
        return video_lengths

    def _check_ctc_loss(self, loss, batch_idx):
        """Catch batches CTC could not align at all.

        zero_infinity=True replaces an infinite loss with zero, which trains as a
        no-op. The usual cause is a target needing more steps than the time axis
        offers, e.g. adjacent duplicate tokens that each need a separating blank.
        """
        value = loss.detach()
        if torch.isfinite(value) and value.item() != 0.0:
            return

        message = (f"Batch {batch_idx} produced a {'non-finite' if not torch.isfinite(value) else 'zero'} "
                   f"CTC loss, so it contributes no gradient. Some target is not "
                   f"alignable within the model's time axis. Re-run "
                   f"scripts/validate_dataset.py, which rejects these.")
        if getattr(self.config, 'strict_data', True):
            raise RuntimeError(message + " Set strict_data=False to train through it.")
        self.logger.warning(message)

    def _check_ctc_lengths(self, input_lengths, text_lengths):
        """Reject batches CTC cannot align.

        zero_infinity=True would turn these into a zero loss and train on them
        silently. Pre-validation is supposed to have removed them, so treat any
        survivor as a bug rather than something to skip.
        """
        invalid = (input_lengths < text_lengths)
        if not invalid.any():
            return

        count = invalid.sum().item()
        detail = (f"{count} sample(s) have more target tokens than CTC time steps. "
                  f"Input lengths: {input_lengths.tolist()}, "
                  f"text lengths: {text_lengths.tolist()}")
        if getattr(self.config, 'strict_data', True):
            raise RuntimeError(
                detail + ". Lower max_bpe_tokens or raise compressed_frames, then "
                "re-run scripts/validate_dataset.py. Set strict_data=False to "
                "train through this."
            )
        self.logger.warning(detail + " (these contribute no gradient)")

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

    def decode_loader(self, loader, max_batches=None):
        """Greedy-decode a whole loader. Returns (targets, predictions) as text."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                videos = batch['videos']
                video_lengths = batch['video_lengths']
                text_labels = batch['text_labels']

                with self._autocast():
                    outputs = self.model(videos, video_lengths)   # (B, T, C)

                _, max_indices = torch.max(outputs, dim=2)
                max_indices = max_indices.cpu().numpy()  # (B, T)

                # Decode only over the model's real CTC time steps
                input_lengths_np = self.get_input_lengths(outputs, video_lengths).cpu().numpy()

                for i in range(len(max_indices)):
                    sequence = max_indices[i][:input_lengths_np[i]]

                    # Remove blanks and collapse repeats
                    decoded = []
                    previous = None
                    for idx in sequence:
                        if idx != 0 and idx != previous:  # 0 is blank token
                            decoded.append(idx)
                        previous = idx

                    all_predictions.append(self._ids_to_text(decoded))
                    all_targets.append(text_labels[i])

        return all_targets, all_predictions

    def calculate_wer_cer(self, epoch):
        """Calculate comprehensive evaluation metrics"""
        interval = max(1, getattr(self.config, 'metrics_interval', 5))
        if epoch % interval != 0:
            return None

        all_targets, all_predictions = self.decode_loader(
            self.val_loader, max_batches=self.val_batches
        )
        metrics = calculate_all_metrics(all_targets, all_predictions)

        # A few decoded examples, because falling loss alone cannot distinguish
        # real learning from CTC collapsing to all-blank (which decodes to "")
        empty = sum(1 for p in all_predictions if not p.strip())
        self.logger.info(
            f"Decoded {len(all_predictions)} val samples, {empty} of them empty "
            f"({100 * empty / max(1, len(all_predictions)):.0f}%)"
        )
        for target, prediction in list(zip(all_targets, all_predictions))[:3]:
            self.logger.info(f"  target: {target[:80]}")
            self.logger.info(f"  pred  : {prediction[:80]}")

        return metrics
