import math

from src.training.ctc_trainer import CTCTrainer
from src.models.vivit_ctc_hf import ViViT_CTC_HF
import torch

class ViViTTrainer(CTCTrainer):
    def setup_model(self):
        """Setup ViViT model with proper multi-GPU handling"""

        # Sizes other than the pretrained 224 work by interpolating the position
        # embeddings inside the model
        frame_h, frame_w = self.config.frame_size
        if frame_h != frame_w or frame_h % 16:
            self.logger.error(f"ViViT needs square frames divisible by 16, got {self.config.frame_size}")
            raise ValueError("ViViT frame_size must be square and divisible by 16")
        if frame_h != 224:
            self.logger.info(
                f"Using {frame_h}x{frame_w} input (pretrained is 224x224); "
                f"position embeddings will be interpolated"
            )

        # Create the base model
        base_model = ViViT_CTC_HF(
            config=self.config,
            num_classes=self.num_classes
        )

        # Multi-GPU handling
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(base_model).to(self.config.device)
        else:
            self.model = base_model.to(self.config.device)

        # Log parameters (handle both single and multi-GPU cases)
        if hasattr(self.model, 'module'):  # DataParallel wrapper
            param_count = sum(p.numel() for p in self.model.module.parameters())
        else:
            param_count = sum(p.numel() for p in self.model.parameters())

        self.logger.info(f"ViViT parameters: {param_count:,}")

    def get_input_lengths(self, outputs, video_lengths):
        """Every clip is compressed to the same number of tubelets, so the CTC
        time axis is constant and unrelated to the raw frame count.

        outputs is (B, T, C).
        """
        batch_size, time_steps = outputs.size(0), outputs.size(1)
        return torch.full(
            (batch_size,), time_steps,
            dtype=torch.long, device=outputs.device
        )

    def setup_optimizer(self):
        """Setup optimizer with learning rate warmup for ViViT"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # train_epoch() steps the optimizer every gradient_accumulation_steps
        # batches plus once at the end of the epoch, so round up; OneCycleLR
        # raises if stepped past total_steps.
        warmup_epochs = getattr(self.config, 'warmup_epochs', 10)
        effective_steps = max(1, math.ceil(self.train_batches / self.gradient_accumulation_steps))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=effective_steps,
            pct_start=warmup_epochs / self.config.num_epochs
        )
        self.step_scheduler_per_batch = True

        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
