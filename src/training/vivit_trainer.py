from src.training.ctc_trainer import CTCTrainer
from src.models.vivit_ctc_hf import ViViT_CTC_HF
import torch

class ViViTTrainer(CTCTrainer):
    def setup_model(self):
        """Setup ViViT model with proper multi-GPU handling"""

        # Validate ViViT input requirements before creating model
        if self.config.frame_size != (224, 224):
            self.logger.error(f"ViViT requires 224x224 input, but config has {self.config.frame_size}")
            raise ValueError("ViViT frame_size must be (224, 224)")

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
        
    def setup_optimizer(self):
        """Setup optimizer with learning rate warmup for ViViT"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        warmup_epochs = getattr(self.config, 'warmup_epochs', 10)
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=warmup_epochs / self.config.num_epochs
        )

        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
