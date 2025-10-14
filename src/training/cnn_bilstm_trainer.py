from src.training.ctc_trainer import CTCTrainer
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
import torch

class CNNBiLSTMTrainer(CTCTrainer):
    def setup_model(self):
        """Setup CNN-BiLSTM model"""

        # Create the base model
        base_model = CNNBiLSTMCTC(
            num_classes=self.num_classes,
            cnn_backbone=self.config.cnn_backbone,
            lstm_hidden_size=self.config.lstm_hidden_size,
            lstm_layers=self.config.lstm_layers,
            dropout=self.config.dropout,
            freeze_cnn=self.config.freeze_cnn_initially
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

        self.logger.info(f"CNN-BiLSTM parameters: {param_count:,}")
