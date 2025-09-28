from src.training.ctc_trainer import CTCTrainer
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC

class CNNBiLSTMTrainer(CTCTrainer):
    def setup_model(self):
        """Setup CNN-BiLSTM model"""
        self.model = CNNBiLSTMCTC(
            num_classes=self.num_classes,
            cnn_backbone=self.config.cnn_backbone,
            lstm_hidden_size=self.config.lstm_hidden_size,
            lstm_layers=self.config.lstm_layers,
            dropout=self.config.dropout,
            freeze_cnn=self.config.freeze_cnn_initially
        ).to(self.config.device)
        
        self.logger.info(f"CNN-BiLSTM parameters: {sum(p.numel() for p in self.model.parameters()):,}")
