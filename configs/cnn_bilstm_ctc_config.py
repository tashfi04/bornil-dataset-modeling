import os
from configs.base_config import config as base_config

class CNNBiLSTMCTCConfig:
    # CNN architecture
    model_type = "cnn_lstm_ctc"

    # CNN parameters
    cnn_backbone = "resnet34"
    lstm_hidden_size = 256
    lstm_layers = 2
    dropout = 0.3
    freeze_cnn_initially = True

    # Training adjustments
    learning_rate = 1e-4
    batch_size = 4
    gradient_accumulation_steps = 4  # 1 = no accumulation
    num_epochs = 50

    # Video processing
    frame_size = (112, 112)
    num_frames = 160  # Explicitly set for CNN-BiLSTM
    sampling_strategy = "strategic"
    sampling_segments = 6

    tokenization_type = "character"  # or "bpe"

# Create a combined config
class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))

        # Override with model-specific attributes
        model_config = CNNBiLSTMCTCConfig()
        for key in dir(model_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(model_config, key))

        # Set model-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
