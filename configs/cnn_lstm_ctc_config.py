import os
from configs.base_config import config as base_config

class CNNLSTMCTCConfig:
    # Model architecture
    model_type = "cnn_lstm_ctc"
    cnn_backbone = "resnet18"
    lstm_hidden_size = 256
    lstm_layers = 2
    dropout = 0.3
    freeze_cnn_initially = True
    
    # Training adjustments specific to this model
    learning_rate = 1e-4
    batch_size = 4
    
    # Video processing for this model
    num_frames = 16
    frame_size = (112, 112)

# Create a combined config
class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))
        
        # Override with model-specific attributes
        model_config = CNNLSTMCTCConfig()
        for key in dir(model_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(model_config, key))

        # Set model-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
