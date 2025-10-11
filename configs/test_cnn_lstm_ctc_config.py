import os
from configs.base_config import config as base_config

class TestConfig:
    # Model architecture (lightweight for testing)
    model_type = "cnn_lstm_ctc_test"
    cnn_backbone = "resnet18"
    lstm_hidden_size = 128
    lstm_layers = 1
    dropout = 0.3
    freeze_cnn_initially = True

    # Training adjustments for testing
    learning_rate = 1e-4
    batch_size = 8          # Increased to utilize GPU better
    num_epochs = 5          # Just 5 epochs for testing

    # Video processing for testing
    frame_size = (112, 112)
    max_frames = 300

# Create a combined config
class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))

        # Override with test-specific attributes
        test_config = TestConfig()
        for key in dir(test_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(test_config, key))

        # Set test-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
