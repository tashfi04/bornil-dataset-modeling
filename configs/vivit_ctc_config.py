import os
from configs.base_config import config as base_config

class ViViTCTCConfig:
    # Model architecture
    model_type = "vivit_ctc"
    vivit_model_name = "google/vivit-b-16x2-kinetics400"
    
    # Model parameters
    freeze_backbone = True
    
    # Training
    learning_rate = 5e-5
    batch_size = 4
    gradient_accumulation_steps = 4  # 1 = no accumulation
    num_epochs = 100
    warmup_epochs = 10
    
    # Video processing
    frame_size = (224, 224)
    num_frames = 160
    sampling_strategy = "strategic"
    sampling_segments = 6

    tokenization_type = "bpe"  # or "character"

class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))

        # Override with ViViT-specific attributes
        vivit_config = ViViTCTCConfig()
        for key in dir(vivit_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(vivit_config, key))

        # Set model-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
