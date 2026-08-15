import os
from configs.base_config import config as base_config

class TestViViTCTCConfig:
# Model architecture
    model_type = "vivit_ctc_test"
    vivit_model_name = "google/vivit-b-16x2-kinetics400"

    # Model parameters
    freeze_backbone = True

    # Training
    learning_rate = 5e-5
    batch_size = 2
    gradient_accumulation_steps = 4  # 1 = no accumulation
    num_epochs = 3
    warmup_epochs = 2
    metrics_interval = 1  # short run, so report metrics every epoch

    # Fewer frames read than production for faster decoding, but the same
    # compressed_frames so the position-embedding interpolation path and the
    # CTC axis match the real run.
    frame_size = (160, 160)
    num_frames = 80
    compressed_frames = 64
    sampling_strategy = "strategic"
    sampling_segments = 6

    gradient_checkpointing = True

    tokenization_type = "bpe"  # or "character"

    valid_samples_path = os.path.join(base_config.repo_root, "data", "valid_samples_vivit.json")

class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))

        # Override with test-specific attributes
        test_config = TestViViTCTCConfig()
        for key in dir(test_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(test_config, key))

        # Set test-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
