import os
from configs.base_config import config as base_config

class OverfitViViTCTCConfig:
    """Deliberately overfit a small fixed subset, as a capability check.

    This answers one question: can the architecture learn this mapping at all?
    A model that cannot memorise 200 clips will not learn 13,000 either, so a
    failure here means something is wrong that more data and more epochs cannot
    fix. It is not a quality measurement - validation numbers are expected to
    get worse, because memorising is the point.
    """

# Model architecture
    model_type = "vivit_ctc_overfit"
    vivit_model_name = "google/vivit-b-16x2-kinetics400"

    # Model parameters
    # Everything trains from the first step. With a frozen backbone the trainer
    # unfreezes it once validation loss plateaus, then swaps to a scheduler that
    # halves the learning rate whenever validation loss stalls. Here validation
    # loss is meant to rise, so both would fire and starve the memorisation.
    freeze_backbone = False

    # Training
    # No accumulation, so every batch is an optimizer step: 100 per epoch
    # instead of 25. The rate is higher than a real run because fast
    # memorisation is the goal, but kept moderate since the whole pretrained
    # transformer is updating.
    learning_rate = 1e-4
    batch_size = 2
    gradient_accumulation_steps = 1  # 1 = no accumulation
    num_epochs = 50
    warmup_epochs = 5
    metrics_interval = 5

    # Decode the clips being memorised rather than the held-out ones, so each
    # metrics pass shows directly whether the model can reproduce them
    metrics_split = 'train'

    # Validation loss is supposed to get worse as the model memorises, so early
    # stopping would cut the run short at exactly the wrong moment.
    early_stopping_patience = 10 ** 6

    # Paths
    csv_path = "/kaggle/input/bornil-bdsl-video-dataset/video_data.csv"
    chunk_base_path = "/kaggle/input/bornil-bdsl-video-dataset"

    # Written outside the repo so re-cloning does not delete the checkpoints.
    # Kaggle still discards this when a session ends unless the version is saved.
    output_dir = "/kaggle/working/outputs"

    # The same 200 clips every epoch. Batch caps cannot be used for this: they
    # reshuffle, so the model would never see a sample twice.
    train_subset_size = 200
    val_subset_size = 40

    # Matches the test config, so the position-embedding interpolation path and
    # the CTC axis are the same ones a real run uses.
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

        # Override with overfit-specific attributes
        overfit_config = OverfitViViTCTCConfig()
        for key in dir(overfit_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(overfit_config, key))

        # Set overfit-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
