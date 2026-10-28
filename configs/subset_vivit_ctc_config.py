import os
from configs.base_config import config as base_config

class SubsetViViTCTCConfig:
    """Train and evaluate on a ~2,000-sentence subset of the dataset.

    A full training run with the prod settings, on a dataset small enough for
    its frames to be cached on Kaggle, to see how the model generalises before
    committing GPU time to all 17,988 recordings. The subset uses the full
    dataset's tokenizer, split and valid-sample list, all of which are keyed by
    recording, so each sentence keeps its split from the full dataset.
    """

    # Model architecture
    model_type = "vivit_ctc_subset"
    vivit_model_name = "google/vivit-b-16x2-kinetics400"
    # Read features from this encoder layer and drop the later ones. The last
    # layers of the Kinetics checkpoint condense a clip into one summary and lose
    # the variation over time that CTC needs; scripts/diagnose_temporal.py
    # measures this per layer.
    vivit_feature_layer = 7

    # Model parameters
    # The whole model trains from the first step, as in the overfit check. The
    # prod config instead starts with the backbone frozen; use_frozen_backbone()
    # below switches to that recipe for comparison.
    freeze_backbone = False

    # Training, as in the prod config
    learning_rate = 5e-5
    batch_size = 4
    gradient_accumulation_steps = 4  # 1 = no accumulation
    num_epochs = 100
    warmup_epochs = 10
    early_stopping_patience = 10

    # Paths. The subset CSV has the same columns as the full one and points at
    # the same videos. scripts/subset_vivit_ctc.py --csv overrides csv_path.
    csv_path = "/kaggle/input/datasets/tashfi04/bornil-bdsl-video-dataset/video_data.csv"
    chunk_base_path = "/kaggle/input/datasets/tashfi04/bornil-bdsl-video-dataset"

    # Written outside the repo so re-cloning does not delete the checkpoints.
    # Kaggle still discards this when a session ends unless the version is saved.
    output_dir = "/kaggle/working/outputs"

    # Built by scripts/preprocess_cache_frames.py --config subset. Training
    # checks every clip is present before starting; --no-cache decodes the
    # videos instead.
    cached_frames_path = "/kaggle/working/frame_cache_subset"

    # Video processing, as in the prod config
    frame_size = (224, 224)
    num_frames = 160        # frames read from each video
    compressed_frames = 64  # frames handed to ViViT after compression
    sampling_strategy = "strategic"
    sampling_segments = 6

    gradient_checkpointing = True

    tokenization_type = "bpe"  # or "character"

    # The full dataset's list. Validity depends only on each recording's video
    # and text, so it applies to any subset of the rows.
    valid_samples_path = os.path.join(base_config.repo_root, "data", "valid_samples_vivit.json")

class CombinedConfig:
    def __init__(self):
        # Copy all base config attributes
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))

        # Override with subset-specific attributes
        subset_config = SubsetViViTCTCConfig()
        for key in dir(subset_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(subset_config, key))

        # Set subset-specific output directory
        self.model_output_dir = os.path.join(self.output_dir, self.model_type)
        os.makedirs(self.model_output_dir, exist_ok=True)

def use_frozen_backbone(config):
    """Switch to the prod config's recipe, for comparison with full training.

    ViViT starts frozen and is unfrozen at a tenth of the learning rate once
    validation loss plateaus. Outputs go to their own directory, so the two runs
    never overwrite each other's checkpoints or results.
    """
    config.freeze_backbone = True
    config.trainable_position_embeddings = False
    config.model_type = SubsetViViTCTCConfig.model_type + "_frozen"
    config.model_output_dir = os.path.join(config.output_dir, config.model_type)
    os.makedirs(config.model_output_dir, exist_ok=True)

# Global config instance
config = CombinedConfig()
