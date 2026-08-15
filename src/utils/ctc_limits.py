"""How many CTC time steps a config produces, and how long a target may be.

Shared by scripts/validate_dataset.py and src/data_loader.py so the pre-validated
sample list and the loader always agree on what counts as valid.
"""

# Temporal span of the pretrained ViViT tubelet. The model reads the real value
# from the checkpoint and raises on a mismatch; this mirrors it for configs.
VIVIT_TUBELET_FRAMES = 2


def ctc_time_steps(config):
    """Number of CTC output steps the model emits.

    ViViT compresses to a fixed number of tubelets, so its axis is
    compressed_frames // tubelet. CNN-BiLSTM keeps one step per input frame.
    """
    compressed = getattr(config, 'compressed_frames', None)
    if compressed is not None:
        return compressed // VIVIT_TUBELET_FRAMES
    return config.num_frames


def target_token_limit(config):
    """Longest target sequence CTC can align under this config."""
    steps = ctc_time_steps(config)
    if getattr(config, 'tokenization_type', 'character') == 'bpe':
        return min(getattr(config, 'max_bpe_tokens', steps), steps)
    return steps
