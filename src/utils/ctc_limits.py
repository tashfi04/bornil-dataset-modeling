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


def required_ctc_steps(ids):
    """Minimum number of CTC time steps needed to emit `ids`.

    A repeated label needs a blank between the two emissions, otherwise the
    collapse step would merge them. So each adjacent duplicate pair costs one
    extra step on top of the sequence length. Filtering on length alone lets
    through targets CTC cannot align, which zero_infinity=True then turns into a
    silent zero loss.
    """
    duplicates = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return len(ids) + duplicates
