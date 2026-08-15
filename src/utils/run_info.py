"""Shared startup reporting for the training entry points."""


def print_run_summary(config, title):
    """Print the settings that determine what a training run actually does."""
    print(f"=== {title} ===")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Device: {config.device}")

    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    print(f"Frames read per video: {config.num_frames}")
    print(f"Frame size: {config.frame_size}")
    print(f"Sampling: {config.sampling_strategy} ({config.sampling_segments} segments)")

    # ViViT compresses the frames it reads, which fixes the CTC time axis
    compressed = getattr(config, 'compressed_frames', None)
    if compressed is not None:
        print(f"Compressed to: {compressed} frames")

    tokenization = getattr(config, 'tokenization_type', 'character')
    print(f"Tokenization: {tokenization}")
    if tokenization == 'bpe':
        print(f"  BPE vocab size: {config.bpe_vocab_size}")
        print(f"  Max BPE tokens: {config.max_bpe_tokens} (longer sentences are dropped)")

    print(f"Mixed precision: {getattr(config, 'use_amp', False)}")
    print(f"Gradient checkpointing: {getattr(config, 'gradient_checkpointing', False)}")

    cache = getattr(config, 'cached_frames_path', None)
    print(f"Frame cache: {cache if cache else 'disabled (decoding video on the fly)'}")
