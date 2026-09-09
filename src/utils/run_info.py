"""Shared startup reporting for the training entry points."""
import os


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

    train_cap = getattr(config, 'max_train_batches', None)
    val_cap = getattr(config, 'max_val_batches', None)
    if train_cap or val_cap:
        print(f"Batch caps: train={train_cap or 'full'}, val={val_cap or 'full'} "
              f"(partial epochs - not a full training pass)")

    subsets = {name: getattr(config, f'{name}_subset_size', None)
               for name in ('train', 'val', 'test')}
    if any(subsets.values()):
        shown = ', '.join(f"{k}={v or 'full'}" for k, v in subsets.items())
        print(f"Fixed subsets: {shown} (same samples every epoch)")

    # Stated plainly because checkpoints are easy to lose track of when the
    # output directory is overridden per environment
    out_dir = getattr(config, 'model_output_dir', None)
    if out_dir:
        print(f"Output dir: {out_dir}")
        print(f"  checkpoints: {os.path.join(out_dir, 'checkpoints')}")
        print(f"  log: {os.path.join(out_dir, 'training.log')}")

    resume = getattr(config, 'resume_from', None)
    print(f"Resume: {resume if resume else ('auto' if getattr(config, 'auto_resume', True) else 'disabled')}")
