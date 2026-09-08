import torch
import os

class BaseConfig:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    csv_path = "/bornil-bdsl-video-dataset/video_data.csv"
    chunk_base_path = "/bornil-bdsl-video-dataset"

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Character level tokenization
    vocab_path = os.path.join(repo_root, "data", "vocab.json")

    # BPE tokenization
    bpe_tokenizer_path = os.path.join(repo_root, "data", "bpe_tokenizer.json") # BPE output
    bpe_vocab_size = 2000
    # Sentences longer than this are dropped. CTC needs input_length >=
    # target_length and the ViViT-CTC time axis is compressed_frames // 2, so
    # this must stay at or below that. 32 tokens keeps 92.6% of the sentences
    # at a 2000-token vocabulary.
    max_bpe_tokens = 32

    # Cache root built by scripts/preprocess_cache_frames.py. Set this to skip
    # video decoding during training; None decodes on the fly.
    cached_frames_path = None

    # Per-video frame counts from scripts/scan_videos.py. Model-independent and
    # expensive to build, so it is scanned once and reused by every validation.
    video_stats_path = os.path.join(repo_root, "data", "video_stats.json")

    # Recordings cleared by scripts/validate_dataset.py. When set, training uses
    # exactly this list and nothing is filtered at runtime. Model configs override
    # this because the two models have different CTC limits.
    valid_samples_path = os.path.join(repo_root, "data", "valid_samples.json")

    # Abort on any unreadable sample or CTC length violation rather than dropping
    # it and training on partial data
    strict_data = True

    # Resume from checkpoints/last_checkpoint.pth if it exists. Set False to
    # force a fresh run, or point resume_from at a specific file.
    auto_resume = True
    resume_from = None

    # Every checkpoint is a few hundred MB, so only the latest and the best are
    # kept by default. Enable to also keep one file per epoch, and cap how many
    # of those are retained.
    keep_epoch_checkpoints = False
    max_epoch_checkpoints = 3

    # Cap batches per epoch. None means a full pass; a small number exercises the
    # whole training loop quickly, which matters while video is decoded live.
    max_train_batches = None
    max_val_batches = None

    train_val_test_split_path = os.path.join(repo_root, "data", "train_val_test_split.json")
    output_dir = os.path.join(repo_root, "outputs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data split
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random_seed = 42

    # Video processing (default values, overridden by model-specific configs)
    num_frames = 160        # Target number of frames read from every video
    sampling_strategy = "strategic"  # "uniform" or "strategic"; default "uniform"
    sampling_segments = 6  # For strategic sampling: number of segments
    frame_size = (112, 112) # (height, width)

    # Training (default values, overridden by model-specific configs)
    batch_size = 4
    num_workers = 4
    learning_rate = 1e-4
    num_epochs = 50
    early_stopping_patience = 10
    grad_clip = 5.0

    # Mixed precision, applied on CUDA only. The CTC loss stays in fp32.
    use_amp = True

    # Gradient Accumulation
    gradient_accumulation_steps = 1  # 1 = no accumulation

    # Logging
    log_interval = 10
    # How often to run WER/CER, which is expensive
    metrics_interval = 5

config = BaseConfig()
