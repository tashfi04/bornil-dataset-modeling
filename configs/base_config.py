import torch
import os

class BaseConfig:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    csv_path = "/kaggle/input/bornil-bdsl-video-dataset/video_data.csv"
    chunk_base_path = "/kaggle/input/bornil-bdsl-video-dataset"

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(repo_root, "data", "vocab.json")
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

    # Gradient Accumulation
    gradient_accumulation_steps = 1  # 1 = no accumulation
    
    # Logging
    log_interval = 10

config = BaseConfig()
