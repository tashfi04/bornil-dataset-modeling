import torch

class BaseConfig:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    csv_path = "/kaggle/input/bornil-bdsl-video-dataset/video_data.csv"  # Update this
    chunk_base_path = "/kaggle/input/bornil-bdsl-video-dataset/chunks"  # Update this
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random_seed = 42
    
    # Text Processing
    vocab_path = "data/vocab.json"
    
    # Video Processing
    num_frames = 16
    frame_size = (112, 112)  # (height, width)
    
    # Training
    batch_size = 4
    num_workers = 4
    learning_rate = 1e-4
    num_epochs = 50
    early_stopping_patience = 10

config = BaseConfig()
