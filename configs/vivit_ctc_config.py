from configs.base_config import config as base_config

class ViViTCTCConfig:
    model_type = "vivit_ctc"
    model_name = "google/vivit-b-16x2-kinetics400"
    
    # Architecture-specific
    num_frames = 32  # ViViT can handle more frames
    frame_size = (224, 224)
    
    # Training adjustments
    learning_rate = 5e-5
    batch_size = 2
    
    # ViViT-specific parameters
    use_pretrained = True
    trainable_layers = 4

class CombinedConfig:
    def __init__(self):
        for key in dir(base_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(base_config, key))
        
        model_config = ViViTCTCConfig()
        for key in dir(model_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(model_config, key))

config = CombinedConfig()
