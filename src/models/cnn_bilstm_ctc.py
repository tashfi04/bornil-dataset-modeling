import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
import torch.nn.functional as F

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, cnn_backbone="resnet18", lstm_hidden_size=256, 
                 lstm_layers=2, dropout=0.3, freeze_cnn=True):
        super(CNNBiLSTMCTC, self).__init__()
        
        # CNN backbone
        if cnn_backbone == "resnet18":
            cnn_model = resnet18(weights="IMAGENET1K_V1")  # FIXED: modern API
            self.cnn_feature_size = 512
        elif cnn_backbone == "resnet34":
            cnn_model = resnet34(weights="IMAGENET1K_V1")  # FIXED: modern API
            self.cnn_feature_size = 512
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Remove the final layers
        self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
        
        # Freeze CNN if requested
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Adaptive pooling to handle different spatial sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bi-LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        
    def forward(self, x, video_lengths=None):
        """
        Args:
            x: padded video tensor of shape (B, C, T, H, W)
            video_lengths: actual lengths of each video in the batch
        """
        batch_size, channels, timesteps, height, width = x.size()

        # Video length validation to ensure sampling is working properly
        if video_lengths is not None and (video_lengths > timesteps).any():
            invalid_indices = (video_lengths > timesteps).nonzero().squeeze()
            print(f"🚨 ERROR: Video lengths {video_lengths[invalid_indices]} exceed temporal dimension {timesteps}")

        # Use the device of the input tensor to ensure consistency between multipleGPUs
        device = x.device

        # OPTIMIZATION: Process all frames in one batch
        # Reshape to (B*T, C, H, W) - combine batch and time dimensions
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x_flat = x_flat.reshape(-1, channels, height, width)  # (B*T, C, H, W)

        # Process all frames through CNN at once
        features = self.cnn(x_flat)  # (B*T, 512, H', W')
        features = self.adaptive_pool(features)  # (B*T, 512, 1, 1)
        features = features.flatten(1)  # (B*T, 512)

        # Reshape back to (B, T, 512)
        cnn_features = features.reshape(batch_size, timesteps, -1)

        # Use packed sequences for variable length
        if video_lengths is not None:
            # Ensure video_lengths is on CPU for pack_padded_sequence
            # Pack the sequence to ignore padding
            packed_input = nn.utils.rnn.pack_padded_sequence(
                cnn_features, 
                video_lengths.to('cpu'),  # Always use CPU for lengths
                batch_first=True, 
                enforce_sorted=True
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=timesteps
            )
        else:
            # Fallback to standard LSTM if no lengths provided
            lstm_out, _ = self.lstm(cnn_features) # (B, T, hidden_size * 2)

        # Apply dropout and classifier
        lstm_out = self.dropout(lstm_out)
        output = self.classifier(lstm_out)  # (B, T, num_classes)

        # Log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        # Return (T, B, C) directly for CTC loss
        return output.permute(1, 0, 2)
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning"""

        if hasattr(self, 'module'):  # DataParallel wrapper
            # Access the underlying model
            for param in self.module.cnn.parameters():
                param.requires_grad = True
        else:
            # Single GPU
            for param in self.cnn.parameters():
                param.requires_grad = True
