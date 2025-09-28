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
            cnn_model = resnet18(pretrained=True)
            self.cnn_feature_size = 512
        elif cnn_backbone == "resnet34":
            cnn_model = resnet34(pretrained=True)
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
        nn.init.constant_(self.classifier.bias, 0)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        batch_size, channels, timesteps, height, width = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(timesteps):
            frame = x[:, :, t, :, :]  # (B, C, H, W)
            features = self.cnn(frame)  # (B, 512, H', W')
            features = self.adaptive_pool(features)  # (B, 512, 1, 1)
            features = features.view(batch_size, -1)  # (B, 512)
            cnn_features.append(features)
        
        # Stack features -> (B, T, 512)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM with packed sequence for efficiency
        lstm_out, _ = self.lstm(cnn_features)  # (B, T, hidden_size * 2)
        
        # Apply dropout and classifier
        lstm_out = self.dropout(lstm_out)
        output = self.classifier(lstm_out)  # (B, T, num_classes)
        
        # Log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning"""
        for param in self.cnn.parameters():
            param.requires_grad = True
