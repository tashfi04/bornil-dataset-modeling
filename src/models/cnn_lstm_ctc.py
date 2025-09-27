import torch
import torch.nn as nn
from torchvision.models import resnet18

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes):
        super(CNNBiLSTMCTC, self).__init__()
        
        # CNN backbone (ResNet-18)
        resnet = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Bi-LSTM
        self.lstm_hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet-18 feature size
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Linear(self.lstm_hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        batch_size, channels, timesteps, height, width = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(timesteps):
            frame = x[:, :, t, :, :]  # (B, C, H, W)
            features = self.cnn(frame)  # (B, 512, H', W')
            features = features.mean(dim=[2, 3])  # Global average pooling -> (B, 512)
            cnn_features.append(features)
        
        # Stack features -> (B, T, 512)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)  # (B, T, hidden_size * 2)
        
        # Classifier
        output = self.classifier(lstm_out)  # (B, T, num_classes)
        output = torch.log_softmax(output, dim=2)  # For CTC loss
        
        return output
