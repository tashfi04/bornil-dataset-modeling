import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

class ViViT_CTC_HF(nn.Module):
    """ViViT model using HuggingFace pre-trained weights with CTC head"""

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Load pre-trained ViViT
        model_name = getattr(config, 'vivit_model_name', 'google/vivit-b-16x2-kinetics400')
        self.vivit = VivitModel.from_pretrained(model_name)

        # Freeze backbone initially (configurable)
        if getattr(config, 'freeze_backbone', True):
            for param in self.vivit.parameters():
                param.requires_grad = False

        # Get hidden size from ViViT
        hidden_size = self.vivit.config.hidden_size

        # CTC classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, video_lengths=None):
        # x: (B, C, T, H, W) - T should be close to num_frames due to sampling
        batch_size, channels, num_frames, height, width = x.size()

        # Reorder for ViViT: (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward pass through pre-trained ViViT
        outputs = self.vivit(x)

        # Use the last hidden states (sequence output)
        # Shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Classification
        logits = self.classifier(sequence_output)  # (B, T, num_classes)

        # Log softmax for CTC
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return log_probs.permute(1, 0, 2)  # (T, B, C) for CTC

    def unfreeze_backbone(self):
        """Unfreeze ViViT backbone for fine-tuning"""
        for param in self.vivit.parameters():
            param.requires_grad = True
