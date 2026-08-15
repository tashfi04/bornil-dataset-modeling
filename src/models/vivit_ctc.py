import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbedding3D(nn.Module):
    """Convert video into spatiotemporal patches"""

    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_frames = config.num_frames
        self.hidden_size = config.hidden_size

        # 3D convolution to extract spatiotemporal patches
        self.projection = nn.Conv3d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=(self.temporal_patch_size, config.patch_size, config.patch_size),
            stride=(self.temporal_patch_size, config.patch_size, config.patch_size),
            padding=0
        )

    def forward(self, x):
        # x: (B, C, T, H, W) - T can be variable
        x = self.projection(x)  # (B, hidden_size, T', H', W')

        # Flatten spatial and temporal dimensions
        x = x.flatten(2)  # (B, hidden_size, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_size)

        return x

class PositionalEncoding3D(nn.Module):
    """Add learnable positional encoding for spatiotemporal patches"""

    def __init__(self, config):
        super().__init__()

        # Use maximum expected sequence length
        max_patches = (config.num_frames // config.temporal_patch_size) * \
                     (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # x: (B, num_patches, hidden_size)
        batch_size, seq_len, hidden_size = x.shape

        # Add positional embeddings for current sequence length
        x = x + self.position_embeddings[:, :seq_len]
        x = self.dropout(x)

        return x

class ViViTAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key"
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class ViViTLayer(nn.Module):
    """Complete ViViT transformer layer"""

    def __init__(self, config):
        super().__init__()
        self.attention = ViViTAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # Self-attention with residual connection
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = hidden_states + attention_output

        # Feed-forward with residual connection
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = F.gelu(layer_output)
        layer_output = self.output(layer_output)
        layer_output = self.dropout(layer_output)

        hidden_states = hidden_states + layer_output
        return hidden_states

class ViViTEncoder(nn.Module):
    """Stack of ViViT layers"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([
            ViViTLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

class ViViT_CTC(nn.Module):
    """ViViT model with CTC for sequence-to-sequence learning"""

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embedding = PatchEmbedding3D(config)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Positional encoding
        self.positional_encoding = PositionalEncoding3D(config)

        # Transformer encoder
        self.encoder = ViViTEncoder(config)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, PatchEmbedding3D):
            # He initialization for conv layers
            n = module.projection.kernel_size[0] * module.projection.kernel_size[1] * module.projection.out_channels
            torch.nn.init.normal_(module.projection.weight, mean=0.0, std=math.sqrt(2.0 / n))

    def forward(self, x, video_lengths=None):
        # x: (B, C, T, H, W) - T is variable
        batch_size, channels, num_frames, height, width = x.size()

        # Extract spatiotemporal patches
        patch_embeddings = self.patch_embedding(x)  # (B, num_patches, hidden_size)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)

        # Transformer encoding
        encoded = self.encoder(embeddings)  # (B, num_patches+1, hidden_size)

        # Use all patch representations (excluding CLS) for temporal modeling
        patch_representations = encoded[:, 1:]  # (B, num_patches, hidden_size)

        # Calculate temporal sequence length dynamically
        num_spatial_patches = (height // self.config.patch_size) * (width // self.config.patch_size)
        temporal_length = num_frames // self.config.temporal_patch_size

        # Reshape to separate temporal and spatial dimensions
        temporal_sequence = patch_representations.view(
            batch_size, temporal_length, num_spatial_patches, self.config.hidden_size
        )

        # Average over spatial patches to get temporal sequence
        temporal_features = temporal_sequence.mean(dim=2)  # (B, T, hidden_size)

        # Classification
        logits = self.classifier(temporal_features)  # (B, T, num_classes)

        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs.permute(1, 0, 2)  # (T, B, C)
