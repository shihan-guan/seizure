import torch
import torch.nn as nn
import torch.nn.functional as F

# Gradient Reversal Layer for adversarial training
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

# CNN-Based Patch Embedding Module
class CNNPatchEmbedding(nn.Module):
    def __init__(self, num_channels, patch_size, embed_dim):
        super(CNNPatchEmbedding, self).__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        batch_size, channels, seq_length = x.size()
        patch_size = self.patch_size
        num_patches = seq_length // patch_size

        # Ensure sequence length is divisible by patch_size
        x = x[:, :, :num_patches * patch_size]
        x = x.view(batch_size, channels, num_patches, patch_size)
        x = x.view(batch_size * num_patches, channels, patch_size)

        # Apply CNN layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        x = x.view(batch_size, num_patches, self.embed_dim)
        return x

# Revised EEG Transformer with CNN Patch Embedding and Adversarial Branch
class EEGTransformerCNN(nn.Module):
    def __init__(self, num_channels, seq_length, patch_size, embed_dim, num_heads, num_layers, num_classes, num_patients, dropout=0.1):
        super(EEGTransformerCNN, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # CNN-Based Patch Embedding
        self.cnn_patch_embed = CNNPatchEmbedding(num_channels=num_channels, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = seq_length // patch_size

        # CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(dim_feedforward=2048, d_model=embed_dim, nhead=num_heads, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        # Seizure Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)

        # Adversarial Patient Classification Branch
        self.patient_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_patients)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_length)
        Returns:
            During training: (seizure_logits, patient_logits)
            During evaluation: seizure_logits
        """
        batch_size = x.size(0)
        # CNN Patch Embedding
        x = self.cnn_patch_embed(x)  # (batch_size, num_patches, embed_dim)
        num_patches = x.size(1)

        # Prepare CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embeddings
        x = x + self.pos_embed[:, :num_patches + 1, :]
        x = self.pos_drop(x)

        # Transformer expects input of shape (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.norm(x)

        # Extract CLS Token for classification
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)

        # Seizure classification branch
        seizure_logits = self.fc(cls_output)

        if self.training:
            # Adversarial branch for patient classification with gradient reversal
            reversed_features = grad_reverse(cls_output, lambda_=1.0)
            patient_logits = self.patient_classifier(reversed_features)
            return seizure_logits, patient_logits
        else:
            return seizure_logits
