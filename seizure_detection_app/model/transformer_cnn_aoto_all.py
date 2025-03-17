import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN-Based Patch Embedding Module with Masking Applied Before Embedding
class CNNPatchEmbedding(nn.Module):
    def __init__(self, num_channels, patch_size, embed_dim):
        super(CNNPatchEmbedding, self).__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Define CNN layers to process each patch
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()

        # Final linear layer to get the desired embedding dimension
        self.fc = nn.Linear(256, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, num_channels, patch_size))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_length)
            mask: Tensor of shape (batch_size, num_patches) indicating which patches to mask

        Returns:
            x: Embedded tensor of shape (batch_size, num_patches, embed_dim)
        """
        batch_size, channels, seq_length = x.size()
        patch_size = self.patch_size
        num_patches = seq_length // patch_size

        # Ensure the sequence length is divisible by patch size
        x = x[:, :, :num_patches * patch_size]
        x = x.view(batch_size, channels, num_patches, patch_size)  # (B, C, N_patches, patch_size)

        if mask is not None:
            # Replace masked patches with mask token
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, num_patches, 1)
            mask_token = self.mask_token.expand(batch_size, -1, -1)  # (batch_size, channels, patch_size)
            mask_token = mask_token.unsqueeze(2)  # (batch_size, channels, 1, patch_size)
            x = x.where(~mask, mask_token)

        # Reshape to process each patch separately
        x = x.view(batch_size * num_patches, channels, patch_size)  # (B * N_patches, C, patch_size)

        # Apply CNN layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        # Global average pooling over the sequence length dimension
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B * N_patches, 256)

        # Final linear projection to embed_dim
        x = self.fc(x)  # (B * N_patches, embed_dim)

        # Reshape back to (B, N_patches, embed_dim)
        x = x.view(batch_size, num_patches, self.embed_dim)

        return x

    def get_patch_dim(self):
        return self.num_channels * self.patch_size  # Dimension of a patch before projection

# Revised EEG Transformer CNN Model with Classification and Pretext Tasks
class EEGTransformerCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        seq_length,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout=0.1
    ):
        super(EEGTransformerCNN, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # CNN-Based Patch Embedding
        self.cnn_patch_embed = CNNPatchEmbedding(
            num_channels=num_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        num_patches = seq_length // patch_size
        self.num_patches = num_patches

        # CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Normalization and Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        # Decoder for Pretext Task
        patch_dim = self.cnn_patch_embed.get_patch_dim()
        self.decoder = nn.Linear(embed_dim, patch_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def forward(self, x, task='classification', mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_length)
            task: 'classification' or 'pretext'
            mask: Tensor of shape (batch_size, num_patches) indicating which patches to mask (only for 'pretext' task)

        Returns:
            out: Output tensor
                - For 'classification': (batch_size, num_classes)
                - For 'pretext': (total_masked_patches, num_channels, patch_size)
        """
        # CNN Patch Embedding with masking
        x = self.cnn_patch_embed(x, mask=mask)  # (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :num_patches + 1, :]
        x = self.pos_drop(x)

        # Transformer expects input of shape (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (num_patches + 1, batch_size, embed_dim)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (num_patches + 1, batch_size, embed_dim)

        # Permute back to (batch_size, sequence_length, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, num_patches + 1, embed_dim)

        # Apply Layer Normalization
        x = self.norm(x)

        if task == 'classification':
            # Classification using the CLS token representation
            cls_output = x[:, 0, :]  # (batch_size, embed_dim)
            out = self.fc(cls_output)  # (batch_size, num_classes)
        elif task == 'pretext':
            # Use the outputs at the masked positions to reconstruct the masked patches
            # x[:, 1:, :] shape: (batch_size, num_patches, embed_dim)
            masked_embeddings = x[:, 1:, :][mask]  # (total_masked_patches, embed_dim)

            # Decode to reconstruct the masked patches
            reconstructed_patches = self.decoder(masked_embeddings)  # (total_masked_patches, patch_dim)

            # Reshape reconstructed patches to (total_masked_patches, num_channels, patch_size)
            num_channels = self.cnn_patch_embed.num_channels
            patch_size = self.cnn_patch_embed.patch_size
            reconstructed_patches = reconstructed_patches.view(-1, num_channels, patch_size)

            out = reconstructed_patches
        else:
            raise ValueError("Task must be 'classification' or 'pretext'")

        return out
