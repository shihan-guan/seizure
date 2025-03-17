import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

# === Define Bipolar Channels ===

# List of bipolar channels (each channel is a bipolar pair)
BIPOLAR_CHANNELS = [
    'Fp2-T4',
    'T4-O2',
    'Fp2-C4',
    'C4-O2',
    'T4-C4',
    'C4-Cz',
    'Cz-C3',
    'C3-T3',
    'Fp1-T3',
    'T3-O1',
    'Fp1-C3',
    'C3-O1'
]

num_channels = len(BIPOLAR_CHANNELS)  # Should be 12

# Create a mapping from bipolar channels to indices
channel_to_idx = {ch: idx for idx, ch in enumerate(BIPOLAR_CHANNELS)}

# === Construct the Adjacency Matrix ===

# Define edges between bipolar channels based on their relationships
edges = [
    ('Fp2-T4', 'T4-O2'),
    ('Fp2-T4', 'Fp2-C4'),
    ('T4-O2', 'C4-O2'),
    ('Fp2-C4', 'C4-O2'),
    ('T4-C4', 'T4-O2'),
    ('T4-C4', 'C4-Cz'),
    ('C4-Cz', 'Cz-C3'),
    ('Cz-C3', 'C3-T3'),
    ('C3-T3', 'T3-O1'),
    ('Fp1-T3', 'T3-O1'),
    ('Fp1-T3', 'Fp1-C3'),
    ('Fp1-C3', 'C3-O1'),
    ('C3-O1', 'T3-O1')
]

edges_indices = [(channel_to_idx[src], channel_to_idx[dst]) for src, dst in edges]

# Function to create the adjacency matrix
def create_adjacency_matrix(num_nodes, edges_indices):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for src_idx, dst_idx in edges_indices:
        A[src_idx, dst_idx] = 1.0
        A[dst_idx, src_idx] = 1.0  # Assuming undirected graph
    return A

# Create the adjacency matrix
adjacency_matrix = create_adjacency_matrix(num_channels, edges_indices)

# Function to normalize the adjacency matrix
def normalize_adjacency_matrix(A):
    D = torch.diag(torch.sum(A, dim=1))
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D + 1e-5))  # Add epsilon for numerical stability
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    return A_normalized

# Normalize the adjacency matrix
normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)

# === Define the Graph Convolutional Network (GCN) ===

# Graph Convolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # (num_nodes, out_features)
        output = torch.matmul(adj, support)         # (num_nodes, out_features)
        return output

# BrainGCN module
class BrainGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, adj):
        super(BrainGCN, self).__init__()
        self.adj = adj
        self.gc1 = GraphConvolution(in_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, num_channels, seq_length = x.size()
        self.adj = self.adj.to(x.device)
        x = x.mean(dim=2)  # Shape: (batch_size, num_channels)

        outputs = []
        for i in range(batch_size):
            node_features = x[i].unsqueeze(-1)  # Shape: (num_channels, 1)
            h = self.relu(self.gc1(node_features, self.adj))
            h = self.gc2(h, self.adj)
            h = h.mean(dim=0)  # Shape: (out_dim)
            outputs.append(h)
        out = torch.stack(outputs)  # Shape: (batch_size, out_dim)
        return out

# === CNN-Based Patch Embedding Module ===

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

    def forward(self, x):
        batch_size, channels, seq_length = x.size()
        patch_size = self.patch_size
        num_patches = seq_length // patch_size

        # Ensure the sequence length is divisible by patch_size
        x = x[:, :, :num_patches * patch_size]
        x = x.view(batch_size, channels, num_patches, patch_size)  # (B, C, N_patches, patch_size)

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

        # Reshape back to (B, N_patches, self.embed_dim)
        x = x.view(batch_size, num_patches, self.embed_dim)

        return x

# === EEG Transformer CNN Model with GCN Integration ===

class EEGTransformerCNN(nn.Module):
    def __init__(
            self,
            num_channels,    # Should be 12
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

        # Use BIPOLAR_CHANNELS and adjacency_matrix defined earlier
        self.BIPOLAR_CHANNELS = BIPOLAR_CHANNELS
        self.channel_to_idx = channel_to_idx
        self.adj = normalized_adjacency

        # CNN-Based Patch Embedding
        self.cnn_patch_embed = CNNPatchEmbedding(
            num_channels=num_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        num_patches = seq_length // patch_size

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

        # GCN Module
        gcn_hidden_dim = 64
        gcn_out_dim = 128
        self.gcn = BrainGCN(
            in_features=1,
            hidden_dim=gcn_hidden_dim,
            out_dim=gcn_out_dim,
            adj=self.adj
        )

        # Combine CNN-Transformer and GCN outputs
        combined_dim = embed_dim + gcn_out_dim
        self.norm = nn.LayerNorm(combined_dim)
        self.fc = nn.Linear(combined_dim, num_classes)

        # Initialize weights
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
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # CNN Patch Embedding
        x_cnn = self.cnn_patch_embed(x)  # (batch_size, num_patches, embed_dim)
        num_patches = x_cnn.size(1)

        # Prepare CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x_cnn = torch.cat((cls_tokens, x_cnn), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add Positional Embeddings
        x_cnn = x_cnn + self.pos_embed[:, :num_patches + 1, :]
        x_cnn = self.pos_drop(x_cnn)

        # Transformer Encoder
        x_cnn = x_cnn.permute(1, 0, 2)  # (num_patches + 1, batch_size, embed_dim)
        x_cnn = self.transformer_encoder(x_cnn)  # (num_patches + 1, batch_size, embed_dim)
        x_cnn = x_cnn.permute(1, 0, 2)  # (batch_size, num_patches + 1, embed_dim)

        # Extract CLS Token for Classification
        cls_output = x_cnn[:, 0, :]  # (batch_size, embed_dim)

        # GCN Module
        x_gcn = self.gcn(x)  # (batch_size, gcn_out_dim)

        # Combine CNN-Transformer and GCN outputs
        combined = torch.cat((cls_output, x_gcn), dim=1)  # (batch_size, combined_dim)

        # Layer Normalization and Classification Head
        combined = self.norm(combined)
        logits = self.fc(combined)  # (batch_size, num_classes)

        return logits

# === EEG Dataset Class ===
#
# class EEGDataset(Dataset):
#     def __init__(self, file_paths, labels, seq_length):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.BIPOLAR_CHANNELS = BIPOLAR_CHANNELS
#         self.channel_to_idx = channel_to_idx
#         self.seq_length = seq_length
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         label = self.labels[idx]
#         # Load data from file
#         try:
#             data_dict = np.load(file_path, allow_pickle=True).item()
#             segment = data_dict['data'].astype(np.float32)
#             # Reorder channels
#             ordered_indices = [self.channel_to_idx[ch] for ch in self.BIPOLAR_CHANNELS]
#             segment = segment[ordered_indices, :]  # Shape: (num_channels, seq_length)
#             # Ensure the segment has the correct sequence length
#             segment = segment[:, :self.seq_length]
#         except Exception as e:
#             print(f"Error loading file {file_path}: {e}")
#             segment = np.zeros((len(self.BIPOLAR_CHANNELS), self.seq_length), dtype=np.float32)
#         return torch.from_numpy(segment), torch.tensor(label, dtype=torch.long)
