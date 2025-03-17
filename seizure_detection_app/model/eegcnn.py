import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGCNN(nn.Module):
    def __init__(
            self,
            num_classes,
            num_channels=12,
            dropout=0.5
    ):
        """
        CNN model for EEG classification.

        Args:
            num_channels (int): Number of EEG channels.
            seq_length (int): Length of the EEG sequence.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
        """
        super(EEGCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduces seq_length by 2

        # Second Convolutional Block
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Reduces seq_length by 2

        # Third Convolutional Block
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Reduces seq_length by 2

        # Fourth Convolutional Block
        self.conv4 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        # No pooling after conv4

        # Adaptive Pooling to ensure fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)  # Output size = 1

        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(256, num_classes)
        # Note: No activation here as we'll apply CrossEntropyLoss which includes Softmax

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for convolutional and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_length).

        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, num_classes).
        """
        # First Convolutional Block
        x = self.conv1(x)  # (batch_size, 64, seq_length)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 64, seq_length / 2)

        # Second Convolutional Block
        x = self.conv2(x)  # (batch_size, 128, seq_length / 2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 128, seq_length / 4)

        # Third Convolutional Block
        x = self.conv3(x)  # (batch_size, 256, seq_length / 4)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # (batch_size, 256, seq_length / 8)

        # Fourth Convolutional Block
        x = self.conv4(x)  # (batch_size, 512, seq_length / 8)
        x = self.bn4(x)
        x = self.relu4(x)
        # No pooling after conv4

        # Adaptive Pooling
        x = self.adaptive_pool(x)  # (batch_size, 512, 1)

        # Flatten the features
        x = x.view(x.size(0), -1)  # (batch_size, 512)

        # Fully Connected Layers
        x = self.fc1(x)  # (batch_size, 256)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        logits = self.fc2(x)  # (batch_size, num_classes)

        return logits
