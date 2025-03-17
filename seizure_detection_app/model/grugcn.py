import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # Batch matrix multiplication: (batch_size, num_nodes, in_features) @ (in_features, out_features)
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), -1, -1))
        output = torch.bmm(adj, support)  # (batch_size, num_nodes, out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GRUGCNModel(nn.Module):
    def __init__(self, input_size, nhid, nclass, num_layers=1, dropout=0.2):
        super(GRUGCNModel, self).__init__()
        self.hidden_size = nhid
        self.num_layers = num_layers
        self.dropout = dropout

        # GRU Layer for Temporal Feature Extraction (working along the time axis for each channel)
        self.gru = nn.GRU(input_size, nhid, num_layers, batch_first=True, dropout=dropout)

        # GCN Layer for Spatial Feature Extraction (working across channels)
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.attention_fc = nn.Linear(nhid, 1)
        # Fully connected layers to combine the features
        self.fc1 = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        batch_size, num_channels, num_features = x.shape

        # Initialize hidden state with zeros for GRU
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # GRU expects (batch_size, num_features, num_channels)
        # So we directly feed the input to GRU without permuting (batch_size, 9, 1281)
        gru_out, _ = self.gru(x, h0)  # Output shape: (batch_size, num_channels, 256 512)

        # Transpose back to (batch_size, num_channels, nhid)
        # GCN for spatial relationships between channels
        x = F.relu(self.gc1(gru_out, adj))  # GCN operates over the channel dimension

        x = F.relu(self.gc2(x, adj))
        # Pooling across channels (nodes) to get a single feature vector per sample
        attention_scores = F.softmax(self.attention_fc(x), dim=1)
        gcn_out = torch.sum(attention_scores * x, dim=1)  # sum pooling across the channels

        # Fully connected layers
        out = F.relu(self.fc1(gcn_out))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)

        return out