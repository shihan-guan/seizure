import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, concat=True, dropout=0.6):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_features, out_features // num_heads, heads=num_heads, concat=concat, dropout=dropout)

    def forward(self, x, edge_index):
        return self.gat_conv(x, edge_index)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_heads, dropout):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(nfeat, nhid, num_heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATLayer(nhid * num_heads, nclass, num_heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)