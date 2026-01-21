import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)   
        x = self.conv2(x, edge_index)
        return x


class SimCLRSubgraph(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLRSubgraph, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.conv2.out_channels, encoder.conv2.out_channels),
            nn.ReLU(),
            nn.Linear(encoder.conv2.out_channels, projection_dim)
        )
    
    def forward(self, x, edge_index):
        representations = self.encoder(x, edge_index)
        projections = self.projection_head(representations)
        return representations, projections


def simclr_loss(z_i, z_j, temperature=0.07):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature
    
    mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
    mask = torch.cat([torch.cat([torch.zeros_like(mask), mask], dim=1),
                      torch.cat([mask, torch.zeros_like(mask)], dim=1)], dim=0)
    
    pos = similarity_matrix[mask].reshape(batch_size, 1)
    neg = similarity_matrix[~mask].reshape(batch_size, -1)
    
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss