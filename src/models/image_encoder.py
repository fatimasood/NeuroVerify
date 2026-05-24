"""
GNN-based image encoder with ResNet backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class GraphNodeEncoder(nn.Module):
    """Node encoder for graph neural network"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super(GraphNodeEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        return F.relu(self.norm(self.fc(x)))


class GNNLayer(nn.Module):
    """Single GNN layer with message passing"""
    
    def __init__(self, hidden_dim):
        super(GNNLayer, self).__init__()
        self.message_fc = nn.Linear(hidden_dim, hidden_dim)
        self.update_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: (num_nodes, hidden_dim)
            adjacency_matrix: (num_nodes, num_nodes)
        """
        # Message passing
        messages = torch.mm(adjacency_matrix, self.message_fc(node_features))
        
        # Update nodes
        combined = torch.cat([node_features, messages], dim=1)
        updated = F.relu(self.update_fc(combined))
        updated = self.norm(updated)
        
        return updated


class GNNImageEncoder(nn.Module):
    """GNN-based image encoder with ResNet backbone"""
    
    def __init__(self, backbone='resnet50', feature_dim=2048, node_dim=128, 
                 num_nodes=16, num_gnn_layers=3):
        super(GNNImageEncoder, self).__init__()
        
        # Load pretrained ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial feature extraction
        self.spatial_pool = nn.AdaptiveAvgPool2d((num_nodes, num_nodes))
        
        # Node encoder
        self.node_encoder = GraphNodeEncoder(feature_dim, node_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_dim) for _ in range(num_gnn_layers)
        ])
        
        # Output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(node_dim * num_nodes, 512)
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
    def extract_regions(self, x):
        """Extract region features using ResNet"""
        # Get feature maps
        features = self.backbone(x)  # (batch, 2048, H, W)
        
        # Spatial pooling to get grid of features
        features = self.spatial_pool(features)  # (batch, 2048, num_nodes, num_nodes)
        
        batch_size = features.shape[0]
        features = features.view(batch_size, self.feature_dim, -1)  # (batch, 2048, num_nodes^2)
        features = features.permute(0, 2, 1)  # (batch, num_nodes^2, 2048)
        
        return features
    
    def build_adjacency_matrix(self, node_features):
        """Build adjacency matrix based on feature similarity"""
        # Normalize node features
        norm_features = F.normalize(node_features, dim=-1)
        
        # Compute similarity matrix
        adj = torch.mm(norm_features, norm_features.t())
        
        # Apply sigmoid to get probabilities
        adj = torch.sigmoid(adj)
        
        # Add self-loops
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        
        # Normalize adjacency matrix
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / (row_sum + 1e-8)
        
        return adj
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, 3, H, W) images
        Returns:
            features: (batch_size, 512) image features
            confidence: (batch_size,) confidence score
        """
        batch_size = x.shape[0]
        
        # Extract region features
        region_features = self.extract_regions(x)  # (batch, num_patches, 2048)
        
        # Encode nodes
        node_features = self.node_encoder(region_features)  # (batch, num_patches, node_dim)
        
        # Apply GNN layers for each sample in batch
        all_node_features = []
        for b in range(batch_size):
            nodes = node_features[b]  # (num_patches, node_dim)
            
            # Build adjacency matrix
            adj = self.build_adjacency_matrix(nodes)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                nodes = gnn_layer(nodes, adj)
            
            all_node_features.append(nodes)
        
        node_features = torch.stack(all_node_features, dim=0)  # (batch, num_patches, node_dim)
        
        # Global pooling over nodes
        node_features = node_features.view(batch_size, -1)  # (batch, num_patches * node_dim)
        
        # Output projection
        features = self.output_proj(node_features)
        
        # Confidence score (using sigmoid on linear projection)
        confidence = torch.sigmoid(self.output_proj(node_features)[:, 0])
        
        return features, confidence
    
    def get_image_features(self, x):
        """Extract image features without confidence"""
        features, _ = self.forward(x)
        return features