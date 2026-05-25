"""
Trustify: Main model architecture
Combines text LSTM + image ResNet50+GNN+Transformer + fuzzy logic layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """LSTM-based text encoder."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize text encoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_text = nn.Linear(lstm_output_dim, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_indices: Text token indices [batch, seq_len]
            
        Returns:
            logits: [batch, 2] (real/fake scores)
            features: [batch, hidden_dim*2 or hidden_dim] (text features)
        """
        # Embedding
        embedded = self.dropout(self.embedding(text_indices))
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = hidden[-1] if isinstance(hidden, torch.Tensor) else hidden[-1]
        
        # Classification
        logits = self.fc_text(last_hidden)
        
        return logits, last_hidden


class ImageEncoder(nn.Module):
    """ResNet50 + GNN + Transformer-based image encoder."""
    
    def __init__(
        self,
        pretrained: bool = True,
        gnn_hidden_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 3
    ):
        """
        Initialize image encoder.
        
        Args:
            pretrained: Use pretrained ResNet50
            gnn_hidden_dim: GNN hidden dimension
            transformer_heads: Transformer attention heads
            transformer_layers: Transformer layers
        """
        super().__init__()
        
        # ResNet50 backbone
        import torchvision.models as models
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove classification layer
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_output_dim = 2048
        
        # GNN layer (simplified - Graph Convolution)
        self.gnn_fc1 = nn.Linear(self.resnet_output_dim, gnn_hidden_dim)
        self.gnn_fc2 = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=gnn_hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Classification head
        self.fc_image = nn.Linear(gnn_hidden_dim, 2)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Image tensor [batch, 3, 224, 224]
            
        Returns:
            logits: [batch, 2] (real/fake scores)
            features: [batch, gnn_hidden_dim] (image features)
        """
        # ResNet50 features
        features = self.resnet_features(images)
        features = features.view(features.size(0), -1)
        
        # GNN layers
        gnn_out = F.relu(self.gnn_fc1(features))
        gnn_out = self.dropout(gnn_out)
        gnn_out = F.relu(self.gnn_fc2(gnn_out))
        
        # Transformer (add sequence dimension)
        transformer_in = gnn_out.unsqueeze(1)
        transformer_out = self.transformer(transformer_in)
        transformer_out = transformer_out.squeeze(1)
        
        # Classification
        logits = self.fc_image(transformer_out)
        
        return logits, transformer_out


class FuzzyDecisionLayer(nn.Module):
    """Fuzzy logic decision layer for combining text and image predictions."""
    
    def __init__(self, num_rules: int = 4):
        """
        Initialize fuzzy layer.
        
        Args:
            num_rules: Number of fuzzy rules
        """
        super().__init__()
        self.num_rules = num_rules
        
        # Learnable rule weights
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        
        # Membership function parameters
        self.mu_low = nn.Parameter(torch.tensor(0.3))
        self.mu_high = nn.Parameter(torch.tensor(0.7))
    
    def fuzzify(self, confidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuzzify confidence scores to membership degrees.
        
        Args:
            confidence: Confidence scores [batch, 2]
            
        Returns:
            low_membership: Membership degree for "Low" [batch]
            high_membership: Membership degree for "High" [batch]
        """
        # Get real class confidence
        real_conf = torch.softmax(confidence, dim=1)[:, 0]
        
        # Gaussian membership functions
        low_membership = torch.exp(-((real_conf - self.mu_low) ** 2) / 0.04)
        high_membership = torch.exp(-((real_conf - self.mu_high) ** 2) / 0.04)
        
        return low_membership, high_membership
    
    def forward(
        self,
        text_logits: torch.Tensor,
        image_logits: torch.Tensor,
        similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuzzy decision making.
        
        Args:
            text_logits: Text model logits [batch, 2]
            image_logits: Image model logits [batch, 2]
            similarity: Text-image similarity [batch]
            
        Returns:
            fused_logits: Final fused logits [batch, 2]
        """
        # Softmax confidence scores
        text_conf = torch.softmax(text_logits, dim=1)
        image_conf = torch.softmax(image_logits, dim=1)
        
        # Fuzzify
        text_low, text_high = self.fuzzify(text_logits)
        image_low, image_high = self.fuzzify(image_logits)
        
        # Fuzzy rules: combine modalities
        # Rule 1: IF text IS real AND image IS real THEN real
        rule1 = text_high * image_high
        
        # Rule 2: IF text IS fake AND image IS real THEN fake
        rule2 = text_low * image_high
        
        # Rule 3: IF text IS real AND image IS fake THEN fake
        rule3 = text_high * image_low
        
        # Rule 4: IF text IS fake AND image IS fake THEN fake
        rule4 = text_low * image_low
        
        # Aggregate rules
        rules = torch.stack([rule1, rule2, rule3, rule4], dim=1)
        
        # Weight rules
        weights = torch.softmax(self.rule_weights.unsqueeze(0), dim=1)
        aggregated = (rules * weights).sum(dim=1)
        
        # Defuzzify: convert back to confidence
        fused_real_conf = aggregated
        fused_fake_conf = 1.0 - aggregated
        
        # Apply similarity weight
        fused_real_conf = fused_real_conf * (0.5 + 0.5 * similarity)
        fused_fake_conf = fused_fake_conf * (0.5 - 0.5 * similarity)
        
        # Normalize
        fused_logits = torch.stack(
            [fused_real_conf, fused_fake_conf],
            dim=1
        )
        
        return fused_logits


class TrustifyModel(nn.Module):
    """Complete Trustify model."""
    
    def __init__(self, config: Dict):
        """
        Initialize Trustify model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=config.get('vocab_size', 10000),
            embedding_dim=config.get('text_embedding_dim', 100),
            hidden_dim=config.get('lstm_hidden_dim', 128),
            num_layers=config.get('lstm_num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            pretrained=True,
            gnn_hidden_dim=config.get('gnn_hidden_dim', 256),
            transformer_heads=config.get('transformer_heads', 8),
            transformer_layers=config.get('transformer_layers', 3)
        )
        
        # Text-Image similarity module
        text_feature_dim = config.get('lstm_hidden_dim', 128) * 2
        image_feature_dim = config.get('gnn_hidden_dim', 256)
        
        self.similarity_fc1 = nn.Linear(text_feature_dim + image_feature_dim, 128)
        self.similarity_fc2 = nn.Linear(128, 1)
        
        # Fuzzy decision layer
        self.fuzzy_layer = FuzzyDecisionLayer(num_rules=4)
        
        # Final classifier
        self.final_classifier = nn.Linear(2, 2)
        
        logger.info("✓ Trustify model initialized")
    
    def forward(
        self,
        text_indices: torch.Tensor,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_indices: Text token indices [batch, seq_len]
            images: Images [batch, 3, 224, 224]
            
        Returns:
            Dictionary with:
            - logits: Final classification logits
            - probabilities: Final class probabilities
            - text_logits: Text encoder output
            - image_logits: Image encoder output
            - similarity: Text-image similarity
        """
        # Encode text
        text_logits, text_features = self.text_encoder(text_indices)
        
        # Encode image
        image_logits, image_features = self.image_encoder(images)
        
        # Compute text-image similarity
        combined_features = torch.cat([text_features, image_features], dim=1)
        similarity_hidden = F.relu(self.similarity_fc1(combined_features))
        similarity = torch.sigmoid(self.similarity_fc2(similarity_hidden)).squeeze(1)
        
        # Fuzzy decision layer
        fuzzy_logits = self.fuzzy_layer(text_logits, image_logits, similarity)
        
        # Final classification
        final_logits = self.final_classifier(fuzzy_logits)
        
        # Probabilities
        probabilities = torch.softmax(final_logits, dim=1)
        
        return {
            'logits': final_logits,
            'probabilities': probabilities,
            'text_logits': text_logits,
            'image_logits': image_logits,
            'similarity': similarity,
            'fuzzy_logits': fuzzy_logits
        }


if __name__ == "__main__":
    # Test model
    config = {
        'vocab_size': 10000,
        'text_embedding_dim': 100,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'gnn_hidden_dim': 256,
        'transformer_heads': 8,
        'transformer_layers': 3,
        'dropout': 0.3
    }
    
    model = TrustifyModel(config)
    
    # Test forward pass
    batch_size = 4
    text_input = torch.randint(0, 10000, (batch_size, 512))
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(text_input, image_input)
    
    print("Model output shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Probabilities: {output['probabilities'].shape}")
    print(f"  Similarity: {output['similarity'].shape}")
    print(f"\n✓ Model test passed!")