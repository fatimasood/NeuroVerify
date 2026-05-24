"""
Transformer encoder-decoder for image captioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerCaptionGenerator(nn.Module):
    """Transformer for image caption generation"""
    
    def __init__(self, feature_dim=512, d_model=512, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=2048, vocab_size=10000, max_len=100):
        super(TransformerCaptionGenerator, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Image feature projection
        self.image_proj = nn.Linear(feature_dim, d_model)
        
        # Transformer components
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        
    def generate_caption(self, image_features, tokenizer=None):
        """Generate caption from image features"""
        batch_size = image_features.shape[0]
        
        # Project image features
        image_embed = self.image_proj(image_features).unsqueeze(1)  # (batch, 1, d_model)
        
        # Encode image
        encoder_out = self.transformer_encoder(image_embed)
        
        # Generate tokens autoregressively
        generated_tokens = torch.zeros(batch_size, self.max_len, dtype=torch.long, 
                                      device=image_features.device)
        
        # Start token (assuming 2 is start token)
        current_tokens = torch.full((batch_size, 1), 2, device=image_features.device)
        
        for i in range(self.max_len):
            # Embed current tokens
            token_embed = self.word_embedding(current_tokens)
            token_embed = self.positional_encoding(token_embed)
            
            # Decode
            decoder_out = self.transformer_decoder(token_embed, encoder_out)
            
            # Get next token logits
            logits = self.fc_out(decoder_out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_tokens[:, i] = next_token.squeeze()
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all generated end token (assuming 3 is end token)
            if (next_token == 3).all():
                break
        
        return generated_tokens