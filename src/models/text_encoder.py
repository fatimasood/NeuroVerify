"""
LSTM-based text encoder as described in paper Section 3.4
Uses BiLSTM architecture with specific parameters from Table 7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTextEncoder(nn.Module):
    """
    LSTM text encoder as per paper Section 3.4
    Parameters from Table 7:
    - Optimizer: Adam
    - Batch size: 20
    - Epochs: 20
    - Activation: Sigmoid
    - Learning rate: 0.005
    - Gradient threshold: 1
    - Learning rate drop factor: 0.2
    """
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMTextEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer (Word2Vec/GloVe style as per paper)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM as per paper Section 3.4
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification layers (as per paper)
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        
        self.layer_norm = nn.LayerNorm(64)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights as per paper"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass as per paper Equation (1) and Section 3.4
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, 2) classification logits
            features: (batch_size, lstm_output_dim) text features
            confidence: (batch_size,) confidence score (probability of real)
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Pack sequence for efficiency (respect attention mask)
        seq_lengths = attention_mask.sum(dim=1).cpu().int()
        if (seq_lengths > 0).all():
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get final hidden states (as per paper)
        if self.bidirectional:
            # Concatenate forward and backward final states
            # Paper Section 3.4: "hidden state of the final token"
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Classification layers
        features = F.relu(self.fc1(hidden))
        features = self.layer_norm(features)
        features = self.dropout(features)
        logits = self.fc2(features)
        
        # Confidence score (probability of being real)
        probs = F.softmax(logits, dim=1)
        confidence = probs[:, 1]  # Real class probability
        
        return logits, features, confidence
    
    def get_text_features(self, input_ids, attention_mask):
        """Extract text features without classification (for similarity matching)"""
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        seq_lengths = attention_mask.sum(dim=1).cpu().int()
        if (seq_lengths > 0).all():
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, cell) = self.lstm(packed)
        else:
            _, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden