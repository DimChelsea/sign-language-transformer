import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and moved with the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input
        # x shape: [batch_size, seq_length, embed_dim]
        return x + self.pe[:, :x.size(1), :]

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers, num_classes, dropout=0.1):
        super().__init__()
        
        # Project input features to embedding dimension
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects: [seq_length, batch_size, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Average pooling over sequence dimension
        x = x.permute(1, 0, 2).mean(dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        output = self.classifier(x)
        
        return output
