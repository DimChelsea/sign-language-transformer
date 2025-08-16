import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FingerSpellingAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FingerSpellingAttention, self).__init__()
        self.attention = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = self.context(torch.tanh(self.attention(x)))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, input_dim)
        return context, attention_weights

class FingerSpellingModel(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=128, num_classes=26):
        super(FingerSpellingModel, self).__init__()
        
        # Hand keypoint features processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = FingerSpellingAttention(hidden_dim * 2, hidden_dim)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features=126)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)  # (batch_size, hidden_dim*2)
        
        # Classification
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights

# Utility function for processing hand keypoints specifically
def extract_hand_features(pose_data):
    """
    Extract hand features from MediaPipe pose data
    Input: pose_data - np array of shape (frames, 543) [full body pose data]
    Output: hand_features - np array of shape (frames, 126) [only hand keypoints]
    """
    # Assuming pose_data includes 21 points per hand (21 points * 2 hands * 3 coordinates = 126 features)
    # MediaPipe hand points start after the body pose points
    hand_start_idx = 33 * 3  # 33 body points * 3 coordinates
    hand_features = pose_data[:, hand_start_idx:hand_start_idx + 126]
    
    # Normalize hand features
    hand_features = (hand_features - np.mean(hand_features, axis=0)) / (np.std(hand_features, axis=0) + 1e-8)
    
    return hand_features

if __name__ == "__main__":
    # Test the model with random data
    batch_size = 8
    seq_length = 16
    input_dim = 126
    
    # Create random input
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Initialize model
    model = FingerSpellingModel()
    
    # Forward pass
    output, attention = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print("FingerSpellingModel test successful!")