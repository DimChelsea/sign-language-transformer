import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm

# Define dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, pose_dir):
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.pose_dir = pose_dir
        self.samples = []
        
        # Process metadata to create a list of samples
        for item in self.metadata:
            video_id = item['video_id']
            pose_file = os.path.join(pose_dir, f"{video_id}.npy")
            
            if os.path.exists(pose_file):
                self.samples.append({
                    'pose_file': pose_file,
                    'label': item['label_index']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pose data
        pose_data = np.load(sample['pose_file'])
        
        # Convert to tensor
        pose_tensor = torch.tensor(pose_data, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return pose_tensor, label

# Collate function to handle variable length sequences
def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get sequences and labels
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Get sequence lengths
    lengths = [seq.shape[0] for seq in sequences]
    max_length = max(lengths)
    
    # Pad sequences
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[1]
    padded_sequences = torch.zeros(batch_size, max_length, feature_dim)
    
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded_sequences[i, :length, :] = seq
    
    # Convert labels to tensor
    labels = torch.stack(labels)
    
    return padded_sequences, labels

def train_model(model, epochs, batch_size, learning_rate, metadata_path, pose_dir, save_dir, device):
    # Create datasets
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Split metadata into train and validation
    np.random.seed(42)
    np.random.shuffle(metadata)
    split_idx = int(len(metadata) * 0.8)  # 80% for training, 20% for validation
    train_metadata = metadata[:split_idx]
    val_metadata = metadata[split_idx:]
    
    # Save split metadata
    with open(os.path.join(save_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f)
    
    with open(os.path.join(save_dir, 'val_metadata.json'), 'w') as f:
        json.dump(val_metadata, f)
    
    # Create temporary metadata files for train and val
    train_metadata_path = os.path.join(save_dir, 'train_metadata.json')
    val_metadata_path = os.path.join(save_dir, 'val_metadata.json')
    
    # Create datasets
    train_dataset = SignLanguageDataset(train_metadata_path, pose_dir)
    val_dataset = SignLanguageDataset(val_metadata_path, pose_dir)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track stats
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print stats
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_correct/train_total:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_correct/val_total:.4f}")
        print("-" * 50)
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"Saved best model at epoch {epoch+1}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Plot training and validation loss
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()
    except:
        print("Couldn't create plot. Matplotlib might not be available.")
    
    print("Training completed!")
    return train_losses, val_losses
