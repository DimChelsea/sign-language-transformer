import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from sign_language_transformer import SignLanguageTransformer
from finger_spelling import FingerSpellingModel
from tqdm import tqdm

# Configuration
BATCH_SIZE = 10  # Slightly increased batch size for faster training with A100
MAX_SEQ_LENGTH = 16
INPUT_DIM = 543  # MediaPipe pose dimensions (33 body + 21*2 hand landmarks) * 3 (x,y,z)
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
USE_MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 5

# Directory for saving models
MODEL_DIR = '/content/drive/MyDrive/sign_language_project/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SignLanguageDataset(Dataset):
    def __init__(self, metadata, max_seq_length=16):
        self.metadata = metadata
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load pose data
        pose_path = item['pose_path']
        pose_data = np.load(pose_path)
        
        # Pad or truncate sequence to max_seq_length
        seq_len = pose_data.shape[0]
        if seq_len > self.max_seq_length:
            # Truncate
            pose_data = pose_data[:self.max_seq_length]
        elif seq_len < self.max_seq_length:
            # Pad
            padding = np.zeros((self.max_seq_length - seq_len, pose_data.shape[1]))
            pose_data = np.vstack([pose_data, padding])
        
        # Convert to tensors
        pose_tensor = torch.FloatTensor(pose_data)
        
        # Get label
        label = item['label_idx']
        label_tensor = torch.LongTensor([label])
        
        return {
            'pose_data': pose_tensor,
            'label': label_tensor,
            'video_id': item['video_id'],
            'gloss': item['gloss']
        }

def collate_fn(batch):
    pose_data = torch.stack([item['pose_data'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    video_ids = [item['video_id'] for item in batch]
    glosses = [item['gloss'] for item in batch]
    
    return {
        'pose_data': pose_data,
        'labels': labels,
        'video_ids': video_ids,
        'glosses': glosses
    }

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        pose_data = batch['pose_data'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(pose_data)
                loss = criterion(outputs, labels)
            
            # Backward and optimize with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            outputs = model(pose_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update statistics
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'accuracy': 100 * correct / total
        })
    
    return total_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Move data to device
            pose_data = batch['pose_data'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pose_data)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update statistics
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'accuracy': 100 * correct / total
            })
    
    return total_loss / len(dataloader), 100 * correct / total

def main():
    print("Starting sign language transformer training...")
    start_time = time.time()
    
    # Load metadata
    print("Loading metadata...")
    with open('/content/wlasl_processed_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Prepare metadata by adding label indices
    print("Preparing metadata...")
    # Get unique glosses (sign classes)
    glosses = sorted(list(set([item['gloss'] for item in metadata])))
    num_classes = len(glosses)
    print(f"Found {num_classes} unique sign classes")
    
    # Create a mapping from gloss to label index
    gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
    
    # Add label indices to metadata
    for item in metadata:
        item['label_idx'] = gloss_to_idx[item['gloss']]
    
    # Train/validation split (80/20)
    print("Splitting data into train and validation sets...")
    np.random.shuffle(metadata)
    split_idx = int(0.8 * len(metadata))
    train_metadata = metadata[:split_idx]
    val_metadata = metadata[split_idx:]
    
    print(f"Training samples: {len(train_metadata)}")
    print(f"Validation samples: {len(val_metadata)}")
    
    # Create datasets and dataloaders
    print("Creating data loaders...")
    train_dataset = SignLanguageDataset(train_metadata, max_seq_length=MAX_SEQ_LENGTH)
    val_dataset = SignLanguageDataset(val_metadata, max_seq_length=MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = SignLanguageTransformer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_seq_length=MAX_SEQ_LENGTH,
        use_finger_spelling=True
    )
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    # Check for checkpoint to resume training
    start_epoch = 0
    checkpoint_path = os.path.join(MODEL_DIR, 'latest_checkpoint.pth')
    best_val_accuracy = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0)
        print(f"Resuming training from epoch {start_epoch}")
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs = []
    
    # Early stopping setup
    no_improvement = 0
    
    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Track metrics
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save checkpoint for every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_val_accuracy': best_val_accuracy
        }, checkpoint_path)
        
        # Save numbered checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            numbered_path = os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'best_val_accuracy': best_val_accuracy
            }, numbered_path)
        
        # Check for best model and save
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'best_val_accuracy': best_val_accuracy,
                'config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_classes': num_classes,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'max_seq_length': MAX_SEQ_LENGTH
                }
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Check for early stopping
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Plot and save metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot loss
            ax1.plot(epochs, train_losses, 'b-', label='Training loss')
            ax1.plot(epochs, val_losses, 'r-', label='Validation loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(epochs, train_accuracies, 'b-', label='Training accuracy')
            ax2.plot(epochs, val_accuracies, 'r-', label='Validation accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f'metrics_epoch_{epoch+1}.png'))
            plt.close()
        
        # Print time elapsed
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Estimate remaining time
        if epoch > start_epoch:
            avg_time_per_epoch = elapsed / (epoch - start_epoch + 1)
            remaining_epochs = NUM_EPOCHS - epoch - 1
            remaining_time = avg_time_per_epoch * remaining_epochs
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Estimated time remaining: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_classes': num_classes,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'max_seq_length': MAX_SEQ_LENGTH
        },
        'class_mapping': {idx: gloss for gloss, idx in gloss_to_idx.items()}
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Create and save final plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation loss')
    ax1.set_title('Training and Validation Loss Over Full Training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Training accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy Over Full Training')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Save final plot
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'final_training_metrics.png'))
    
    # Also save the raw metrics for further analysis
    metrics = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(os.path.join(MODEL_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Training completed!")
    
    # Print total training time
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    main()