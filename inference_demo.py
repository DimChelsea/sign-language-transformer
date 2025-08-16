import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sign_language_transformer import SignLanguageTransformer
import cv2
from PIL import Image
from io import BytesIO
import base64
import time
from IPython.display import HTML, display

def load_model(model_path):
    """Load the trained model from a checkpoint file"""
    checkpoint = torch.load(model_path)
    
    # Get model configuration
    config = checkpoint.get('config', {
        'input_dim': 543,  # Default if not in checkpoint
        'hidden_dim': 256,
        'num_classes': len(checkpoint.get('class_mapping', {})),
        'num_layers': 4,
        'dropout': 0.1,
        'max_seq_length': 16
    })
    
    # Create model
    model = SignLanguageTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config.get('max_seq_length', 16),
        use_finger_spelling=True
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get class mapping
    class_mapping = checkpoint.get('class_mapping', {})
    
    return model, class_mapping

def get_demo_samples(metadata_path, num_samples=5):
    """Get random samples from metadata for demo"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Select random samples
    np.random.seed(42)  # For reproducibility
    selected_samples = np.random.choice(metadata, size=min(num_samples, len(metadata)), replace=False)
    
    return selected_samples

def load_and_preprocess_pose_data(pose_path, max_seq_length=16):
    """Load and preprocess pose data for the model"""
    # Load pose data
    pose_data = np.load(pose_path)
    
    # Pad or truncate sequence to max_seq_length
    seq_len = pose_data.shape[0]
    if seq_len > max_seq_length:
        # Truncate
        pose_data = pose_data[:max_seq_length]
    elif seq_len < max_seq_length:
        # Pad
        padding = np.zeros((max_seq_length - seq_len, pose_data.shape[1]))
        pose_data = np.vstack([pose_data, padding])
    
    return torch.FloatTensor(pose_data).unsqueeze(0)  # Add batch dimension

def visualize_prediction(sample, prediction, confidence, class_mapping, frame_paths):
    """Visualize the prediction with sample frames"""
    # Get ground truth
    ground_truth = sample['gloss']
    
    # Get predicted class name
    predicted_class = class_mapping.get(str(prediction), "Unknown")
    
    # Create a figure with frames and prediction
    num_frames = min(5, len(frame_paths))  # Show at most 5 frames
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
    
    # Selected frames (evenly spaced)
    indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
    
    # Plot frames
    for i, idx in enumerate(indices):
        frame_path = frame_paths[idx]
        img = plt.imread(frame_path)
        if num_frames > 1:
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Frame {idx}")
        else:
            axes.imshow(img)
            axes.axis('off')
            axes.set_title(f"Frame {idx}")
    
    # Add overall title with prediction information
    correct = predicted_class.lower() == ground_truth.lower()
    color = 'green' if correct else 'red'
    fig.suptitle(f"Ground Truth: {ground_truth} | Prediction: {predicted_class} ({confidence:.2f}%)", 
                 fontsize=16, color=color)
    
    plt.tight_layout()
    plt.show()
    
    return correct

def create_animation(frame_paths):
    """Create an animation from frames"""
    frames = []
    for path in frame_paths:
        img = Image.open(path)
        # Resize for better display
        img = img.resize((200, 200), Image.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        frames.append(f'data:image/jpeg;base64,{img_str}')
    
    # Create HTML with animation
    html = """
    <div style="display: flex; align-items: center; justify-content: center;">
      <img id="animation" src="{}" style="max-height: 200px;">
    </div>
    <script>
      const frames = {};
      let currentFrame = 0;
      const animationElement = document.getElementById('animation');
      
      function updateAnimation() {
        animationElement.src = frames[currentFrame];
        currentFrame = (currentFrame + 1) % frames.length;
      }
      
      // Start animation
      setInterval(updateAnimation, 150);
    </script>
    """.format(frames[0], frames)
    
    return HTML(html)

def main():
    # Paths
    model_path = '/content/drive/MyDrive/sign_language_project/models/final_model.pth'
    metadata_path = '/content/wlasl_processed_metadata.json'
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_mapping = load_model(model_path)
    model = model.to(device)
    model.eval()
    
    # Create reverse mapping (index to class name)
    idx_to_class = {int(idx): class_name for idx, class_name in class_mapping.items()}
    
    # Get demo samples
    print("Getting demo samples...")
    demo_samples = get_demo_samples(metadata_path)
    
    # Run inference on demo samples
    print("Running inference...")
    correct_count = 0
    total_count = 0
    
    for sample in demo_samples:
        print(f"\nProcessing sign: {sample['gloss']} (Video ID: {sample['video_id']})")
        
        # Load and preprocess pose data
        pose_tensor = load_and_preprocess_pose_data(sample['pose_path'])
        pose_tensor = pose_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            outputs = model(pose_tensor)
            inference_time = time.time() - start_time
            
            # Get prediction
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item() * 100
        
        print(f"Inference completed in {inference_time*1000:.2f}ms")
        print(f"Predicted sign: {idx_to_class[prediction]} with {confidence:.2f}% confidence")
        print(f"Ground truth: {sample['gloss']}")
        
        # Visualize prediction
        correct = visualize_prediction(sample, prediction, confidence, idx_to_class, sample['frame_paths'])
        
        # Show animation of the sign
        print("Sign animation:")
        display(create_animation(sample['frame_paths']))
        
        # Update statistics
        correct_count += int(correct)
        total_count += 1
        
        print("-" * 50)
    
    # Show overall accuracy
    accuracy = (correct_count / total_count) * 100
    print(f"\nDemo accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()