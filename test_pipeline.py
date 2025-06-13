"""
Quick test of the conditional TimeGAN pipeline with minimal training
"""
import sys
sys.path.append('src')

import numpy as np
import torch
from conditional_timegan import ConditionalTimeGAN, create_conditional_data_loader

def quick_test():
    """Run a quick test with minimal training"""
    print("=== Quick Pipeline Test ===")
    
    # Load data with mapped labels
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train_mapped.npy")
    
    print(f"Training data: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    
    # Get data dimensions
    seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    print(f"Sequence length: {seq_len}, Features: {input_dim}, Classes: {num_classes}")
    
    # Create small dataset for testing
    n_samples = min(32, len(X_train))
    X_small = X_train[:n_samples]
    y_small = y_train[:n_samples]
    
    print(f"Using {n_samples} samples for quick test")
    
    # Create data loader
    train_loader = create_conditional_data_loader(X_small, y_small, batch_size=8, shuffle=True)
    
    # Initialize conditional TimeGAN with small dimensions
    device = 'cpu'  # Use CPU for quick test
    conditional_timegan = ConditionalTimeGAN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=32,
        embedding_dim=16,
        device=device
    )
    
    print("Training with minimal epochs...")
    
    # Quick autoencoder training
    conditional_timegan.train_autoencoder(train_loader, epochs=5)
    
    # Quick adversarial training
    conditional_timegan.train_adversarial(train_loader, epochs=5)
    
    # Test generation
    print("Testing generation...")
    for class_label in range(num_classes):
        synthetic_data = conditional_timegan.generate_synthetic_data(
            target_class=class_label, 
            num_samples=5, 
            seq_len=seq_len
        )
        print(f"Generated {synthetic_data.shape[0]} samples for class {class_label}")
        print(f"  Shape: {synthetic_data.shape}")
        print(f"  Range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    
    print("✅ Quick test completed successfully!")

if __name__ == "__main__":
    quick_test()