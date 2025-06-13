import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os

class ConditionalTimeGANGenerator(nn.Module):
    """
    Conditional TimeGAN Generator Network
    Generates synthetic time-series data conditioned on class labels
    """
    
    def __init__(self, input_dim, condition_dim, hidden_dim, output_dim, num_layers=3):
        super(ConditionalTimeGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Condition embedding layer
        self.condition_embedding = nn.Embedding(condition_dim, hidden_dim // 4)
        
        # RNN layers for temporal dynamics
        self.rnn = nn.GRU(
            input_size=input_dim + hidden_dim // 4,  # concatenate noise with condition
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, noise, condition, seq_len):
        """
        Forward pass of conditional generator
        
        Args:
            noise: Random noise tensor [batch_size, noise_dim]
            condition: Class condition [batch_size] (integer labels)
            seq_len: Length of sequence to generate
            
        Returns:
            Generated sequences [batch_size, seq_len, output_dim]
        """
        batch_size = noise.shape[0]
        
        # Embed condition
        condition_emb = self.condition_embedding(condition)  # [batch_size, emb_dim]
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(noise.device)
        
        # Generate sequences step by step
        generated_seq = []
        hidden = h0
        
        # Concatenate noise with condition embedding for initial input
        noise_with_condition = torch.cat([noise, condition_emb], dim=1)
        current_input = noise_with_condition.unsqueeze(1)  # [batch_size, 1, input_dim + cond_dim]
        
        for t in range(seq_len):
            # RNN forward pass
            output, hidden = self.rnn(current_input, hidden)
            
            # Generate output for this timestep
            generated_output = self.output_layer(output)
            generated_seq.append(generated_output)
            
            # For next timestep, concatenate generated output with condition
            next_input = torch.cat([generated_output.squeeze(1), condition_emb], dim=1)
            current_input = next_input.unsqueeze(1)
        
        # Concatenate all timesteps
        generated_sequence = torch.cat(generated_seq, dim=1)
        
        return generated_sequence

class ConditionalTimeGANDiscriminator(nn.Module):
    """
    Conditional TimeGAN Discriminator Network
    Distinguishes between real and synthetic data, considering class labels
    """
    
    def __init__(self, input_dim, condition_dim, hidden_dim, num_layers=3):
        super(ConditionalTimeGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, hidden_dim // 4)
        
        # RNN layers for temporal pattern recognition
        self.rnn = nn.GRU(
            input_size=input_dim + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequences, condition):
        """
        Forward pass of conditional discriminator
        
        Args:
            sequences: Input sequences [batch_size, seq_len, input_dim]
            condition: Class condition [batch_size] (integer labels)
            
        Returns:
            Probability of being real [batch_size, 1]
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Embed condition
        condition_emb = self.condition_embedding(condition)  # [batch_size, emb_dim]
        
        # Repeat condition for each timestep
        condition_seq = condition_emb.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, emb_dim]
        
        # Concatenate sequences with condition
        sequences_with_condition = torch.cat([sequences, condition_seq], dim=2)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(sequences.device)
        
        # RNN forward pass
        output, hidden = self.rnn(sequences_with_condition, h0)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Classification
        prob_real = self.classifier(last_hidden)
        
        return prob_real

class ConditionalTimeGAN:
    """
    Complete Conditional TimeGAN implementation for class-conditional generation
    """
    
    def __init__(self, input_dim, num_classes, hidden_dim=128, embedding_dim=64, 
                 num_layers=3, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        
        # Initialize networks (reuse embedder and recovery from base TimeGAN)
        from timegan import TimeGANEmbedder, TimeGANRecovery
        
        self.embedder = TimeGANEmbedder(
            input_dim, hidden_dim, embedding_dim, num_layers
        ).to(device)
        
        self.recovery = TimeGANRecovery(
            embedding_dim, hidden_dim, input_dim, num_layers
        ).to(device)
        
        # Conditional generator and discriminator
        self.generator = ConditionalTimeGANGenerator(
            embedding_dim, num_classes, hidden_dim, embedding_dim, num_layers
        ).to(device)
        
        self.discriminator = ConditionalTimeGANDiscriminator(
            embedding_dim, num_classes, hidden_dim, num_layers
        ).to(device)
        
        # Initialize optimizers
        self.embedder_optimizer = optim.Adam(self.embedder.parameters(), lr=1e-3)
        self.recovery_optimizer = optim.Adam(self.recovery.parameters(), lr=1e-3)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def train_autoencoder(self, real_data, epochs=100):
        """
        Pre-train the embedder and recovery networks (autoencoder phase)
        """
        print("Training Conditional Autoencoder (Embedder + Recovery)...")
        
        self.embedder.train()
        self.recovery.train()
        
        for epoch in tqdm(range(epochs), desc="Autoencoder Training"):
            total_loss = 0
            
            for batch_data, _ in real_data:  # Ignore labels for autoencoder training
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                embedded = self.embedder(batch_data)
                recovered = self.recovery(embedded)
                
                # Reconstruction loss
                loss = self.mse_loss(recovered, batch_data)
                
                # Backward pass
                self.embedder_optimizer.zero_grad()
                self.recovery_optimizer.zero_grad()
                loss.backward()
                self.embedder_optimizer.step()
                self.recovery_optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(real_data)
                print(f"Epoch {epoch}, Autoencoder Loss: {avg_loss:.6f}")
    
    def train_adversarial(self, real_data, epochs=200):
        """
        Train the conditional adversarial components
        """
        print("Training Conditional Adversarial Networks...")
        
        for epoch in tqdm(range(epochs), desc="Conditional Adversarial Training"):
            total_g_loss = 0
            total_d_loss = 0
            
            for batch_data, batch_labels in real_data:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_size, seq_len, _ = batch_data.shape
                
                # Train Discriminator
                self.discriminator.train()
                self.discriminator_optimizer.zero_grad()
                
                # Real data embeddings
                with torch.no_grad():
                    real_embeddings = self.embedder(batch_data)
                
                # Generate fake embeddings with real labels
                noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
                with torch.no_grad():
                    fake_embeddings = self.generator(noise, batch_labels, seq_len)
                
                # Discriminator predictions
                real_pred = self.discriminator(real_embeddings, batch_labels)
                fake_pred = self.discriminator(fake_embeddings, batch_labels)
                
                # Discriminator loss
                real_labels_disc = torch.ones_like(real_pred)
                fake_labels_disc = torch.zeros_like(fake_pred)
                
                d_loss_real = self.bce_loss(real_pred, real_labels_disc)
                d_loss_fake = self.bce_loss(fake_pred, fake_labels_disc)
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                # Train Generator
                self.generator.train()
                self.generator_optimizer.zero_grad()
                
                # Generate fake embeddings
                noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
                fake_embeddings = self.generator(noise, batch_labels, seq_len)
                
                # Generator loss (fool discriminator)
                fake_pred = self.discriminator(fake_embeddings, batch_labels)
                g_loss_adv = self.bce_loss(fake_pred, real_labels_disc)
                
                # Supervised loss (temporal consistency)
                with torch.no_grad():
                    real_embeddings = self.embedder(batch_data)
                g_loss_sup = self.mse_loss(fake_embeddings, real_embeddings)
                
                # Combined generator loss
                g_loss = g_loss_adv + 10 * g_loss_sup
                
                g_loss.backward()
                self.generator_optimizer.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
            
            if epoch % 50 == 0:
                avg_g_loss = total_g_loss / len(real_data)
                avg_d_loss = total_d_loss / len(real_data)
                print(f"Epoch {epoch}, G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}")
    
    def generate_synthetic_data(self, target_class, num_samples, seq_len):
        """
        Generate synthetic time-series data for a specific class
        
        Args:
            target_class: Target class label (integer)
            num_samples: Number of samples to generate
            seq_len: Length of sequences to generate
        """
        self.generator.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            # Create condition tensor
            condition = torch.full((num_samples,), target_class, dtype=torch.long).to(self.device)
            
            # Generate noise
            noise = torch.randn(num_samples, self.embedding_dim).to(self.device)
            
            # Generate embeddings
            fake_embeddings = self.generator(noise, condition, seq_len)
            
            # Recover to data space
            synthetic_data = self.recovery(fake_embeddings)
        
        return synthetic_data.cpu().numpy()
    
    def generate_balanced_dataset(self, class_counts, seq_len):
        """
        Generate a balanced dataset by oversampling minority classes
        
        Args:
            class_counts: Dictionary {class_label: desired_count}
            seq_len: Length of sequences
            
        Returns:
            Dictionary with synthetic data for each class
        """
        synthetic_data = {}
        
        for class_label, count in class_counts.items():
            print(f"Generating {count} samples for class {class_label}...")
            synthetic_data[class_label] = self.generate_synthetic_data(
                class_label, count, seq_len
            )
        
        return synthetic_data
    
    def save_model(self, path):
        """Save all model components"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'embedder_state_dict': self.embedder.state_dict(),
            'recovery_state_dict': self.recovery.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers
            }
        }, path)
        
        print(f"Conditional TimeGAN model saved to {path}")
    
    def load_model(self, path):
        """Load all model components"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.embedder.load_state_dict(checkpoint['embedder_state_dict'])
        self.recovery.load_state_dict(checkpoint['recovery_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"Conditional TimeGAN model loaded from {path}")

def create_conditional_data_loader(X, y, batch_size=32, shuffle=True):
    """
    Create data loader for conditional training
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and loader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader