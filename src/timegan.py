import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os

class TimeGANGenerator(nn.Module):
    """
    TimeGAN Generator Network
    Generates synthetic time-series data from noise and embeddings
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(TimeGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # RNN layers for temporal dynamics
        self.rnn = nn.GRU(
            input_size=input_dim,
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
            nn.Tanh()  # Normalized output for generated data
        )
        
    def forward(self, noise, seq_len):
        """
        Forward pass of generator
        
        Args:
            noise: Random noise tensor [batch_size, noise_dim]
            seq_len: Length of sequence to generate
            
        Returns:
            Generated sequences [batch_size, seq_len, output_dim]
        """
        batch_size = noise.shape[0]
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(noise.device)
        
        # Generate sequences step by step
        generated_seq = []
        hidden = h0
        
        # Use noise as initial input, then use previous output
        current_input = noise.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        for t in range(seq_len):
            # RNN forward pass
            output, hidden = self.rnn(current_input, hidden)
            
            # Generate output for this timestep
            generated_output = self.output_layer(output)
            generated_seq.append(generated_output)
            
            # Use generated output as next input (teacher forcing disabled)
            current_input = generated_output
        
        # Concatenate all timesteps
        generated_sequence = torch.cat(generated_seq, dim=1)
        
        return generated_sequence

class TimeGANDiscriminator(nn.Module):
    """
    TimeGAN Discriminator Network
    Distinguishes between real and synthetic time-series data
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(TimeGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layers for temporal pattern recognition
        self.rnn = nn.GRU(
            input_size=input_dim,
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
        
    def forward(self, sequences):
        """
        Forward pass of discriminator
        
        Args:
            sequences: Input sequences [batch_size, seq_len, input_dim]
            
        Returns:
            Probability of being real [batch_size, 1]
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(sequences.device)
        
        # RNN forward pass
        output, hidden = self.rnn(sequences, h0)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Classification
        prob_real = self.classifier(last_hidden)
        
        return prob_real

class TimeGANEmbedder(nn.Module):
    """
    TimeGAN Embedder Network
    Maps real data to latent embedding space
    """
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=3):
        super(TimeGANEmbedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # RNN layers for embedding
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Embedding projection
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid()
        )
        
    def forward(self, sequences):
        """
        Forward pass of embedder
        
        Args:
            sequences: Real sequences [batch_size, seq_len, input_dim]
            
        Returns:
            Embedded sequences [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(sequences.device)
        
        # RNN forward pass
        output, _ = self.rnn(sequences, h0)
        
        # Apply embedding layer to each timestep
        embedded_seq = self.embedding_layer(output)
        
        return embedded_seq

class TimeGANRecovery(nn.Module):
    """
    TimeGAN Recovery Network
    Maps from embedding space back to data space
    """
    
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=3):
        super(TimeGANRecovery, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # RNN layers for recovery
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Recovery projection
        self.recovery_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, embedded_sequences):
        """
        Forward pass of recovery network
        
        Args:
            embedded_sequences: Embedded sequences [batch_size, seq_len, embedding_dim]
            
        Returns:
            Recovered sequences [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = embedded_sequences.shape
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(embedded_sequences.device)
        
        # RNN forward pass
        output, _ = self.rnn(embedded_sequences, h0)
        
        # Apply recovery layer to each timestep
        recovered_seq = self.recovery_layer(output)
        
        return recovered_seq

class TimeGAN:
    """
    Complete TimeGAN implementation for time-series generation
    """
    
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64, 
                 num_layers=3, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        
        # Initialize networks
        self.embedder = TimeGANEmbedder(
            input_dim, hidden_dim, embedding_dim, num_layers
        ).to(device)
        
        self.recovery = TimeGANRecovery(
            embedding_dim, hidden_dim, input_dim, num_layers
        ).to(device)
        
        self.generator = TimeGANGenerator(
            embedding_dim, hidden_dim, embedding_dim, num_layers
        ).to(device)
        
        self.discriminator = TimeGANDiscriminator(
            embedding_dim, hidden_dim, num_layers
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
        print("Training Autoencoder (Embedder + Recovery)...")
        
        self.embedder.train()
        self.recovery.train()
        
        for epoch in tqdm(range(epochs), desc="Autoencoder Training"):
            total_loss = 0
            
            for batch in real_data:
                batch = batch.to(self.device)
                
                # Forward pass
                embedded = self.embedder(batch)
                recovered = self.recovery(embedded)
                
                # Reconstruction loss
                loss = self.mse_loss(recovered, batch)
                
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
        Train the adversarial components (Generator vs Discriminator)
        """
        print("Training Adversarial Networks...")
        
        for epoch in tqdm(range(epochs), desc="Adversarial Training"):
            total_g_loss = 0
            total_d_loss = 0
            
            for batch in real_data:
                batch = batch.to(self.device)
                batch_size, seq_len, _ = batch.shape
                
                # Train Discriminator
                self.discriminator.train()
                self.discriminator_optimizer.zero_grad()
                
                # Real data embeddings
                with torch.no_grad():
                    real_embeddings = self.embedder(batch)
                
                # Generate fake embeddings
                noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
                with torch.no_grad():
                    fake_embeddings = self.generator(noise, seq_len)
                
                # Discriminator predictions
                real_pred = self.discriminator(real_embeddings)
                fake_pred = self.discriminator(fake_embeddings)
                
                # Discriminator loss
                real_labels = torch.ones_like(real_pred)
                fake_labels = torch.zeros_like(fake_pred)
                
                d_loss_real = self.bce_loss(real_pred, real_labels)
                d_loss_fake = self.bce_loss(fake_pred, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                # Train Generator
                self.generator.train()
                self.generator_optimizer.zero_grad()
                
                # Generate fake embeddings
                noise = torch.randn(batch_size, self.embedding_dim).to(self.device)
                fake_embeddings = self.generator(noise, seq_len)
                
                # Generator loss (fool discriminator)
                fake_pred = self.discriminator(fake_embeddings)
                g_loss_adv = self.bce_loss(fake_pred, real_labels)
                
                # Supervised loss (temporal consistency)
                with torch.no_grad():
                    real_embeddings = self.embedder(batch)
                g_loss_sup = self.mse_loss(fake_embeddings, real_embeddings)
                
                # Combined generator loss
                g_loss = g_loss_adv + 10 * g_loss_sup  # Weight supervised loss higher
                
                g_loss.backward()
                self.generator_optimizer.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
            
            if epoch % 50 == 0:
                avg_g_loss = total_g_loss / len(real_data)
                avg_d_loss = total_d_loss / len(real_data)
                print(f"Epoch {epoch}, G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}")
    
    def generate_synthetic_data(self, num_samples, seq_len):
        """
        Generate synthetic time-series data
        """
        self.generator.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(num_samples, self.embedding_dim).to(self.device)
            
            # Generate embeddings
            fake_embeddings = self.generator(noise, seq_len)
            
            # Recover to data space
            synthetic_data = self.recovery(fake_embeddings)
        
        return synthetic_data.cpu().numpy()
    
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
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers
            }
        }, path)
        
        print(f"TimeGAN model saved to {path}")
    
    def load_model(self, path):
        """Load all model components"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.embedder.load_state_dict(checkpoint['embedder_state_dict'])
        self.recovery.load_state_dict(checkpoint['recovery_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"TimeGAN model loaded from {path}")

def create_data_loader(X, y, batch_size=32, shuffle=True):
    """
    Create data loader for training
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and loader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader