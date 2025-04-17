import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'model.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DroneDetector(nn.Module):
    """PyTorch model for drone detection."""
    
    def __init__(self, input_shape):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input features (n_features,)
        """
        super(DroneDetector, self).__init__()
        
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)
        
        # Convolutional layers with adjusted architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            # Fourth conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # Additional conv block to match desired output size
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Calculate size after convolutions
        self._calculate_conv_output_size()
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def _calculate_conv_output_size(self):
        """Calculate the size of the flattened features after convolutions."""
        # Create a dummy input with the correct shape
        x = torch.randn(1, 2, self.input_shape[0])
        logger.info(f"Input shape for conv size calculation: {x.shape}")
        
        # Pass through conv layers
        x = self.conv_layers(x)
        logger.info(f"Output shape after conv layers: {x.shape}")
        
        # Calculate flattened size
        self.conv_output_size = x.view(1, -1).size(1)
        logger.info(f"Flattened conv output size: {self.conv_output_size}")
        
    def forward(self, x):
        """Forward pass through the network."""
        # Handle complex input
        if torch.is_complex(x):
            x_real = x.real
            x_imag = x.imag
            x = torch.stack([x_real, x_imag], dim=1)
        else:
            x = x.view(x.size(0), 2, -1)
        
        logger.debug(f"Input shape after complex handling: {x.shape}")
        
        # Pass through conv layers
        x = self.conv_layers(x)
        logger.debug(f"Shape after conv layers: {x.shape}")
        
        # Flatten
        x = x.view(x.size(0), -1)
        logger.debug(f"Shape after flattening: {x.shape}")
        
        # Pass through dense layers
        x = self.dense_layers(x)
        logger.debug(f"Final output shape: {x.shape}")
        
        return x
        
    def train_model(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            validation_data (tuple): (X_val, y_val) for validation
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        try:
            # Convert data to PyTorch tensors
            if np.iscomplexobj(X_train):
                X_train = torch.complex(
                    torch.FloatTensor(np.real(X_train)),
                    torch.FloatTensor(np.imag(X_train))
                )
            else:
                X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).reshape(-1, 1)
            
            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                if np.iscomplexobj(X_val):
                    X_val = torch.complex(
                        torch.FloatTensor(np.real(X_val)),
                        torch.FloatTensor(np.imag(X_val))
                    )
                else:
                    X_val = torch.FloatTensor(X_val)
                y_val = torch.FloatTensor(y_val).reshape(-1, 1)
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters())
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [] if validation_data is not None else None
            }
            
            # Training loop
            for epoch in range(epochs):
                self.train()  # Set model to training mode
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Calculate average loss for the epoch
                train_loss /= len(train_loader)
                history['train_loss'].append(train_loss)
                
                # Validation
                if validation_data is not None:
                    self.eval()  # Set model to evaluation mode
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = self(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    history['val_loss'].append(val_loss)
                    
                    logger.info(
                        "Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f",
                        epoch + 1, epochs, train_loss, val_loss
                    )
                else:
                    logger.info(
                        "Epoch [%d/%d], Train Loss: %.4f",
                        epoch + 1, epochs, train_loss
                    )
            
            return history
            
        except Exception as e:
            logger.error("Error during training: %s", str(e))
            raise
            
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        try:
            self.eval()  # Set model to evaluation mode
            X = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self(X)
            
            return predictions.cpu().numpy()
            
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise
            
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
            logger.info("Model saved to %s", path)
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise 