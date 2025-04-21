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
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.25),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            
            # Fourth conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25)
        )
        
        # Calculate the actual output size by passing a dummy input through the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, self.input_shape[0])
            dummy_output = self.conv_layers(dummy_input)
            self.conv_output_size = dummy_output.view(1, -1).size(1)
            logger.info(f"Calculated conv output size: {self.conv_output_size}")
        
        # Dense layers with 6 layers of 64 neurons each
        self.dense_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Handle complex input
        if torch.is_complex(x):
            x_real = x.real
            x_imag = x.imag
            x = torch.stack([x_real, x_imag], dim=1)
        else:
            if x.dim() == 2:  # If input is (batch, length)
                x = x.view(x.size(0), 2, -1)
        
        # Verify input dimensions
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels for complex data, got {x.size(1)}")
        
        # Pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Verify flattened size matches expected size
        if x.size(1) != self.conv_output_size:
            raise ValueError(f"Expected flattened size {self.conv_output_size}, got {x.size(1)}")
        
        # Pass through dense layers
        x = self.dense_layers(x)
        
        return x
        
    def train_model(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, target_accuracy=0.96):
        """
        Train the model for exactly the specified number of epochs.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            validation_data (tuple): (X_val, y_val) for validation
            epochs (int): Number of epochs to train (default: 50)
            batch_size (int): Batch size for training
            target_accuracy (float): Target accuracy to achieve (default: 0.96)
            
        Returns:
            dict: Training history and final number of epochs
        """
        try:
            # Force epochs to be exactly 50
            epochs = 50
            logger.info(f"Starting training with exactly {epochs} epochs")
            
            # Convert data to PyTorch tensors
            if np.iscomplexobj(X_train):
                X_train = torch.complex(
                    torch.FloatTensor(np.real(X_train)),
                    torch.FloatTensor(np.imag(X_train))
                )
            else:
                X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).reshape(-1, 1)
            
            logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
            
            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"Number of training batches: {len(train_loader)}")
            
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
                logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
                logger.info(f"Number of validation batches: {len(val_loader)}")
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [] if validation_data is not None else None,
                'val_accuracy': [] if validation_data is not None else None
            }
            
            best_accuracy = 0.0
            best_model_state = None
            
            # Strict training loop
            completed_epochs = 0
            while completed_epochs < epochs:
                try:
                    epoch = completed_epochs
                    logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                    
                    # Training phase
                    self.train()
                    train_loss = 0.0
                    batch_count = 0
                    
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        batch_count += 1
                        
                        if batch_count % 10 == 0:
                            logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
                    train_loss /= len(train_loader)
                    history['train_loss'].append(train_loss)
                    
                    # Validation phase
                    if validation_data is not None:
                        self.eval()
                        val_loss = 0.0
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                                outputs = self(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()
                                
                                predicted = (outputs > 0.5).float()
                                total += batch_y.size(0)
                                correct += (predicted == batch_y).sum().item()
                        
                        val_loss /= len(val_loader)
                        val_accuracy = correct / total
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)
                        
                        logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
                        
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            best_model_state = self.state_dict().copy()
                            logger.info(f"New best accuracy: {best_accuracy:.4f}")
                    else:
                        logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")
                    
                    # Increment completed epochs only after successful completion
                    completed_epochs += 1
                    logger.info(f"Completed epoch {epoch + 1}/{epochs}")
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {str(e)}")
                    # Don't increment completed_epochs if there was an error
                    continue
            
            # Verify we completed all epochs
            if completed_epochs != epochs:
                logger.error(f"Training stopped at {completed_epochs} epochs instead of {epochs}")
                raise RuntimeError(f"Training did not complete all {epochs} epochs")
            
            # Restore best model at the end of training
            if best_model_state is not None:
                logger.info("Restoring best model state")
                self.load_state_dict(best_model_state)
            
            # Add final metrics to history
            history['final_epochs'] = completed_epochs
            history['final_accuracy'] = best_accuracy if validation_data is not None else None
            
            logger.info(f"Training completed successfully with {completed_epochs} epochs. Final accuracy: {best_accuracy:.4f}")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
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