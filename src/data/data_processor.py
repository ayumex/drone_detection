import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'data_processing.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for loading and preprocessing drone detection data."""
    
    def __init__(self, drone_data_path, no_drone_data_path, segment_size=1024, downsample_factor=10):
        """
        Initialize the DataProcessor.
        
        Args:
            drone_data_path (str): Path to .npz file containing drone signals
            no_drone_data_path (str): Path to .npz file containing background noise
            segment_size (int): Size of each segment to process
            downsample_factor (int): Factor by which to downsample the signal
        """
        self.drone_data_path = drone_data_path
        self.no_drone_data_path = no_drone_data_path
        self.segment_size = segment_size
        self.downsample_factor = downsample_factor
        self.scaler_real = StandardScaler()
        self.scaler_imag = StandardScaler()
    
    def segment_and_downsample(self, data):
        """
        Segment and downsample the input data.
        
        Args:
            data (np.ndarray): Input IQ samples
            
        Returns:
            np.ndarray: Segmented and downsampled data
        """
        # Downsample the data
        downsampled = signal.decimate(data, self.downsample_factor, axis=1)
        
        # Calculate number of complete segments
        n_samples = downsampled.shape[1]
        n_segments = n_samples // self.segment_size
        
        # Reshape into segments, dropping any incomplete segment at the end
        segments = downsampled[:, :n_segments * self.segment_size].reshape(-1, self.segment_size)
        
        return segments
        
    def load_and_combine_data(self):
        """
        Load and combine drone and no-drone data.
        
        Returns:
            tuple: (features, labels) where features is a numpy array of signal data
                  and labels is a numpy array of binary labels (1 for drone, 0 for no drone)
        """
        try:
            logger.info("Loading drone data from %s", self.drone_data_path)
            drone_data = np.load(self.drone_data_path)
            drone_features = drone_data['iq_samples']
            
            logger.info("Loading no-drone data from %s", self.no_drone_data_path)
            no_drone_data = np.load(self.no_drone_data_path)
            no_drone_features = no_drone_data['iq_samples']
            
            # Process drone data
            drone_segments = self.segment_and_downsample(drone_features)
            drone_labels = np.ones(len(drone_segments))
            
            # Process no-drone data
            no_drone_segments = self.segment_and_downsample(no_drone_features)
            no_drone_labels = np.zeros(len(no_drone_segments))
            
            # Combine data
            features = np.vstack((drone_segments, no_drone_segments))
            labels = np.hstack((drone_labels, no_drone_labels))
            
            logger.info("Successfully loaded and combined data:")
            logger.info("Total segments: %d", len(features))
            logger.info("Segment size: %d", self.segment_size)
            logger.info("Features shape: %s", str(features.shape))
            
            return features, labels
            
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise
            
    def preprocess_data(self, features, labels, test_size=0.2, random_state=42):
        """
        Preprocess the data by splitting into train/test sets and scaling features.
        Handles complex IQ samples by separating real and imaginary parts.
        
        Args:
            features (np.ndarray): Input features (complex IQ samples)
            labels (np.ndarray): Target labels
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # Split data first
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state,
                stratify=labels
            )
            
            # Separate real and imaginary parts
            X_train_real = np.real(X_train)
            X_train_imag = np.imag(X_train)
            X_test_real = np.real(X_test)
            X_test_imag = np.imag(X_test)
            
            # Scale real and imaginary parts separately
            X_train_real_scaled = self.scaler_real.fit_transform(X_train_real)
            X_train_imag_scaled = self.scaler_imag.fit_transform(X_train_imag)
            X_test_real_scaled = self.scaler_real.transform(X_test_real)
            X_test_imag_scaled = self.scaler_imag.transform(X_test_imag)
            
            # Combine real and imaginary parts into complex numbers
            X_train_processed = X_train_real_scaled + 1j * X_train_imag_scaled
            X_test_processed = X_test_real_scaled + 1j * X_test_imag_scaled
            
            logger.info("Data preprocessed successfully")
            logger.info("Training set size: %d", len(X_train_processed))
            logger.info("Test set size: %d", len(X_test_processed))
            logger.info("Processed feature shape: %s", str(X_train_processed.shape))
            
            return X_train_processed, X_test_processed, y_train, y_test
            
        except Exception as e:
            logger.error("Error preprocessing data: %s", str(e))
            raise
            
    def save_scaler(self, path):
        """
        Save the fitted scalers for later use.
        
        Args:
            path (str): Base path to save the scalers
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save both scalers with appropriate suffixes
            real_path = path.replace('.joblib', '_real.joblib')
            imag_path = path.replace('.joblib', '_imag.joblib')
            
            joblib.dump(self.scaler_real, real_path)
            joblib.dump(self.scaler_imag, imag_path)
            
            logger.info("Scalers saved to %s and %s", real_path, imag_path)
        except Exception as e:
            logger.error("Error saving scalers: %s", str(e))
            raise 