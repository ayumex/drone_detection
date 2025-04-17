import argparse
import logging
import os
import yaml
import mlflow
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.models.drone_detector import DroneDetector
from src.evaluation.evaluator import ModelEvaluator

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'training.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train drone detection model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--drone_data', type=str, required=True,
                      help='Path to .npz file containing drone signals')
    parser.add_argument('--no_drone_data', type=str, required=True,
                      help='Path to .npz file containing background noise (no drone)')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize MLflow
        mlflow.set_experiment('drone_detection')
        
        # Initialize data processor
        data_processor = DataProcessor(args.drone_data, args.no_drone_data)
        
        # Load and preprocess data
        features, labels = data_processor.load_and_combine_data()
        X_train, X_test, y_train, y_test = data_processor.preprocess_data(features, labels)
        
        # Save scaler for inference
        data_processor.save_scaler('models/scaler.joblib')
        
        # Initialize and train model
        input_shape = (X_train.shape[1],)  # Assuming 1D input
        model = DroneDetector(input_shape)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'input_shape': input_shape,
                'conv_blocks': config['model']['conv_blocks'],
                'dense_layers': config['model']['dense_layers'],
                'neurons_per_layer': config['model']['neurons_per_layer'],
                'dropout_rate': config['model']['dropout_rate'],
                'n_drone_samples': sum(labels == 1),
                'n_no_drone_samples': sum(labels == 0)
            })
            
            # Train model
            history = model.train_model(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=config['training']['epochs'],
                batch_size=config['training']['batch_size']
            )
            
            # Save model
            model.save_model('models/drone_detector.pt')
            
            # Evaluate model
            evaluator = ModelEvaluator()
            evaluator.evaluate_model(model, X_test, y_test)
            
            # Log metrics and artifacts
            mlflow.log_artifacts('results')
            
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 