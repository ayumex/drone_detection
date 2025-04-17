# Drone Detection System

This repository contains a deep learning-based system for detecting drones using radio frequency (RF) signals. The system uses complex IQ samples from RF receivers to classify whether a drone is present in the signal or not.

## Overview

The system consists of three main components:
1. Data Processing: Handles loading and preprocessing of IQ samples
2. Deep Learning Model: A convolutional neural network for drone detection
3. Evaluation: Tools for assessing model performance

### Key Features
- Handles complex IQ samples (preserving both real and imaginary components)
- Uses convolutional neural networks for signal processing
- Includes data augmentation for improved model robustness
- Provides comprehensive evaluation metrics
- Supports both training and inference modes

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- SciPy
- MLflow (for experiment tracking)
- TensorBoard (for visualization)

Install the required packages:
```bash
pip install -r requirements.txt
```
To run the file:
Make sure you have the new unzipped dataset.

Run the following commands to execute the code:
Assuming the cloned repo is in `C:\Users\HP\drone_detection:`
Go inside this directory, run:
```bash
python -m venv venv
.\venv\Scripts\Activate
 pip install -r requirements.txt   
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
 pip install PyYAML  
 pip install mlflow 
 pip install seaborn 
 python src/train.py --drone_data D:\Gash\drone_dataset\drone_dataset\drone_data_20250414_144053.npz --no_drone_data D:\Gash\drone_dataset\drone_dataset\background_20250414_142850.npz 
```

## Project Structure
             
```
drone_detection/
├── config/
│   └── model_config.yaml      # Model and training configuration
├── src/
│   ├── data/
│   │   └── data_processor.py  # Data loading and preprocessing
│   ├── models/
│   │   └── drone_detector.py  # Neural network model
│   ├── evaluation/
│   │   └── evaluator.py       # Model evaluation tools
│   └── train.py               # Training script
├── logs/                      # Training and processing logs
├── models/                    # Saved models and scalers
├── results/                   # Evaluation results
└── README.md
```

## Data Format

The system expects input data in the following format:
- `.npz` files containing complex IQ samples
- Each file should have a key 'iq_samples' containing the complex data
- Data should be in the format (n_samples, n_features) where features are complex numbers

## Configuration

The `config/model_config.yaml` file contains all configurable parameters:

- Model Architecture:
  - Number of convolutional blocks
  - Filter sizes
  - Dense layer configuration
  - Dropout rates
  - Activation functions

- Training Parameters:
  - Number of epochs
  - Batch size
  - Learning rate
  - Early stopping patience
  - Validation split

- Data Processing:
  - Sample rate
  - Window size
  - Hop length
  - Normalization settings
  - Data augmentation parameters

## Usage

### Training

To train the model:

```bash
python src/train.py \
    --config config/model_config.yaml \
    --drone_data path/to/drone_data.npz \
    --no_drone_data path/to/no_drone_data.npz
```

### Inference

To use the trained model for inference:

```python
from src.models.drone_detector import DroneDetector
from src.data.data_processor import DataProcessor

# Load the model
model = DroneDetector(input_shape=(1024,))
model.load_state_dict(torch.load('models/drone_detector.pt'))

# Load and preprocess new data
processor = DataProcessor(None, None)  # No paths needed for inference
processor.load_scaler('models/scaler.joblib')

# Process and predict
features = processor.preprocess_data(new_data)
predictions = model.predict(features)
```

## Model Architecture

The model uses a 1D convolutional neural network architecture:
- Input: Complex IQ samples (real and imaginary parts)
- Convolutional layers: 4 blocks with increasing filter sizes
- Dense layers: 6 fully connected layers
- Output: Binary classification (drone present/not present)

## Evaluation Metrics

The system provides several evaluation metrics:
- Accuracy
- F1 Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

Results are saved in the `results/` directory.

## Logging and Monitoring

- Training logs are saved in `logs/training.log`
- Data processing logs in `logs/data_processing.log`
- Model logs in `logs/model.log`
- MLflow is used for experiment tracking
- TensorBoard can be used for visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Special thanks to the open-source community for the tools and libraries used in this project 