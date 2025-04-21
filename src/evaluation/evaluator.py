import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import logging
import os
import torch
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'evaluation.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for evaluating drone detection model performance."""
    
    def __init__(self, output_dir='results'):
        """
        Initialize the ModelEvaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            model (DroneDetector): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        try:
            # Convert test data to tensor with correct format
            if np.iscomplexobj(X_test):
                # Handle complex data by stacking real and imaginary parts
                X_test_tensor = torch.stack([
                    torch.FloatTensor(np.real(X_test)),
                    torch.FloatTensor(np.imag(X_test))
                ], dim=1).to(model.device)
            else:
                # Ensure shape is (batch, 2, length)
                X_test_tensor = torch.FloatTensor(X_test).view(X_test.shape[0], 2, -1).to(model.device)
            
            # Convert labels to tensor
            y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(model.device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Get predictions
            with torch.no_grad():
                predictions = model(X_test_tensor)
            
            # Convert predictions to numpy
            predictions = predictions.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions > 0.5)
            precision = precision_score(y_test, predictions > 0.5)
            recall = recall_score(y_test, predictions > 0.5)
            f1 = f1_score(y_test, predictions > 0.5)
            
            # Log metrics
            logger.info("Model Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Save results
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Save to file
            with open('results/evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            return results
            
        except Exception as e:
            logger.error("Error during model evaluation: %s", str(e))
            raise
            
    def _generate_classification_report(self, y_true, y_pred):
        """Generate and save classification report."""
        try:
            report = classification_report(y_true, y_pred)
            
            # Save to file
            report_path = os.path.join(self.output_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
                
            logger.info(f"Classification report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise
            
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plt_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(plt_path)
            plt.close()
            
            logger.info(f"Confusion matrix plot saved to {plt_path}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot and save ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # Save plot
            plt_path = os.path.join(self.output_dir, 'roc_curve.png')
            plt.savefig(plt_path)
            plt.close()
            
            logger.info(f"ROC curve plot saved to {plt_path}")
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
            
    def _plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot and save precision-recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            
            # Save plot
            plt_path = os.path.join(self.output_dir, 'precision_recall_curve.png')
            plt.savefig(plt_path)
            plt.close()
            
            logger.info(f"Precision-Recall curve plot saved to {plt_path}")
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curve: {str(e)}")
            raise 