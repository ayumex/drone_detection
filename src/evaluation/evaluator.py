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
    accuracy_score
)
import logging
import os
import torch

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
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained PyTorch model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        try:
            # Convert data to PyTorch tensors
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).cpu().numpy().squeeze()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Generate and save evaluation results
            self._generate_classification_report(y_test, y_pred)
            self._plot_confusion_matrix(y_test, y_pred)
            self._plot_roc_curve(y_test, y_pred_proba)
            self._plot_precision_recall_curve(y_test, y_pred_proba)
            
            # Log results
            logger.info("Model evaluation completed successfully")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
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