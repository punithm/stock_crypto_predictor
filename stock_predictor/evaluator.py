"""
Module for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """Evaluates model predictions against actual values."""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

    @staticmethod
    def print_metrics(metrics):
        """Print metrics in a formatted way."""
        print("\n=== Model Performance Metrics ===")
        for key, value in metrics.items():
            if key == 'R2':
                print(f"{key:6s}: {value:.4f}")
            elif key == 'MAPE':
                print(f"{key:6s}: {value:.2f}%")
            else:
                print(f"{key:6s}: {value:.4f}")
        print("=" * 35)

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """
        Calculate directional accuracy (% of correct up/down predictions).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Accuracy percentage
        """
        y_true_direction = np.diff(y_true.flatten()) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        
        accuracy = np.mean(y_true_direction == y_pred_direction) * 100
        return accuracy
