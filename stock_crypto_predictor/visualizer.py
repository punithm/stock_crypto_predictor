"""
Module for visualizing stock prices and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np


class StockVisualizer:
    """Visualizes stock price data and model predictions."""

    @staticmethod
    def plot_price_history(data, ticker: str, save_path: str = None):
        """
        Plot historical stock prices.
        
        Args:
            data: DataFrame with stock data
            ticker: Stock symbol
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], linewidth=2, color='blue')
        plt.title(f'{ticker} Historical Close Prices', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (€)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_predictions(y_true, y_pred, title: str = "Model Predictions vs Actual",
                        save_path: str = None):
        """
        Plot actual vs predicted prices.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 6))
        
        x_axis = np.arange(len(y_true))
        plt.plot(x_axis, y_true.flatten(), label='Actual Price', linewidth=2, color='blue')
        plt.plot(x_axis, y_pred.flatten(), label='Predicted Price', linewidth=2, 
                color='red', alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Price (€)')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_comparison(y_true, *predictions, titles: list = None, save_path: str = None):
        """
        Compare multiple models' predictions.
        
        Args:
            y_true: Actual values
            predictions: Variable number of prediction arrays
            titles: Labels for each prediction (optional)
            save_path: Optional path to save the figure
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        x_axis = np.arange(len(y_true))
        
        for idx, (y_pred, ax) in enumerate(zip(predictions, axes)):
            ax.plot(x_axis, y_true.flatten(), label='Actual', linewidth=2, color='blue')
            ax.plot(x_axis, y_pred.flatten(), label='Predicted', linewidth=2, 
                   color='red', alpha=0.7)
            
            title = titles[idx] if titles and idx < len(titles) else f"Model {idx + 1}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Price (€)')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Sample Index')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
