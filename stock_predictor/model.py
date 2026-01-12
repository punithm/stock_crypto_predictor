"""
Machine learning models for stock price prediction.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# TensorFlow is imported lazily only when LSTMModel is used (see _load_tensorflow() below)
_tf = None
_keras = None

def _load_tensorflow():
    """Lazy-load TensorFlow only when LSTM is needed."""
    global _tf, _keras
    if _tf is None:
        print("[TensorFlow] Initializing... this may take 10-20 seconds on first load")
        import tensorflow as _tf_module
        from tensorflow import keras as _keras_module
        _tf = _tf_module
        _keras = _keras_module
        print("[TensorFlow] ✅ Loaded successfully")
    return _tf, _keras


class StockPredictionModel:
    """Base class for stock prediction models."""

    def __init__(self):
        self.model = None
        # Separate scalers for multivariate inputs and target
        self.x_scaler = None
        self.y_scaler = None
        self.is_trained = False

    def normalize_data(self, X):
        """Deprecated generic normalize helper.
        Prefer per-model scaling in training/predict methods.
        """
        return X

    def train(self, X, y):
        """Train the model. To be implemented by subclasses."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions. To be implemented by subclasses."""
        raise NotImplementedError


class LSTMModel(StockPredictionModel):
    """LSTM-based model for time series prediction."""

    def __init__(self, lookback_days: int = 60, epochs: int = 50, batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Args:
            lookback_days: Sequence length for LSTM
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        super().__init__()
        self.lookback_days = lookback_days
        self.epochs = epochs
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        """Build the LSTM neural network."""
        # Lazy load TensorFlow
        tf, keras = _load_tensorflow()
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        
        # input_shape will be set when training based on X.shape
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X, y):
        """
        Train the LSTM model.
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training labels
        """
        # Fit scalers for multivariate X and y
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        X_flat_scaled = self.x_scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, n_timesteps, n_features)

        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = self.y_scaler.fit_transform(y).flatten()

        # Rebuild model with correct input shape if needed
        if not self.model or self.model.layers[0].input_shape is None:
            self._build_model()

        # Ensure model input shape matches
        try:
            if self.model.layers[0].input_shape[1:] != (n_timesteps, n_features):
                # rebuild with the right input shape
                self.model = Sequential([
                    LSTM(50, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception:
            pass

        print(f"Training LSTM model with {self.epochs} epochs...")
        self.model.fit(
            X_scaled, y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1
        )
        self.is_trained = True
        print("Model training completed")

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input sequences (samples, timesteps, features)
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Apply x_scaler then predict, inverse transform using y_scaler
        if self.x_scaler is None or self.y_scaler is None:
            raise ValueError("Scalers not found. Train the model before predicting.")

        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat_scaled = self.x_scaler.transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, n_timesteps, n_features)

        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        return predictions


class EnsembleModel(StockPredictionModel):
    """Simple adaptive ensemble that weights models by inverse RMSE."""

    def __init__(self, models: dict = None):
        super().__init__()
        self.models = models or {}
        self.weights = {k: 1.0 / len(self.models) for k in self.models} if self.models else {}

    def fit_weights(self, y_true, preds_dict):
        """Compute weights based on inverse RMSE (normalized)."""
        scores = {}
        for name, preds in preds_dict.items():
            mse = np.mean((y_true.flatten() - preds.flatten()) ** 2)
            rmse = np.sqrt(mse)
            scores[name] = rmse

        inv = {k: 1.0 / (v + 1e-8) for k, v in scores.items()}
        s = sum(inv.values())
        self.weights = {k: v / s for k, v in inv.items()}
        return self.weights

    def predict(self, X):
        if not self.models:
            raise ValueError("No models in ensemble")
        preds = None
        for name, model in self.models.items():
            p = model.predict(X)
            w = self.weights.get(name, 1.0 / len(self.models))
            if preds is None:
                preds = w * p
            else:
                preds += w * p
        return preds


class RandomForestModel(StockPredictionModel):
    """Random Forest model for stock prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 20):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training labels
        """
        # Flatten the 3D array to 2D for Random Forest
        X_reshaped = X.reshape(X.shape[0], -1)
        
        print("Training Random Forest model...")
        self.model.fit(X_reshaped, y.ravel())
        self.is_trained = True
        print("Model training completed")

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features (samples, timesteps, features)
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped).reshape(-1, 1)


class SimpleLinearModel(StockPredictionModel):
    """Simple Linear Regression model for stock prediction."""

    def __init__(self):
        """Initialize Linear Regression model."""
        super().__init__()
        self.model = LinearRegression()

    def train(self, X, y):
        """
        Train the Linear Regression model.
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training labels
        """
        # Flatten the 3D array to 2D for Linear Regression
        X_reshaped = X.reshape(X.shape[0], -1)
        
        print("Training Linear Regression model...")
        self.model.fit(X_reshaped, y.ravel())
        self.is_trained = True
        print("Model training completed")

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features (samples, timesteps, features)
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped).reshape(-1, 1)
