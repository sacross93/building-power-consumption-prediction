"""
LSTM Time Series Model for Power Consumption Prediction
=======================================================

This module implements LSTM/GRU models that can properly capture
temporal dependencies in power consumption data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print(f"TensorFlow version: {tf.__version__}")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    import tensorflow as tf
    from tensorflow import keras  
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True

from ts_validation import smape, TimeSeriesCV


class PowerConsumptionLSTM:
    """LSTM model for power consumption prediction."""
    
    def __init__(self, 
                 sequence_length: int = 24,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 64):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences (hours)
        lstm_units : List[int]
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.building_encoder = LabelEncoder()
        self.building_type_encoder = LabelEncoder()
        
        # Training history
        self.history = None
        
    def prepare_sequences(self, 
                         data: pd.DataFrame,
                         target_col: str = '전력소비량(kWh)',
                         building_col: str = '건물번호',
                         datetime_col: str = 'datetime') -> Dict[str, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_col : str
            Name of target column
        building_col : str
            Name of building column
        datetime_col : str
            Name of datetime column
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing sequences and metadata
        """
        print(f"Preparing sequences with length {self.sequence_length}...")
        
        # Sort by building and datetime
        data = data.sort_values([building_col, datetime_col]).reset_index(drop=True)
        
        # Select relevant features
        feature_cols = [
            'temp', 'humidity', 'rainfall', 'wind_speed',
            'hour', 'weekday', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
            'hour_peak_flag', 'temp_squared', 'temp_cooling_need', 'temp_heating_need',
            'total_area', 'cooling_area', 'area_ratio', 'pv_capacity', 'pv_per_area'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in data.columns]
        print(f"Using {len(available_features)} features: {available_features[:10]}...")
        
        sequences_X = []
        sequences_y = []
        building_ids = []
        timestamps = []
        
        # Process each building separately
        for building_id in data[building_col].unique():
            building_data = data[data[building_col] == building_id].copy()
            
            if len(building_data) < self.sequence_length + 1:
                continue
                
            # Extract features and target
            X_building = building_data[available_features].values
            y_building = building_data[target_col].values
            timestamps_building = building_data[datetime_col].values
            
            # Create sequences
            for i in range(len(building_data) - self.sequence_length):
                # Input sequence (features + lagged target)
                X_seq = X_building[i:i+self.sequence_length]
                y_lag = y_building[i:i+self.sequence_length].reshape(-1, 1)
                
                # Combine features with lagged target
                X_combined = np.concatenate([X_seq, y_lag], axis=1)
                
                # Target (next value)
                y_next = y_building[i+self.sequence_length]
                
                sequences_X.append(X_combined)
                sequences_y.append(y_next)
                building_ids.append(building_id)
                timestamps.append(timestamps_building[i+self.sequence_length])
        
        # Convert to arrays
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        building_ids = np.array(building_ids)
        timestamps = np.array(timestamps)
        
        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return {
            'X': X,
            'y': y,
            'building_ids': building_ids,
            'timestamps': timestamps,
            'feature_names': available_features + ['lag_target']
        }
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of input sequences (sequence_length, n_features)
        """
        print(f"Building LSTM model with input shape: {input_shape}")
        
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_seq = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Dense layers for prediction
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
    
    def fit(self, 
            train_data: Dict[str, np.ndarray],
            val_data: Optional[Dict[str, np.ndarray]] = None,
            epochs: int = 100,
            verbose: int = 1) -> None:
        """
        Train the LSTM model.
        
        Parameters
        ----------
        train_data : Dict[str, np.ndarray]
            Training data from prepare_sequences
        val_data : Optional[Dict[str, np.ndarray]]
            Validation data
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level
        """
        print("Training LSTM model...")
        
        X_train = train_data['X']
        y_train = train_data['y']
        
        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.feature_scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale target
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Prepare validation data if provided
        validation_data = None
        if val_data is not None:
            X_val = val_data['X']
            y_val = val_data['y']
            
            # Scale validation features
            n_val_samples, n_val_timesteps, n_val_features = X_val.shape
            X_val_reshaped = X_val.reshape(-1, n_val_features)
            X_val_scaled = self.feature_scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(n_val_samples, n_val_timesteps, n_val_features)
            
            # Scale validation target
            y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            
            validation_data = (X_val_scaled, y_val_scaled)
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
    
    def predict(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Data from prepare_sequences
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = data['X']
        
        # Scale features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def plot_training_history(self, save_path: Optional[Path] = None) -> None:
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {save_path}")
        
        plt.show()


def evaluate_lstm_with_cv(data: pd.DataFrame, 
                         target_col: str = '전력소비량(kWh)',
                         sequence_length: int = 24,
                         n_splits: int = 3) -> Dict[str, Any]:
    """
    Evaluate LSTM model using time series cross-validation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of target column
    sequence_length : int
        Length of input sequences
    n_splits : int
        Number of CV splits
        
    Returns
    -------
    Dict[str, Any]
        Evaluation results
    """
    print(f"Evaluating LSTM with {n_splits}-fold Time Series CV...")
    
    # Initialize CV
    ts_cv = TimeSeriesCV(n_splits=n_splits, test_size_days=7, gap_days=1)
    splits = ts_cv.split(data, 'datetime')
    
    results = {
        'fold_scores': [],
        'fold_details': [],
        'mean_score': 0,
        'std_score': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{len(splits)}")
        
        # Split data
        train_fold = data.iloc[train_idx]
        val_fold = data.iloc[val_idx]
        
        print(f"Train period: {train_fold['datetime'].min()} to {train_fold['datetime'].max()}")
        print(f"Val period: {val_fold['datetime'].min()} to {val_fold['datetime'].max()}")
        
        # Initialize model
        lstm_model = PowerConsumptionLSTM(
            sequence_length=sequence_length,
            lstm_units=[64, 32],  # Smaller for faster training
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32
        )
        
        # Prepare sequences
        train_sequences = lstm_model.prepare_sequences(train_fold, target_col)
        val_sequences = lstm_model.prepare_sequences(val_fold, target_col)
        
        if len(train_sequences['X']) == 0 or len(val_sequences['X']) == 0:
            print(f"Insufficient data for fold {fold + 1}, skipping...")
            continue
        
        # Train model
        lstm_model.fit(
            train_sequences,
            val_sequences,
            epochs=50,  # Reduced for faster training
            verbose=0
        )
        
        # Predict on validation set
        y_pred = lstm_model.predict(val_sequences)
        y_true = val_sequences['y']
        
        # Calculate SMAPE
        fold_smape = smape(y_true, y_pred)
        results['fold_scores'].append(fold_smape)
        
        # Store fold details
        fold_detail = {
            'fold': fold + 1,
            'smape': fold_smape,
            'train_size': len(train_sequences['X']),
            'val_size': len(val_sequences['X']),
            'train_period': (train_fold['datetime'].min(), train_fold['datetime'].max()),
            'val_period': (val_fold['datetime'].min(), val_fold['datetime'].max())
        }
        results['fold_details'].append(fold_detail)
        
        print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    # Calculate summary statistics
    if results['fold_scores']:
        results['mean_score'] = np.mean(results['fold_scores'])
        results['std_score'] = np.std(results['fold_scores'])
        
        print(f"\nLSTM Cross-Validation Results:")
        print(f"Mean SMAPE: {results['mean_score']:.4f} (±{results['std_score']:.4f})")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in results['fold_scores']]}")
    
    return results


if __name__ == "__main__":
    # This will be called from the main LSTM testing script
    pass