"""
Quick LSTM Test for Performance Comparison
==========================================

Simplified LSTM test to quickly compare with XGBoost.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
import sys
sys.path.append('.')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not available")
    exit(1)

from ts_validation import smape


def create_simple_sequences(data, target_col='전력소비량(kWh)', seq_len=12):
    """Create simple sequences for LSTM."""
    # Sort by building and datetime
    data = data.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    
    # Simple features
    feature_cols = ['temp', 'humidity', 'hour', 'weekday', 'month', 'is_weekend']
    available_features = [col for col in feature_cols if col in data.columns]
    
    sequences_X = []
    sequences_y = []
    
    # Sample only some buildings for speed
    buildings = data['건물번호'].unique()[:20]  # Only first 20 buildings
    
    for building_id in buildings:
        building_data = data[data['건물번호'] == building_id].copy()
        
        if len(building_data) < seq_len + 1:
            continue
            
        # Sample every 4th point for speed
        building_data = building_data.iloc[::4].reset_index(drop=True)
        
        if len(building_data) < seq_len + 1:
            continue
        
        X_building = building_data[available_features].values
        y_building = building_data[target_col].values
        
        # Create sequences
        for i in range(len(building_data) - seq_len):
            X_seq = X_building[i:i+seq_len]
            y_next = y_building[i+seq_len]
            
            sequences_X.append(X_seq)
            sequences_y.append(y_next)
    
    return np.array(sequences_X), np.array(sequences_y)


def build_simple_lstm(input_shape):
    """Build a simple LSTM model."""
    model = keras.Sequential([
        layers.LSTM(32, input_shape=input_shape, dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def quick_comparison():
    """Quick comparison between XGBoost and LSTM."""
    print("=" * 60)
    print("QUICK LSTM vs XGBOOST COMPARISON")
    print("=" * 60)
    
    # Load data
    from solution import load_data, engineer_features
    
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    print(f"Data loaded: {len(train_fe)} samples")
    
    # Simple time split
    cutoff = train_fe['datetime'].max() - pd.Timedelta(days=7)
    train_data = train_fe[train_fe['datetime'] < cutoff]
    val_data = train_fe[train_fe['datetime'] >= cutoff]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # 1. Test XGBoost (simplified)
    print("\n1. Testing XGBoost...")
    
    import xgboost as xgb
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_data.columns if c not in drop_cols]
    categorical_cols = ['건물번호', 'building_type']
    categorical_cols = [col for col in categorical_cols if col in feature_cols]
    
    # Prepare XGBoost data
    X_train = train_data[feature_cols].copy()
    y_train = train_data['전력소비량(kWh)']
    X_val = val_data[feature_cols].copy()
    y_val = val_data['전력소비량(kWh)']
    
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    
    xgb_model = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    xgb_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', xgb_model)])
    xgb_pipeline.fit(X_train, y_train)
    
    xgb_pred = xgb_pipeline.predict(X_val)
    xgb_smape = smape(y_val.values, xgb_pred)
    
    print(f"XGBoost SMAPE: {xgb_smape:.4f}")
    
    # 2. Test LSTM
    print("\n2. Testing LSTM...")
    
    # Create sequences
    print("Creating sequences...")
    X_train_seq, y_train_seq = create_simple_sequences(train_data)
    X_val_seq, y_val_seq = create_simple_sequences(val_data)
    
    print(f"Train sequences: {len(X_train_seq)}, Val sequences: {len(X_val_seq)}")
    
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        print("❌ No sequences created")
        return
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    
    # Flatten for scaling
    n_train_samples, seq_len, n_features = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(-1, n_features)
    
    n_val_samples, _, _ = X_val_seq.shape
    X_val_flat = X_val_seq.reshape(-1, n_features)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(n_train_samples, seq_len, n_features)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(n_val_samples, seq_len, n_features)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val_seq.reshape(-1, 1)).flatten()
    
    # Build and train LSTM
    print("Building LSTM model...")
    lstm_model = build_simple_lstm((seq_len, n_features))
    
    print("Training LSTM...")
    lstm_model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=20,
        batch_size=64,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Predict
    print("Making predictions...")
    lstm_pred_scaled = lstm_model.predict(X_val_scaled, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()
    lstm_pred = np.maximum(lstm_pred, 0)  # Ensure non-negative
    
    lstm_smape = smape(y_val_seq, lstm_pred)
    
    print(f"LSTM SMAPE: {lstm_smape:.4f}")
    
    # 3. Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"XGBoost SMAPE: {xgb_smape:.4f}")
    print(f"LSTM SMAPE:    {lstm_smape:.4f}")
    
    improvement = xgb_smape - lstm_smape
    if improvement > 0:
        print(f"✅ LSTM improves by {improvement:.4f} SMAPE points")
    else:
        print(f"❌ XGBoost better by {-improvement:.4f} SMAPE points")
    
    target = 6.0
    print(f"\nTarget Achievement (SMAPE ≤ {target}%):")
    print(f"XGBoost: {'✅' if xgb_smape <= target else '❌'} ({xgb_smape:.2f}%)")
    print(f"LSTM:    {'✅' if lstm_smape <= target else '❌'} ({lstm_smape:.2f}%)")
    
    return {
        'xgboost_smape': xgb_smape,
        'lstm_smape': lstm_smape,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = quick_comparison()