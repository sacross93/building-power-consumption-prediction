"""
GPU-Accelerated Final Power Consumption Prediction Solution
=========================================================

GPU-optimized XGBoost solution for faster training and better performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def check_gpu_availability():
    """Check if GPU is available for XGBoost."""
    try:
        # Test GPU availability
        dtrain = xgb.DMatrix(np.random.rand(100, 10), label=np.random.rand(100))
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        
        # Try to train a small model
        model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("✅ GPU is available for XGBoost")
        return True
    except Exception as e:
        print(f"❌ GPU not available: {e}")
        print("Falling back to CPU training")
        return False


def enhanced_feature_engineering_gpu(train_df, test_df):
    """Enhanced feature engineering optimized for GPU training."""
    print("Creating GPU-optimized features...")
    
    def create_features(df):
        df = df.copy()
        
        # Parse datetime if needed
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday  
        df['month'] = df['datetime'].dt.month
        df['week'] = df['datetime'].dt.isocalendar().week
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['quarter'] = df['datetime'].dt.quarter
        
        # Business time indicators
        df['is_weekend'] = (df['weekday'] >= 5).astype(np.float32)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(np.float32)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(np.float32)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(np.float32)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(np.float32)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(np.float32)
        
        # Cyclical encoding (GPU-friendly)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7).astype(np.float32)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7).astype(np.float32)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365).astype(np.float32)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365).astype(np.float32)
        
        # Enhanced weather features (GPU-optimized)
        if 'temp' in df.columns:
            df['temp'] = df['temp'].astype(np.float32)
            df['temp_squared'] = (df['temp'] ** 2).astype(np.float32)
            df['temp_cubed'] = (df['temp'] ** 3).astype(np.float32)
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0).astype(np.float32)
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0).astype(np.float32)
            df['temp_comfort'] = np.abs(df['temp'] - 22).astype(np.float32)
            
            # Weather interactions
            if 'humidity' in df.columns:
                df['humidity'] = df['humidity'].astype(np.float32)
                df['temp_humidity'] = (df['temp'] * df['humidity']).astype(np.float32)
                df['humidity_squared'] = (df['humidity'] ** 2).astype(np.float32)
            
            if 'wind_speed' in df.columns:
                df['wind_speed'] = df['wind_speed'].astype(np.float32)
                df['temp_wind'] = (df['temp'] * df['wind_speed']).astype(np.float32)
            
            if 'rainfall' in df.columns:
                df['rainfall'] = df['rainfall'].astype(np.float32)
                df['has_rain'] = (df['rainfall'] > 0).astype(np.float32)
                df['rain_log'] = np.log1p(df['rainfall']).astype(np.float32)
            
            # Temperature zones
            df['temp_very_cold'] = (df['temp'] < 10).astype(np.float32)
            df['temp_cold'] = ((df['temp'] >= 10) & (df['temp'] < 18)).astype(np.float32)
            df['temp_comfortable'] = ((df['temp'] >= 18) & (df['temp'] < 26)).astype(np.float32)
            df['temp_hot'] = ((df['temp'] >= 26) & (df['temp'] < 30)).astype(np.float32)
            df['temp_very_hot'] = (df['temp'] >= 30).astype(np.float32)
        
        # Building features (GPU-optimized)
        if 'total_area' in df.columns:
            df['total_area'] = df['total_area'].astype(np.float32)
            df['cooling_area'] = df['cooling_area'].astype(np.float32)
            df['area_ratio'] = (df['cooling_area'] / (df['total_area'] + 1)).astype(np.float32)
            df['log_total_area'] = np.log1p(df['total_area']).astype(np.float32)
            df['log_cooling_area'] = np.log1p(df['cooling_area']).astype(np.float32)
            
            # Area categories
            df['small_building'] = (df['total_area'] < 10000).astype(np.float32)
            df['medium_building'] = ((df['total_area'] >= 10000) & (df['total_area'] < 50000)).astype(np.float32)
            df['large_building'] = (df['total_area'] >= 50000).astype(np.float32)
        
        # PV capacity features
        if 'pv_capacity' in df.columns:
            df['pv_capacity'] = df['pv_capacity'].astype(np.float32)
            df['has_pv'] = (df['pv_capacity'] > 0).astype(np.float32)
            if 'total_area' in df.columns:
                df['pv_per_area'] = (df['pv_capacity'] / (df['total_area'] + 1)).astype(np.float32)
        
        return df
    
    # Apply feature engineering
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # Sort by building and datetime for lag features
    train_fe = train_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    test_fe = test_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    
    # Create optimized lag features
    print("Creating optimized lag features...")
    
    def add_lag_features(df, is_test=False):
        df = df.copy()
        
        # Essential lag features only (for GPU efficiency)
        lag_periods = [1, 24, 168]  # 1h, 1day, 1week
        
        for building_id in df['건물번호'].unique():
            mask = df['건물번호'] == building_id
            building_data = df[mask].copy()
            
            if '전력소비량(kWh)' in building_data.columns:
                power_series = building_data['전력소비량(kWh)'].astype(np.float32)
                
                for lag in lag_periods:
                    lag_col = f'power_lag_{lag}'
                    df.loc[mask, lag_col] = power_series.shift(lag)
                    
                    # Rolling statistics for important lags only
                    if lag in [1, 24]:
                        df.loc[mask, f'power_rolling_mean_{lag}'] = power_series.rolling(
                            lag, min_periods=1
                        ).mean().shift(1)
                        df.loc[mask, f'power_rolling_std_{lag}'] = power_series.rolling(
                            lag, min_periods=1
                        ).std().shift(1)
        
        # Fill lag features efficiently
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        
        for col in lag_cols:
            # Fill with forward fill then global mean
            global_mean = df[col].mean() if df[col].notna().any() else 1000.0
            df[col] = df[col].fillna(method='ffill').fillna(global_mean).astype(np.float32)
        
        return df
    
    train_fe = add_lag_features(train_fe)
    test_fe = add_lag_features(test_fe, is_test=True)
    
    print(f"GPU-optimized feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def train_gpu_model(train_df, use_gpu=True):
    """Train XGBoost model with GPU acceleration."""
    print("Training GPU-accelerated model...")
    
    # Prepare features
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['전력소비량(kWh)'].astype(np.float32)
    
    # Handle categorical columns with label encoding
    encoders = {}
    categorical_cols = ['건물번호', 'building_type']
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Handle any remaining object columns
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float32)
        else:
            X[col] = X[col].astype(np.float32)
    
    # GPU-optimized parameters
    if use_gpu:
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        n_estimators = 1000  # More estimators with GPU
    else:
        # CPU fallback
        params = {
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        n_estimators = 500  # Fewer estimators for CPU
    
    # Create DMatrix for better GPU performance
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False
    )
    
    return model, feature_cols, encoders


def gpu_validation_and_predict():
    """GPU-accelerated validation and prediction workflow."""
    print("=" * 80)
    print("GPU-ACCELERATED POWER CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # Check GPU availability
    use_gpu = check_gpu_availability()
    
    # Load data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # Apply basic feature engineering first
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # Apply GPU-optimized feature engineering
    train_enhanced, test_enhanced = enhanced_feature_engineering_gpu(train_fe, test_fe)
    
    # Quick validation
    print("\n" + "=" * 60)
    print("GPU MODEL VALIDATION")
    print("=" * 60)
    
    # Simple time split for validation
    cutoff = train_enhanced['datetime'].max() - pd.Timedelta(days=7)
    train_data = train_enhanced[train_enhanced['datetime'] < cutoff]
    val_data = train_enhanced[train_enhanced['datetime'] >= cutoff]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Prepare validation data
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_data.columns if c not in drop_cols]
    
    X_train = train_data[feature_cols].copy()
    y_train = train_data['전력소비량(kWh)'].astype(np.float32)
    X_val = val_data[feature_cols].copy()
    y_val = val_data['전력소비량(kWh)'].astype(np.float32)
    
    # Handle categorical encoding
    encoders = {}
    categorical_cols = ['건물번호', 'building_type']
    
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_val[col] = X_val[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            encoders[col] = le
    
    # Handle remaining object columns
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_val[col] = X_val[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            X_train[col] = X_train[col].astype(np.float32)
            X_val[col] = X_val[col].astype(np.float32)
    
    # Train validation model
    if use_gpu:
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        n_estimators = 800
    else:
        params = {
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        n_estimators = 400
    
    # Create DMatrix and train
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(dval)
    validation_smape = smape(y_val.values, y_pred)
    
    print(f"Validation SMAPE: {validation_smape:.4f}")
    
    if validation_smape <= 6.0:
        print(f"🎯 Model meets target (SMAPE ≤ 6%)!")
    else:
        improvement_needed = validation_smape - 6.0
        print(f"❌ Need {improvement_needed:.2f}% improvement to reach target")
    
    # Train final model on all data
    print("\n" + "=" * 60)
    print("TRAINING FINAL GPU MODEL")
    print("=" * 60)
    
    final_model, feature_cols, encoders = train_gpu_model(train_enhanced, use_gpu)
    
    # Make predictions
    print("\n" + "=" * 60)
    print("MAKING GPU PREDICTIONS")
    print("=" * 60)
    
    # Handle feature mismatch between train and test
    test_feature_cols = [col for col in feature_cols if col in test_enhanced.columns]
    missing_features = [col for col in feature_cols if col not in test_enhanced.columns]
    
    if missing_features:
        print(f"Missing features in test data: {missing_features}")
        print("Adding missing features with default values...")
        for col in missing_features:
            if 'lag' in col or 'rolling' in col:
                test_enhanced[col] = 1000.0  # Default power value
            else:
                test_enhanced[col] = 0.0  # Default for other features
    
    X_test = test_enhanced[feature_cols].copy()
    
    # Apply same encoding to test data
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    # Handle remaining columns
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
        else:
            X_test[col] = X_test[col].astype(np.float32)
    
    # Create test DMatrix and predict
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    predictions = final_model.predict(dtest)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    # Create submission
    submission = pd.DataFrame({
        'num_date_time': test_enhanced['num_date_time'],
        'prediction': predictions
    })
    
    submission_file = 'submission_gpu_final.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ GPU-accelerated solution completed!")
    print(f"GPU used: {'Yes' if use_gpu else 'No (CPU fallback)'}")
    print(f"Validation SMAPE: {validation_smape:.4f}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: {submission_file}")
    
    return submission, validation_smape, use_gpu


if __name__ == "__main__":
    submission, smape_score, gpu_used = gpu_validation_and_predict()
    
    print(f"\n" + "=" * 80)
    print("FINAL GPU RESULTS SUMMARY")
    print("=" * 80)
    print(f"GPU Acceleration: {'Enabled' if gpu_used else 'Disabled (CPU fallback)'}")
    print(f"Final Validation SMAPE: {smape_score:.4f}")
    
    if smape_score <= 6.0:
        print("🎯 TARGET ACHIEVED: SMAPE ≤ 6%")
    else:
        gap = smape_score - 6.0
        print(f"❌ Target missed by {gap:.2f}%")
        
        print("\nNext optimization steps:")
        print("- Ensemble multiple GPU models")
        print("- Advanced hyperparameter tuning with Optuna")
        print("- Building-specific GPU models")
        print("- Time series cross-validation with GPU")
    
    print(f"\nSubmission file: submission_gpu_final.csv")
    print(f"Ready for submission!")