"""
Final Power Consumption Prediction Solution
==========================================

Optimized XGBoost solution to achieve SMAPE ≤ 6%.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our validation tools
from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def enhanced_feature_engineering(train_df, test_df):
    """Enhanced feature engineering based on validation insights."""
    print("Creating enhanced features...")
    
    def create_features(df):
        df = df.copy()
        
        # Parse datetime if needed
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # Enhanced time features
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday  
        df['month'] = df['datetime'].dt.month
        df['week'] = df['datetime'].dt.isocalendar().week
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Business and peak indicators
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Cyclical encoding for better continuity
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weather features (focus on temperature as it's most important)
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0)
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0)
            df['temp_comfort'] = np.abs(df['temp'] - 22)
            
            # Weather interactions
            if 'humidity' in df.columns:
                df['temp_humidity'] = df['temp'] * df['humidity']
            df['temp_hour'] = df['temp'] * df['hour']
        
        # Building features - handle missing values properly
        if 'total_area' in df.columns:
            df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
            df['log_total_area'] = np.log1p(df['total_area'])
        
        # PV capacity
        if 'pv_capacity' in df.columns:
            df['has_pv'] = (df['pv_capacity'] > 0).astype(int)
            if 'total_area' in df.columns:
                df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
        
        return df
    
    # Apply feature engineering
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # Sort by building and datetime for lag features
    train_fe = train_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    test_fe = test_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    
    # Create lag features (focused on most important ones)
    print("Creating lag features...")
    
    def add_lag_features(df, is_test=False):
        df = df.copy()
        
        # Only create essential lag features to avoid overfitting
        lag_periods = [1, 24, 168]  # 1h, 1day, 1week
        
        for building_id in df['건물번호'].unique():
            mask = df['건물번호'] == building_id
            building_data = df[mask].copy()
            
            if '전력소비량(kWh)' in building_data.columns:
                power_series = building_data['전력소비량(kWh)']
                
                for lag in lag_periods:
                    df.loc[mask, f'power_lag_{lag}'] = power_series.shift(lag)
                    
                    # Rolling mean for smoothing
                    if lag <= 24:
                        df.loc[mask, f'power_rolling_mean_{lag}'] = power_series.rolling(lag, min_periods=1).mean().shift(1)
        
        # Fill lag features
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        
        for col in lag_cols:
            # Fill with global mean for missing values
            global_mean = df[col].mean() if df[col].notna().any() else 1000
            df[col] = df[col].fillna(global_mean)
        
        return df
    
    train_fe = add_lag_features(train_fe)
    test_fe = add_lag_features(test_fe, is_test=True)
    
    print(f"Enhanced feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def train_optimized_model(train_df):
    """Train optimized XGBoost model."""
    print("Training optimized model...")
    
    # Prepare features
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['전력소비량(kWh)']
    
    # Handle categorical and object columns
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Train with carefully tuned parameters based on validation insights
    model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=500,
        learning_rate=0.05,  # Lower learning rate for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,  # L1 regularization
        reg_lambda=1,  # L2 regularization  
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    return model, feature_cols, encoders


def validate_and_predict():
    """Full workflow: validate, train, and predict."""
    print("=" * 80)
    print("FINAL POWER CONSUMPTION PREDICTION SOLUTION")
    print("=" * 80)
    
    # Load data using existing functions
    data_dir = Path('./data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # Apply our basic feature engineering first
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # Apply enhanced feature engineering
    train_enhanced, test_enhanced = enhanced_feature_engineering(train_fe, test_fe)
    
    # Validate model performance
    print("\n" + "=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_enhanced.columns if c not in drop_cols]
    
    X = train_enhanced[feature_cols].copy()
    y = train_enhanced['전력소비량(kWh)']
    
    # Handle data types
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Time series cross-validation
    ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_enhanced, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Validation fold {fold + 1}/3")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx] 
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(
            max_depth=8,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        fold_smape = smape(y_val.values, y_pred)
        validation_scores.append(fold_smape)
        
        print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    mean_smape = np.mean(validation_scores)
    std_smape = np.std(validation_scores)
    
    print(f"\nValidation Results:")
    print(f"Mean SMAPE: {mean_smape:.4f} (±{std_smape:.4f})")
    
    if mean_smape <= 6.0:
        print(f"🎯 Model meets target (SMAPE ≤ 6%)!")
    else:
        improvement_needed = mean_smape - 6.0
        print(f"❌ Need {improvement_needed:.2f}% improvement to reach target")
    
    # Train final model on all data
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    model, feature_cols, encoders = train_optimized_model(train_enhanced)
    
    # Make predictions
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    X_test = test_enhanced[feature_cols].copy()
    
    # Apply same encoding as training
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
            X_test[col] = X_test[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
    # Handle any remaining non-numeric columns
    for col in X_test.columns:
        if not pd.api.types.is_numeric_dtype(X_test[col]):
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    # Create submission
    submission = pd.DataFrame({
        'num_date_time': test_enhanced['num_date_time'],
        'prediction': predictions
    })
    
    submission.to_csv('submission_final.csv', index=False)
    
    print(f"\n✅ Final solution completed!")
    print(f"Validation SMAPE: {mean_smape:.4f}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: submission_final.csv")
    
    return {
        'validation_smape': mean_smape,
        'validation_std': std_smape,
        'predictions': predictions,
        'submission': submission
    }


if __name__ == "__main__":
    results = validate_and_predict()