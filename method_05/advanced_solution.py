"""
Advanced Power Consumption Prediction Solution
=============================================

Enhanced XGBoost solution with advanced feature engineering and optimization
to achieve SMAPE ≤ 6%.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import optuna
import warnings
warnings.filterwarnings('ignore')

from ts_validation import smape, TimeSeriesCV


class AdvancedPowerPredictor:
    """Advanced power consumption predictor with enhanced features."""
    
    def __init__(self):
        self.models = {}
        self.building_encoders = {}
        self.feature_scalers = {}
        self.building_stats = {}
        
    def load_data(self, train_path, test_path, building_path):
        """Load and merge data files."""
        print("Loading data...")
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        building_df = pd.read_csv(building_path)
        
        # Merge with building info
        train_df = train_df.merge(building_df, on='건물번호', how='left')
        test_df = test_df.merge(building_df, on='건물번호', how='left')
        
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
        
    def advanced_feature_engineering(self, train_df, test_df):
        """Create advanced features for better prediction."""
        print("Creating advanced features...")
        
        def create_features(df):
            df = df.copy()
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
            
            # Basic time features
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['week'] = df['datetime'].dt.isocalendar().week
            df['is_weekend'] = (df['weekday'] >= 5).astype(int)
            
            # Cyclical encoding for better time representation
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Advanced time features
            df['quarter'] = df['datetime'].dt.quarter
            df['day_of_year'] = df['datetime'].dt.dayofyear
            df['week_of_year'] = df['datetime'].dt.isocalendar().week
            
            # Business hours and peak time features
            df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
            df['is_peak_hour'] = ((df['hour'].isin([9, 10, 11, 14, 15, 16, 17]))).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_morning_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
            df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
            
            # Weather-related features
            df['temp_squared'] = df['기온(°C)'] ** 2
            df['temp_cubed'] = df['기온(°C)'] ** 3
            df['humidity_squared'] = df['습도(%)'] ** 2
            
            # Comfort zone features
            df['temp_comfort'] = np.abs(df['기온(°C)'] - 22)  # Distance from 22°C
            df['temp_cooling_need'] = np.maximum(df['기온(°C)'] - 26, 0)
            df['temp_heating_need'] = np.maximum(18 - df['기온(°C)'], 0)
            
            # Weather interaction features
            df['temp_humidity'] = df['기온(°C)'] * df['습도(%)']
            df['temp_wind'] = df['기온(°C)'] * df['풍속(m/s)']
            df['humidity_rain'] = df['습도(%)'] * df['강수량(mm)']
            
            # Building area features - handle string values
            df['total_area'] = pd.to_numeric(df['연면적(m2)'], errors='coerce').fillna(1000)
            df['cooling_area'] = pd.to_numeric(df['냉방면적(m2)'], errors='coerce').fillna(1000)
            df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
            df['log_total_area'] = np.log1p(df['total_area'])
            df['log_cooling_area'] = np.log1p(df['cooling_area'])
            
            # Solar panel features - handle string values
            df['pv_capacity'] = pd.to_numeric(df['태양광용량(kW)'], errors='coerce').fillna(0)
            df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
            df['has_pv'] = (df['pv_capacity'] > 0).astype(int)
            
            # Building type features
            df['building_type'] = df['건물유형']
            
            return df
        
        # Apply feature engineering
        train_fe = create_features(train_df)
        test_fe = create_features(test_df)
        
        # Sort by building and time for lag features
        train_fe = train_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
        test_fe = test_fe.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
        
        print("Creating building-specific and lag features...")
        
        # Create building-specific features
        train_fe = self.create_building_features(train_fe)
        test_fe = self.create_building_features(test_fe, is_test=True)
        
        # Create lag features
        train_fe = self.create_lag_features(train_fe)
        test_fe = self.create_lag_features(test_fe, is_test=True)
        
        print(f"Feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
        return train_fe, test_fe
    
    def create_building_features(self, df, is_test=False):
        """Create building-specific statistical features."""
        df = df.copy()
        
        if not is_test:
            # Calculate building statistics from training data
            building_stats = {}
            
            for building_id in df['건물번호'].unique():
                building_data = df[df['건물번호'] == building_id]
                
                if '전력소비량(kWh)' in building_data.columns:
                    power_data = building_data['전력소비량(kWh)']
                    
                    stats = {
                        'mean_power': power_data.mean(),
                        'std_power': power_data.std(),
                        'min_power': power_data.min(),
                        'max_power': power_data.max(),
                        'median_power': power_data.median(),
                        'q25_power': power_data.quantile(0.25),
                        'q75_power': power_data.quantile(0.75)
                    }
                    
                    # Hour-specific statistics
                    hour_stats = building_data.groupby('hour')['전력소비량(kWh)'].agg(['mean', 'std']).reset_index()
                    hour_stats.columns = ['hour', 'hour_mean_power', 'hour_std_power']
                    stats['hour_stats'] = hour_stats
                    
                    # Weekday-specific statistics
                    weekday_stats = building_data.groupby('weekday')['전력소비량(kWh)'].agg(['mean', 'std']).reset_index()
                    weekday_stats.columns = ['weekday', 'weekday_mean_power', 'weekday_std_power']
                    stats['weekday_stats'] = weekday_stats
                    
                    building_stats[building_id] = stats
            
            self.building_stats = building_stats
        
        # Apply building statistics to dataframe
        for building_id in df['건물번호'].unique():
            if building_id in self.building_stats:
                stats = self.building_stats[building_id]
                mask = df['건물번호'] == building_id
                
                # Basic stats
                df.loc[mask, 'building_mean_power'] = stats['mean_power']
                df.loc[mask, 'building_std_power'] = stats['std_power']
                df.loc[mask, 'building_min_power'] = stats['min_power']
                df.loc[mask, 'building_max_power'] = stats['max_power']
                df.loc[mask, 'building_median_power'] = stats['median_power']
                
                # Hour-specific stats
                hour_stats = stats['hour_stats']
                df.loc[mask, 'hour_mean_power'] = df.loc[mask, 'hour'].map(
                    hour_stats.set_index('hour')['hour_mean_power']
                ).fillna(stats['mean_power'])
                
                df.loc[mask, 'hour_std_power'] = df.loc[mask, 'hour'].map(
                    hour_stats.set_index('hour')['hour_std_power']
                ).fillna(stats['std_power'])
                
                # Weekday-specific stats
                weekday_stats = stats['weekday_stats']
                df.loc[mask, 'weekday_mean_power'] = df.loc[mask, 'weekday'].map(
                    weekday_stats.set_index('weekday')['weekday_mean_power']
                ).fillna(stats['mean_power'])
                
                df.loc[mask, 'weekday_std_power'] = df.loc[mask, 'weekday'].map(
                    weekday_stats.set_index('weekday')['weekday_std_power']
                ).fillna(stats['std_power'])
                
        # Fill missing building stats with global means
        global_mean = df['building_mean_power'].mean() if 'building_mean_power' in df.columns else 1000
        building_feature_cols = [
            'building_mean_power', 'building_std_power', 'building_min_power',
            'building_max_power', 'building_median_power', 'hour_mean_power',
            'hour_std_power', 'weekday_mean_power', 'weekday_std_power'
        ]
        
        for col in building_feature_cols:
            if col not in df.columns:
                df[col] = global_mean
            df[col] = df[col].fillna(global_mean)
        
        return df
    
    def create_lag_features(self, df, is_test=False):
        """Create lag features for time series prediction."""
        df = df.copy()
        
        if '전력소비량(kWh)' not in df.columns and not is_test:
            return df
        
        # Create lag features for each building
        lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 1w
        
        for building_id in df['건물번호'].unique():
            mask = df['건물번호'] == building_id
            building_data = df[mask].copy()
            
            if '전력소비량(kWh)' in building_data.columns:
                power_series = building_data['전력소비량(kWh)']
                
                for lag in lag_periods:
                    df.loc[mask, f'power_lag_{lag}'] = power_series.shift(lag)
                    
                    # Rolling statistics
                    if lag <= 24:
                        df.loc[mask, f'power_rolling_mean_{lag}'] = power_series.rolling(lag).mean().shift(1)
                        df.loc[mask, f'power_rolling_std_{lag}'] = power_series.rolling(lag).std().shift(1)
                        df.loc[mask, f'power_rolling_max_{lag}'] = power_series.rolling(lag).max().shift(1)
                        df.loc[mask, f'power_rolling_min_{lag}'] = power_series.rolling(lag).min().shift(1)
            
            # Weather lag features
            weather_cols = ['기온(°C)', '습도(%)', '풍속(m/s)']
            for col in weather_cols:
                if col in building_data.columns:
                    series = building_data[col]
                    for lag in [1, 2, 3, 6]:
                        df.loc[mask, f'{col}_lag_{lag}'] = series.shift(lag)
        
        # Fill lag features with appropriate values
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        
        for col in lag_cols:
            if 'power' in col:
                # Fill with building mean or global mean
                df[col] = df.groupby('건물번호')[col].fillna(method='bfill')
                df[col] = df[col].fillna(df.get('building_mean_power', 1000))
            else:
                # Fill weather lags with forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=50):
        """Optimize XGBoost hyperparameters using Optuna."""
        print("Optimizing hyperparameters...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbosity': 0
            }
            
            # Use TimeSeriesCV for validation
            ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
            scores = []
            
            # Need to recreate the full dataframe for time series splits
            # For now, use regular KFold as approximation
            kfold = KFold(n_splits=3, shuffle=False)
            
            for train_idx, val_idx in kfold.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)
                
                y_pred = model.predict(X_val)
                score = smape(y_val.values, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best SMAPE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train_model(self, train_df):
        """Train the advanced model."""
        print("Training advanced model...")
        
        # Prepare features
        drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
        feature_cols = [c for c in train_df.columns if c not in drop_cols]
        
        X = train_df[feature_cols].copy()
        y = train_df['전력소비량(kWh)']
        
        # Handle all object/string columns
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
            elif not pd.api.types.is_numeric_dtype(X[col]):
                # Convert any non-numeric to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X, y, n_trials=30)
        
        # Train final model with best parameters
        model = xgb.XGBRegressor(**best_params)
        model.fit(X, y)
        
        self.model = model
        self.feature_cols = feature_cols
        self.encoders = encoders
        
        print("Model training completed!")
        return self
    
    def predict(self, test_df):
        """Make predictions on test data."""
        X_test = test_df[self.feature_cols].copy()
        
        # Apply the same encoding as training
        for col, encoder in self.encoders.items():
            if col in X_test.columns:
                # Handle unseen categories
                X_test[col] = X_test[col].astype(str)
                X_test[col] = X_test[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        
        predictions = self.model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        return predictions
    
    def create_submission(self, test_df, predictions, output_path):
        """Create submission file."""
        submission = pd.DataFrame({
            'num_date_time': test_df['num_date_time'],
            'prediction': predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
        return submission


def main():
    """Main training and prediction workflow."""
    print("=" * 80)
    print("ADVANCED POWER CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # Paths
    data_dir = Path('data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    # Initialize predictor
    predictor = AdvancedPowerPredictor()
    
    # Load data
    train_df, test_df = predictor.load_data(train_path, test_path, building_path)
    
    # Feature engineering
    train_fe, test_fe = predictor.advanced_feature_engineering(train_df, test_df)
    
    # Validate model with time series CV before training
    print("\n" + "=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    from ts_validation import TimeSeriesCV
    
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]
    
    X = train_fe[feature_cols].copy()
    y = train_fe['전력소비량(kWh)']
    
    # Handle all object/string columns
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        elif not pd.api.types.is_numeric_dtype(X[col]):
            # Convert any non-numeric to numeric
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    preprocessor = None  # No preprocessing needed
    
    # Quick validation with default parameters
    ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_fe, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Validation fold {fold + 1}/3")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Quick model for validation
        model = xgb.XGBRegressor(
            max_depth=8,
            n_estimators=300,
            learning_rate=0.1,
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
    
    # Train final model
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    predictor.train_model(train_fe)
    
    # Make predictions
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    predictions = predictor.predict(test_fe)
    
    # Create submission
    submission = predictor.create_submission(
        test_fe, 
        predictions, 
        'submission_advanced.csv'
    )
    
    print(f"\n✅ Advanced solution completed!")
    print(f"Validation SMAPE: {mean_smape:.4f}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: submission_advanced.csv")
    
    return {
        'validation_smape': mean_smape,
        'validation_std': std_smape,
        'predictions': predictions,
        'submission': submission
    }


if __name__ == "__main__":
    results = main()