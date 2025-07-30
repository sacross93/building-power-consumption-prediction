"""
Prophet Time Series Model for Power Consumption Prediction
=========================================================

This module implements Prophet models for automatic seasonality detection
and time series forecasting of power consumption data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Prophet imports
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
    print("Prophet is available")
except ImportError:
    print("Prophet not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'prophet'])
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True

from ts_validation import smape, TimeSeriesCV


class PowerConsumptionProphet:
    """Prophet model for power consumption prediction."""
    
    def __init__(self, 
                 seasonality_mode: str = 'multiplicative',
                 yearly_seasonality: bool = False,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 add_country_holidays: bool = False,
                 interval_width: float = 0.8):
        """
        Initialize Prophet model.
        
        Parameters
        ----------
        seasonality_mode : str
            'additive' or 'multiplicative'
        yearly_seasonality : bool
            Include yearly seasonality
        weekly_seasonality : bool
            Include weekly seasonality
        daily_seasonality : bool
            Include daily seasonality
        add_country_holidays : bool
            Include country holidays
        interval_width : float
            Uncertainty interval width
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.add_country_holidays = add_country_holidays
        self.interval_width = interval_width
        
        # Store models for each building
        self.models = {}
        self.building_data = {}
        
    def prepare_prophet_data(self, 
                           data: pd.DataFrame,
                           target_col: str = '전력소비량(kWh)',
                           building_col: str = '건물번호',
                           datetime_col: str = 'datetime') -> Dict[str, pd.DataFrame]:
        """
        Prepare data for Prophet training.
        
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
        Dict[str, pd.DataFrame]
            Dictionary with building data in Prophet format
        """
        print("Preparing data for Prophet...")
        
        building_datasets = {}
        
        for building_id in data[building_col].unique():
            building_data = data[data[building_col] == building_id].copy()
            
            if len(building_data) < 48:  # At least 2 days of data
                continue
            
            # Sort by datetime
            building_data = building_data.sort_values(datetime_col).reset_index(drop=True)
            
            # Prepare Prophet format (ds, y)
            prophet_df = pd.DataFrame({
                'ds': building_data[datetime_col],
                'y': building_data[target_col]
            })
            
            # Add regressors (external features)
            regressor_cols = ['temp', 'humidity', 'hour', 'weekday', 'month', 'is_weekend']
            available_regressors = [col for col in regressor_cols if col in building_data.columns]
            
            for col in available_regressors:
                prophet_df[col] = building_data[col].values
            
            building_datasets[building_id] = {
                'data': prophet_df,
                'regressors': available_regressors
            }
        
        print(f"Prepared data for {len(building_datasets)} buildings")
        return building_datasets
    
    def fit_building_model(self, 
                          building_id: int,
                          train_data: pd.DataFrame,
                          regressors: List[str]) -> Prophet:
        """
        Fit Prophet model for a single building.
        
        Parameters
        ----------
        building_id : int
            Building identifier
        train_data : pd.DataFrame
            Training data in Prophet format
        regressors : List[str]
            List of external regressors
            
        Returns
        -------
        Prophet
            Fitted Prophet model
        """
        # Initialize Prophet model
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )
        
        # Add external regressors
        for regressor in regressors:
            model.add_regressor(regressor)
        
        # Add custom seasonalities
        if len(train_data) >= 24 * 7:  # At least a week of hourly data
            model.add_seasonality(name='hourly', period=24, fourier_order=8)
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_data)
        
        return model
    
    def fit(self, building_datasets: Dict[str, Dict]) -> None:
        """
        Fit Prophet models for all buildings.
        
        Parameters
        ----------
        building_datasets : Dict[str, Dict]
            Building datasets from prepare_prophet_data
        """
        print("Training Prophet models for all buildings...")
        
        self.models = {}
        self.building_data = {}
        
        for building_id, dataset in building_datasets.items():
            try:
                train_data = dataset['data']
                regressors = dataset['regressors']
                
                print(f"Training building {building_id}: {len(train_data)} samples")
                
                model = self.fit_building_model(building_id, train_data, regressors)
                
                self.models[building_id] = model
                self.building_data[building_id] = {
                    'regressors': regressors,
                    'last_ds': train_data['ds'].max()
                }
                
            except Exception as e:
                print(f"Failed to train building {building_id}: {e}")
                continue
        
        print(f"Successfully trained {len(self.models)} building models")
    
    def predict_building(self, 
                        building_id: int,
                        future_data: pd.DataFrame) -> np.ndarray:
        """
        Predict for a single building.
        
        Parameters
        ----------
        building_id : int
            Building identifier
        future_data : pd.DataFrame
            Future data in Prophet format
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if building_id not in self.models:
            # Return mean prediction if model not available
            return np.full(len(future_data), future_data['y'].mean() if 'y' in future_data.columns else 100)
        
        model = self.models[building_id]
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future_data)
        
        predictions = forecast['yhat'].values
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict(self, building_datasets: Dict[str, Dict]) -> Dict[int, np.ndarray]:
        """
        Make predictions for all buildings.
        
        Parameters
        ----------
        building_datasets : Dict[str, Dict]
            Building datasets for prediction
            
        Returns
        -------
        Dict[int, np.ndarray]
            Predictions for each building
        """
        predictions = {}
        
        for building_id, dataset in building_datasets.items():
            future_data = dataset['data']
            predictions[building_id] = self.predict_building(building_id, future_data)
        
        return predictions


def test_prophet_model(data: pd.DataFrame,
                      target_col: str = '전력소비량(kWh)',
                      building_col: str = '건물번호',
                      datetime_col: str = 'datetime') -> Dict[str, Any]:
    """
    Test Prophet model with simple time split.
    
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
    Dict[str, Any]
        Test results
    """
    print("=" * 60)
    print("PROPHET MODEL TEST")
    print("=" * 60)
    
    # Simple time split
    cutoff = data[datetime_col].max() - pd.Timedelta(days=7)
    train_data = data[data[datetime_col] < cutoff]
    val_data = data[data[datetime_col] >= cutoff]
    
    print(f"Train period: {train_data[datetime_col].min()} to {train_data[datetime_col].max()}")
    print(f"Val period: {val_data[datetime_col].min()} to {val_data[datetime_col].max()}")
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize Prophet model
    prophet_model = PowerConsumptionProphet(
        seasonality_mode='multiplicative',
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    # Prepare data
    train_datasets = prophet_model.prepare_prophet_data(train_data, target_col, building_col, datetime_col)
    val_datasets = prophet_model.prepare_prophet_data(val_data, target_col, building_col, datetime_col)
    
    if len(train_datasets) == 0:
        print("❌ No training data available")
        return {'error': 'No training data'}
    
    # Train models
    prophet_model.fit(train_datasets)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = prophet_model.predict(val_datasets)
    
    # Collect results
    y_true_all = []
    y_pred_all = []
    building_ids = []
    
    for building_id in val_datasets.keys():
        if building_id in predictions:
            val_y = val_datasets[building_id]['data']['y'].values
            pred_y = predictions[building_id]
            
            # Ensure same length
            min_len = min(len(val_y), len(pred_y))
            val_y = val_y[:min_len]
            pred_y = pred_y[:min_len]
            
            y_true_all.extend(val_y)
            y_pred_all.extend(pred_y)
            building_ids.extend([building_id] * min_len)
    
    if len(y_true_all) == 0:
        print("❌ No predictions generated")
        return {'error': 'No predictions'}
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Calculate metrics
    prophet_smape = smape(y_true_all, y_pred_all)
    mae = np.mean(np.abs(y_true_all - y_pred_all))
    rmse = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    
    print(f"\nProphet Model Results:")
    print(f"SMAPE: {prophet_smape:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    results = {
        'smape': prophet_smape,
        'mae': mae,
        'rmse': rmse,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'building_ids': building_ids,
        'n_buildings_trained': len(prophet_model.models),
        'n_buildings_predicted': len(predictions)
    }
    
    return results


def compare_prophet_vs_xgboost(data: pd.DataFrame,
                              target_col: str = '전력소비량(kWh)',
                              n_splits: int = 3) -> Dict[str, Any]:
    """
    Compare Prophet and XGBoost models using time series cross-validation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of target column
    n_splits : int
        Number of CV splits
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    print("=" * 70)
    print("PROPHET vs XGBOOST MODEL COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # Test XGBoost first (simplified)
    print("\n1. Testing XGBoost Model...")
    print("-" * 50)
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    
    # Prepare XGBoost data
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    categorical_cols = ['건물번호', 'building_type']
    categorical_cols = [col for col in categorical_cols if col in feature_cols]
    
    ts_cv = TimeSeriesCV(n_splits=n_splits, test_size_days=7, gap_days=1)
    splits = ts_cv.split(data, 'datetime')
    
    xgb_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"XGBoost Fold {fold + 1}/{len(splits)}")
        
        # Prepare data
        X = data[feature_cols].copy()
        y = data[target_col]
        
        # Convert categorical features
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
        
        # Train model
        model = xgb.XGBRegressor(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        
        pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_val)
        fold_smape = smape(y_val.values, y_pred)
        xgb_scores.append(fold_smape)
        
        print(f"  XGBoost Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    xgb_mean = np.mean(xgb_scores)
    xgb_std = np.std(xgb_scores)
    
    results['xgboost'] = {
        'scores': xgb_scores,
        'mean': xgb_mean,
        'std': xgb_std
    }
    
    print(f"\nXGBoost Results:")
    print(f"Mean SMAPE: {xgb_mean:.4f} (±{xgb_std:.4f})")
    
    # Test Prophet with simple split (too slow for full CV)
    print("\n2. Testing Prophet Model...")
    print("-" * 50)
    
    prophet_results = test_prophet_model(data, target_col)
    
    if 'error' not in prophet_results:
        prophet_smape = prophet_results['smape']
        results['prophet'] = {
            'smape': prophet_smape,
            'mae': prophet_results['mae'],
            'rmse': prophet_results['rmse'],
            'n_buildings': prophet_results['n_buildings_trained']
        }
        
        print(f"Prophet SMAPE: {prophet_smape:.4f}")
        
        # Comparison
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        print(f"XGBoost:  {xgb_mean:.4f} (±{xgb_std:.4f})")
        print(f"Prophet:  {prophet_smape:.4f}")
        
        improvement = xgb_mean - prophet_smape
        if improvement > 0:
            print(f"✅ Prophet improves by {improvement:.4f} SMAPE points")
        else:
            print(f"❌ XGBoost better by {-improvement:.4f} SMAPE points")
        
        # Target achievement check
        target = 6.0
        print(f"\nTarget Achievement (SMAPE ≤ {target}%):") 
        print(f"XGBoost: {'✅' if xgb_mean <= target else '❌'} ({xgb_mean:.2f}%)")
        print(f"Prophet:  {'✅' if prophet_smape <= target else '❌'} ({prophet_smape:.2f}%)")
        
    else:
        print("❌ Prophet testing failed")
        results['prophet'] = {'error': prophet_results['error']}
    
    return results


if __name__ == "__main__":
    # Test Prophet model functionality
    from solution import load_data, engineer_features
    
    # Load data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # Run comparison
    results = compare_prophet_vs_xgboost(train_fe, n_splits=3)