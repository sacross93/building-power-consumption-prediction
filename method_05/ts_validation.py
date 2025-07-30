"""
Time Series Cross-Validation for Power Consumption Prediction
=============================================================

This module implements proper time series validation strategies to get
reliable performance estimates before making predictions on test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    smape_values = np.zeros_like(numerator, dtype=float)
    smape_values[mask] = numerator[mask] / denominator[mask]
    return 100.0 * np.mean(smape_values)


class TimeSeriesCV:
    """Time Series Cross-Validation for power consumption data."""
    
    def __init__(self, n_splits: int = 5, test_size_days: int = 7, gap_days: int = 0):
        """
        Initialize TimeSeriesCV.
        
        Parameters
        ----------
        n_splits : int
            Number of validation splits
        test_size_days : int
            Size of each validation set in days
        gap_days : int
            Gap between train and validation sets
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        
    def split(self, data: pd.DataFrame, datetime_col: str = 'datetime') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series cross-validation splits.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with datetime column
        datetime_col : str
            Name of datetime column
            
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_idx, val_idx) tuples
        """
        data = data.sort_values(datetime_col).reset_index(drop=True)
        
        splits = []
        test_size_hours = self.test_size_days * 24
        gap_hours = self.gap_days * 24
        
        # Get total data timespan
        start_date = data[datetime_col].min()
        end_date = data[datetime_col].max()
        total_hours = int((end_date - start_date).total_seconds() / 3600) + 1
        
        # Calculate minimum training hours needed
        min_train_hours = 24 * 30  # At least 30 days for training
        
        # Calculate step size for walk-forward validation
        available_hours = total_hours - min_train_hours
        step_hours = (available_hours - test_size_hours) // (self.n_splits - 1) if self.n_splits > 1 else available_hours
        step_hours = max(24, step_hours)  # At least 1 day step
        
        for i in range(self.n_splits):
            # Calculate validation end (working backwards from the end)
            val_end = end_date - pd.Timedelta(hours=i * step_hours)
            val_start = val_end - pd.Timedelta(hours=test_size_hours - 1)
            
            # Calculate training end (before gap)
            train_end = val_start - pd.Timedelta(hours=gap_hours + 1) if gap_hours > 0 else val_start - pd.Timedelta(hours=1)
            
            # Skip if not enough training data
            if train_end <= start_date:
                continue
            
            # Get indices
            train_mask = (data[datetime_col] >= start_date) & (data[datetime_col] <= train_end)
            val_mask = (data[datetime_col] >= val_start) & (data[datetime_col] <= val_end)
            
            train_idx = data[train_mask].index.values
            val_idx = data[val_mask].index.values
            
            if len(train_idx) < 1000 or len(val_idx) < 100:  # Minimum data requirements
                continue
                
            splits.append((train_idx, val_idx))
            
        return splits
    
    def validate_model(self, 
                      data: pd.DataFrame, 
                      target_col: str,
                      feature_cols: List[str],
                      categorical_cols: List[str],
                      model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on the given model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        target_col : str
            Name of target column
        feature_cols : List[str]
            List of feature column names
        categorical_cols : List[str]
            List of categorical column names
        model_params : Dict[str, Any]
            XGBoost model parameters
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        if model_params is None:
            model_params = {
                'max_depth': 8,
                'n_estimators': 400,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': 42
            }
        
        # Prepare data
        X = data[feature_cols].copy()
        y = data[target_col]
        
        # Convert categorical features
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
        
        # Get splits
        splits = self.split(data, 'datetime')
        
        results = {
            'fold_scores': [],
            'fold_details': [],
            'mean_score': 0,
            'std_score': 0,
            'feature_importance': None
        }
        
        feature_importances = []
        
        print(f"Running {len(splits)}-fold Time Series Cross-Validation...")
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold + 1}/{len(splits)}")
            
            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**model_params)
            pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
            
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_val)
            
            # Calculate SMAPE
            fold_smape = smape(y_val.values, y_pred)
            results['fold_scores'].append(fold_smape)
            
            # Store fold details
            fold_detail = {
                'fold': fold + 1,
                'smape': fold_smape,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_period': (data.iloc[train_idx]['datetime'].min(), data.iloc[train_idx]['datetime'].max()),
                'val_period': (data.iloc[val_idx]['datetime'].min(), data.iloc[val_idx]['datetime'].max())
            }
            results['fold_details'].append(fold_detail)
            
            # Store feature importance
            feature_importances.append(pipeline.named_steps['model'].feature_importances_)
            
            print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
            print(f"  Train period: {fold_detail['train_period'][0]} to {fold_detail['train_period'][1]}")
            print(f"  Validation period: {fold_detail['val_period'][0]} to {fold_detail['val_period'][1]}")
            print()
        
        # Calculate summary statistics
        results['mean_score'] = np.mean(results['fold_scores'])
        results['std_score'] = np.std(results['fold_scores'])
        
        # Calculate feature importance (handle different array sizes safely)
        if feature_importances:
            # Find the minimum length to handle different preprocessor output sizes
            min_length = min(len(imp) for imp in feature_importances)
            if min_length > 0:
                # Truncate all arrays to the same length and compute mean
                truncated_importances = [imp[:min_length] for imp in feature_importances]
                results['feature_importance'] = np.mean(truncated_importances, axis=0)
            else:
                results['feature_importance'] = None
        else:
            results['feature_importance'] = None
        
        print(f"Cross-Validation Results:")
        print(f"Mean SMAPE: {results['mean_score']:.4f} (±{results['std_score']:.4f})")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in results['fold_scores']]}")
        
        return results


def compare_validation_strategies(data: pd.DataFrame, 
                                target_col: str,
                                feature_cols: List[str],
                                categorical_cols: List[str]) -> Dict[str, Any]:
    """
    Compare different validation strategies.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_col : str
        Name of target column
    feature_cols : List[str]
        List of feature column names
    categorical_cols : List[str]
        List of categorical column names
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    model_params = {
        'max_depth': 8,
        'n_estimators': 200,  # Reduced for faster comparison
        'learning_rate': 0.08,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    results = {}
    
    print("=" * 60)
    print("VALIDATION STRATEGY COMPARISON")
    print("=" * 60)
    
    # 1. Simple Time Split (current method)
    print("\n1. Simple Time Split (Last 7 days as validation)")
    print("-" * 50)
    
    cutoff = data['datetime'].max() - pd.Timedelta(days=7)
    train_mask = data['datetime'] < cutoff
    val_mask = ~train_mask
    
    X = data[feature_cols].copy()
    y = data[target_col]
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    
    model = xgb.XGBRegressor(**model_params)
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    simple_smape = smape(y_val.values, y_pred)
    
    results['simple_split'] = {
        'smape': simple_smape,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }
    
    print(f"Simple Split SMAPE: {simple_smape:.4f}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # 2. Time Series Cross-Validation
    print("\n2. Time Series Cross-Validation (5-fold)")
    print("-" * 50)
    
    ts_cv = TimeSeriesCV(n_splits=5, test_size_days=7, gap_days=0)
    cv_results = ts_cv.validate_model(data, target_col, feature_cols, categorical_cols, model_params)
    results['ts_cv'] = cv_results
    
    # 3. Time Series CV with Gap
    print("\n3. Time Series Cross-Validation with Gap (5-fold, 1-day gap)")
    print("-" * 50)
    
    ts_cv_gap = TimeSeriesCV(n_splits=5, test_size_days=7, gap_days=1)
    cv_gap_results = ts_cv_gap.validate_model(data, target_col, feature_cols, categorical_cols, model_params)
    results['ts_cv_gap'] = cv_gap_results
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Simple Split:        {results['simple_split']['smape']:.4f}")
    print(f"TS CV (5-fold):      {results['ts_cv']['mean_score']:.4f} (±{results['ts_cv']['std_score']:.4f})")
    print(f"TS CV with Gap:      {results['ts_cv_gap']['mean_score']:.4f} (±{results['ts_cv_gap']['std_score']:.4f})")
    
    return results


def analyze_building_specific_performance(data: pd.DataFrame,
                                        target_col: str,
                                        feature_cols: List[str],
                                        categorical_cols: List[str]) -> Dict[str, Any]:
    """
    Analyze performance by building type and individual buildings.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_col : str
        Name of target column
    feature_cols : List[str]
        List of feature column names
    categorical_cols : List[str]
        List of categorical column names
        
    Returns
    -------
    Dict[str, Any]
        Building-specific analysis results
    """
    print("\n" + "=" * 60)
    print("BUILDING-SPECIFIC PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Use simple split for faster analysis
    cutoff = data['datetime'].max() - pd.Timedelta(days=7)
    train_mask = data['datetime'] < cutoff
    val_mask = ~train_mask
    
    X = data[feature_cols].copy()
    y = data[target_col]
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    
    # Train model
    model_params = {
        'max_depth': 8,
        'n_estimators': 200,
        'learning_rate': 0.08,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**model_params)
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = pipeline.predict(X_val)
    
    # Get validation data with predictions
    val_data = data.loc[val_mask].copy()
    val_data['predicted'] = y_pred
    val_data['actual'] = y_val.values
    val_data['abs_error'] = np.abs(val_data['predicted'] - val_data['actual'])
    val_data['smape_contrib'] = (val_data['abs_error'] / 
                                ((np.abs(val_data['actual']) + np.abs(val_data['predicted'])) / 2.0)) * 100
    
    results = {}
    
    # 1. Performance by building type
    print("\n1. Performance by Building Type:")
    print("-" * 40)
    
    building_type_performance = val_data.groupby('building_type').agg({
        'smape_contrib': 'mean',
        'abs_error': 'mean',
        'actual': ['count', 'mean', 'std'],
        'predicted': 'mean'
    }).round(4)
    
    building_type_performance.columns = ['SMAPE', 'MAE', 'Count', 'Actual_Mean', 'Actual_Std', 'Pred_Mean']
    results['building_type'] = building_type_performance
    
    print(building_type_performance)
    
    # 2. Top 10 worst performing buildings
    print("\n2. Top 10 Worst Performing Buildings:")
    print("-" * 40)
    
    building_performance = val_data.groupby('건물번호').agg({
        'smape_contrib': 'mean',
        'abs_error': 'mean',
        'actual': ['count', 'mean'],
        'building_type': 'first'
    }).round(4)
    
    building_performance.columns = ['SMAPE', 'MAE', 'Count', 'Actual_Mean', 'Building_Type']
    worst_buildings = building_performance.sort_values('SMAPE', ascending=False).head(10)
    results['worst_buildings'] = worst_buildings
    
    print(worst_buildings)
    
    # 3. Top 10 best performing buildings
    print("\n3. Top 10 Best Performing Buildings:")
    print("-" * 40)
    
    best_buildings = building_performance.sort_values('SMAPE', ascending=True).head(10)
    results['best_buildings'] = best_buildings
    
    print(best_buildings)
    
    return results


if __name__ == "__main__":
    # This will be called from the main validation script
    pass