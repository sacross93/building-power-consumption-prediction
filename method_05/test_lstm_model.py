"""
Test LSTM Model for Power Consumption Prediction
===============================================

This script tests our LSTM model and compares it with the XGBoost baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from lstm_model import PowerConsumptionLSTM, evaluate_lstm_with_cv
from ts_validation import smape, TimeSeriesCV
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering from our current solution
import sys
sys.path.append('.')


def load_data_for_lstm():
    """Load and prepare data for LSTM testing."""
    from solution import load_data, engineer_features
    
    print("Loading data for LSTM testing...")
    
    # Load raw data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    print(f"Data loaded: {len(train_fe)} training samples")
    
    return train_fe, test_fe


def compare_xgboost_vs_lstm(data: pd.DataFrame, 
                           target_col: str = '전력소비량(kWh)',
                           n_splits: int = 3) -> Dict[str, Any]:
    """
    Compare XGBoost and LSTM models using time series cross-validation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
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
    print("XGBOOST vs LSTM MODEL COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # 1. Test XGBoost (simplified version)
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
    
    # Filter available categorical columns
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
    
    # 2. Test LSTM Model
    print("\n2. Testing LSTM Model...")
    print("-" * 50)
    
    lstm_results = evaluate_lstm_with_cv(data, target_col, sequence_length=24, n_splits=n_splits)
    results['lstm'] = lstm_results
    
    # 3. Comparison Summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"XGBoost:  {xgb_mean:.4f} (±{xgb_std:.4f})")
    if lstm_results['fold_scores']:
        lstm_mean = lstm_results['mean_score']
        lstm_std = lstm_results['std_score']
        print(f"LSTM:     {lstm_mean:.4f} (±{lstm_std:.4f})")
        
        improvement = xgb_mean - lstm_mean
        if improvement > 0:
            print(f"✅ LSTM improves by {improvement:.4f} SMAPE points")
        else:
            print(f"❌ XGBoost better by {-improvement:.4f} SMAPE points")
        
        # Target achievement check
        target = 6.0
        print(f"\nTarget Achievement (SMAPE ≤ {target}%):")
        print(f"XGBoost: {'✅' if xgb_mean <= target else '❌'} ({xgb_mean:.2f}%)")
        print(f"LSTM:    {'✅' if lstm_mean <= target else '❌'} ({lstm_mean:.2f}%)")
    else:
        print("LSTM: Failed to produce results")
    
    return results


def test_single_lstm_model(data: pd.DataFrame,
                          target_col: str = '전력소비량(kWh)',
                          sequence_length: int = 24) -> Dict[str, Any]:
    """
    Test a single LSTM model with detailed analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_col : str
        Name of target column
    sequence_length : int
        Length of input sequences
        
    Returns
    -------
    Dict[str, Any]
        Test results
    """
    print("=" * 70)
    print("DETAILED LSTM MODEL TESTING")
    print("=" * 70)
    
    # Use simple time split for detailed analysis
    cutoff = data['datetime'].max() - pd.Timedelta(days=7)
    train_data = data[data['datetime'] < cutoff]
    val_data = data[data['datetime'] >= cutoff]
    
    print(f"Train period: {train_data['datetime'].min()} to {train_data['datetime'].max()}")
    print(f"Val period: {val_data['datetime'].min()} to {val_data['datetime'].max()}")
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize LSTM model
    lstm_model = PowerConsumptionLSTM(
        sequence_length=sequence_length,
        lstm_units=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=64
    )
    
    # Prepare sequences
    print("\nPreparing sequences...")
    train_sequences = lstm_model.prepare_sequences(train_data, target_col)
    val_sequences = lstm_model.prepare_sequences(val_data, target_col)
    
    if len(train_sequences['X']) == 0 or len(val_sequences['X']) == 0:
        print("❌ Insufficient data for sequence generation")
        return {'error': 'Insufficient data'}
    
    print(f"Train sequences: {len(train_sequences['X'])}")
    print(f"Val sequences: {len(val_sequences['X'])}")
    
    # Train model
    print("\nTraining LSTM model...")
    lstm_model.fit(
        train_sequences,
        val_sequences,
        epochs=100,
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = lstm_model.predict(val_sequences)
    y_true = val_sequences['y']
    
    # Calculate metrics
    lstm_smape = smape(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    print(f"\nLSTM Model Results:")
    print(f"SMAPE: {lstm_smape:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Plot training history
    output_dir = Path('.')
    lstm_model.plot_training_history(output_dir / 'lstm_training_history.png')
    
    # Analyze predictions
    results = {
        'smape': lstm_smape,
        'mae': mae,
        'rmse': rmse,
        'y_true': y_true,
        'y_pred': y_pred,
        'timestamps': val_sequences['timestamps'],
        'building_ids': val_sequences['building_ids']
    }
    
    # Create prediction analysis plots
    create_lstm_analysis_plots(results, output_dir)
    
    return results


def create_lstm_analysis_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Create analysis plots for LSTM results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # 1. Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('LSTM: Predictions vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('LSTM: Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time series plot (first 500 points)
    n_plot = min(500, len(y_true))
    x_axis = range(n_plot)
    axes[1, 0].plot(x_axis, y_true[:n_plot], label='Actual', alpha=0.7)
    axes[1, 0].plot(x_axis, y_pred[:n_plot], label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Power Consumption')
    axes[1, 0].set_title('LSTM: Time Series Comparison (First 500 points)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution
    errors = np.abs(y_pred - y_true)
    axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('LSTM: Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LSTM analysis plots saved to: {output_dir / 'lstm_analysis.png'}")


def main():
    """Main testing workflow."""
    print("=" * 80)
    print("LSTM MODEL TESTING FOR POWER CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # Load data
    train_fe, test_fe = load_data_for_lstm()
    
    # Test 1: Quick comparison
    print("\n" + "="*80)
    print("TEST 1: QUICK MODEL COMPARISON (3-fold CV)")
    print("="*80)
    
    comparison_results = compare_xgboost_vs_lstm(train_fe, n_splits=3)
    
    # Test 2: Detailed single model analysis
    print("\n" + "="*80)
    print("TEST 2: DETAILED LSTM ANALYSIS")
    print("="*80)
    
    lstm_results = test_single_lstm_model(train_fe, sequence_length=24)
    
    # Summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    
    if 'error' not in lstm_results:
        print(f"✅ LSTM Model Performance:")
        print(f"   - SMAPE: {lstm_results['smape']:.4f}")
        print(f"   - MAE: {lstm_results['mae']:.2f}")
        print(f"   - RMSE: {lstm_results['rmse']:.2f}")
        
        if lstm_results['smape'] <= 6.0:
            print(f"🎯 TARGET ACHIEVED! LSTM meets SMAPE ≤ 6% requirement")
        else:
            improvement_needed = lstm_results['smape'] - 6.0
            print(f"❌ Need {improvement_needed:.2f}% improvement to reach target")
    else:
        print("❌ LSTM testing failed")
    
    print(f"\n📊 Results saved to:")
    print(f"   - LSTM Analysis: lstm_analysis.png")
    print(f"   - Training History: lstm_training_history.png")
    
    return {'comparison': comparison_results, 'lstm_detailed': lstm_results}


if __name__ == "__main__":
    results = main()