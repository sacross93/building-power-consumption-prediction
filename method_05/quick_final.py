"""
Quick Final Solution for Power Consumption Prediction
===================================================

Simplified but optimized version for quick execution and submission generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape


def quick_validation():
    """Quick validation of the current approach."""
    print("=" * 60)
    print("QUICK VALIDATION CHECK")
    print("=" * 60)
    
    # Load data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # Simple time split for quick validation
    cutoff = train_fe['datetime'].max() - pd.Timedelta(days=7)
    train_data = train_fe[train_fe['datetime'] < cutoff]
    val_data = train_fe[train_fe['datetime'] >= cutoff]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Prepare features
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_data.columns if c not in drop_cols]
    
    X_train = train_data[feature_cols].copy()
    y_train = train_data['전력소비량(kWh)']
    X_val = val_data[feature_cols].copy()
    y_val = val_data['전력소비량(kWh)']
    
    # Handle categorical columns
    categorical_cols = ['건물번호', 'building_type']
    encoders = {}
    
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_val[col] = X_val[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            encoders[col] = le
    
    # Handle any remaining object columns
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_val[col] = X_val[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            encoders[col] = le
    
    # Train model
    model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=300,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    validation_smape = smape(y_val.values, y_pred)
    
    print(f"Validation SMAPE: {validation_smape:.4f}")
    
    if validation_smape <= 6.0:
        print(f"🎯 Model meets target (SMAPE ≤ 6%)!")
    else:
        improvement_needed = validation_smape - 6.0
        print(f"❌ Need {improvement_needed:.2f}% improvement to reach target")
    
    return validation_smape, model, feature_cols, encoders, train_fe, test_fe


def generate_final_submission():
    """Generate final submission with best available model."""
    print("=" * 60)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 60)
    
    # Get validation results and trained model
    val_smape, model, feature_cols, encoders, train_fe, test_fe = quick_validation()
    
    print("\nTraining final model on all data...")
    
    # Prepare full training data
    X_full = train_fe[feature_cols].copy()
    y_full = train_fe['전력소비량(kWh)']
    
    # Apply same encoding
    for col, encoder in encoders.items():
        if col in X_full.columns:
            X_full[col] = encoder.fit_transform(X_full[col].astype(str))
    
    # Handle any remaining object columns
    for col in X_full.columns:
        if X_full[col].dtype == 'object':
            le = LabelEncoder()
            X_full[col] = le.fit_transform(X_full[col].astype(str))
            encoders[col] = le
    
    # Train final model
    final_model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=500,  # More estimators for final model
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    
    final_model.fit(X_full, y_full)
    
    # Prepare test data
    X_test = test_fe[feature_cols].copy()
    
    # Apply same encoding to test data
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
    # Handle any remaining object columns
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            # For unseen categories, use mode or default encoding
            X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))
    
    # Make predictions
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    # Create submission
    submission = pd.DataFrame({
        'num_date_time': test_fe['num_date_time'],
        'prediction': predictions
    })
    
    submission.to_csv('submission_final_quick.csv', index=False)
    
    print(f"\n✅ Final submission completed!")
    print(f"Validation SMAPE: {val_smape:.4f}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: submission_final_quick.csv")
    
    return submission, val_smape


if __name__ == "__main__":
    submission, smape_score = generate_final_submission()
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Validation SMAPE: {smape_score:.4f}")
    
    if smape_score <= 6.0:
        print("🎯 TARGET ACHIEVED: SMAPE ≤ 6%")
    else:
        gap = smape_score - 6.0
        print(f"❌ Target missed by {gap:.2f}%")
        
        print("\nPossible improvements:")
        print("- More sophisticated feature engineering")
        print("- Ensemble of multiple models")
        print("- Building-specific models")
        print("- Advanced hyperparameter tuning")
    
    print(f"\nSubmission file: submission_final_quick.csv")
    print(f"Ready for submission!")