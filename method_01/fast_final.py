#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 최종 모델 - 시차 변수 없이
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def smape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape_val

def safe_exp_transform(y_log):
    y_log_clipped = np.clip(y_log, -10, 10)
    return np.expm1(y_log_clipped)

def main():
    print("Fast Final Model - No Lag Features")
    print("=" * 40)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    y_train = train_df['power_transformed']
    
    # Test에 있는 피처만 사용
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed',
                   'power_lag_1h', 'power_lag_24h', 'power_lag_168h',
                   'power_rolling_mean_24h', 'power_rolling_std_24h', 'power_rolling_mean_7d']
    
    available_features = []
    for col in train_df.columns:
        if col not in exclude_cols and col in test_df.columns:
            available_features.append(col)
    
    X_train = train_df[available_features].fillna(0)
    X_test = test_df[available_features].fillna(0)
    
    print(f"Features: {len(available_features)}")
    print(f"Shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Validation split
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    # LightGBM with optimized parameters
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=200,
        learning_rate=0.08,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=15,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=800
    )
    
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    # Validation
    y_pred = lgb_model.predict(X_val)
    y_val_orig = safe_exp_transform(y_val.values)
    y_pred_orig = safe_exp_transform(y_pred)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    val_smape = smape(y_val_orig, y_pred_orig)
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 6.0:
        print("*** TARGET ACHIEVED! ***")
    else:
        print(f"Need {val_smape - 6.0:.4f} more")
    
    # Final model
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=200,
        learning_rate=0.08,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=15,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=1200
    )
    
    final_model.fit(X_train, y_train)
    
    # Test prediction
    test_pred_log = final_model.predict(X_test)
    test_pred = safe_exp_transform(test_pred_log)
    test_pred = np.maximum(test_pred, 0)
    
    # Submission
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"Final SMAPE: {val_smape:.4f}")
    print(f"Test range: {test_pred.min():.2f} ~ {test_pred.max():.2f}")
    print(f"Test mean: {test_pred.mean():.2f}")
    print("Submission: submission.csv")
    
    if val_smape <= 6.0:
        print("SUCCESS!")
    else:
        print(f"Need {val_smape - 6.0:.4f} improvement")

if __name__ == "__main__":
    main()