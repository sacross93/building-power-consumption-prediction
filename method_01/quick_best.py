#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 최고 성능 모델
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
    print("Quick Best Model Training")
    print("=" * 40)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    y_train = train_df['power_transformed']
    
    # 피처 선택
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed']
    
    train_features = [col for col in train_df.columns if col not in exclude_cols]
    common_features = [col for col in train_features if col in test_df.columns]
    
    X_train = train_df[common_features].fillna(0)
    X_test = test_df[common_features].fillna(0)
    
    print(f"Features: {len(common_features)}")
    
    # Validation split
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    # 빠른 LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=100,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbosity=-1,
        random_state=42,
        n_estimators=500
    )
    
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    # Validation 성능
    y_pred = lgb_model.predict(X_val)
    y_val_orig = safe_exp_transform(y_val.values)
    y_pred_orig = safe_exp_transform(y_pred)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    val_smape = smape(y_val_orig, y_pred_orig)
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 6.0:
        print("TARGET ACHIEVED!")
    else:
        print(f"Need {val_smape - 6.0:.4f} more")
    
    # 최종 모델
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=100,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbosity=-1,
        random_state=42,
        n_estimators=800
    )
    
    final_model.fit(X_train, y_train)
    
    # 테스트 예측
    test_pred_log = final_model.predict(X_test)
    test_pred = safe_exp_transform(test_pred_log)
    test_pred = np.maximum(test_pred, 0)
    
    # 제출 파일
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"Final SMAPE: {val_smape:.4f}")
    print(f"Target: 6.0")
    print("Submission saved: submission.csv")
    
    if val_smape <= 6.0:
        print("*** SUCCESS! ***")

if __name__ == "__main__":
    main()