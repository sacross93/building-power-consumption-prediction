#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최고 성능 모델 - 시차 변수 포함
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
    print("Best Model Training with Lag Features")
    print("=" * 50)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    y_train = train_df['power_transformed']
    
    print(f"Available train columns: {len(train_df.columns)}")
    print(f"Available test columns: {len(test_df.columns)}")
    
    # 모든 가능한 피처 사용 (target과 ID 제외)
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed']
    
    # Train 피처
    train_features = [col for col in train_df.columns if col not in exclude_cols]
    
    # Test에도 있는 피처만 선택
    common_features = [col for col in train_features if col in test_df.columns]
    
    print(f"Total train features: {len(train_features)}")
    print(f"Common features: {len(common_features)}")
    
    # 시차 변수 확인
    lag_features = [col for col in common_features if 'lag' in col or 'rolling' in col]
    print(f"Lag features found: {len(lag_features)}")
    print(f"Lag features: {lag_features}")
    
    X_train = train_df[common_features].fillna(0)
    X_test = test_df[common_features].fillna(0)
    
    print(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Validation split (마지막 20%)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    print(f"Split - Train: {X_tr.shape}, Val: {X_val.shape}")
    
    # LightGBM with optimized parameters
    print("Training LightGBM with lag features...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=150,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=2000
    )
    
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])
    
    # Validation 성능 확인
    y_pred = lgb_model.predict(X_val)
    y_val_orig = safe_exp_transform(y_val.values)
    y_pred_orig = safe_exp_transform(y_pred)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    val_smape = smape(y_val_orig, y_pred_orig)
    print(f"\nValidation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 6.0:
        print("*** TARGET ACHIEVED! SMAPE <= 6.0 ***")
    else:
        print(f"Need {val_smape - 6.0:.4f} more improvement to reach target")
    
    # 전체 데이터로 최종 모델 훈련
    print("Training final model on full data...")
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=150,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=3000
    )
    
    final_model.fit(X_train, y_train)
    
    # 테스트 예측
    test_pred_log = final_model.predict(X_test)
    test_pred = safe_exp_transform(test_pred_log)
    test_pred = np.maximum(test_pred, 0)
    
    print(f"Test predictions range: {test_pred.min():.2f} ~ {test_pred.max():.2f}")
    print(f"Test predictions mean: {test_pred.mean():.2f}")
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")
    
    # 피처 중요도 확인 (상위 15개)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']}: {row['importance']:.0f}")
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Validation SMAPE: {val_smape:.4f}")
    print(f"Target SMAPE: 6.0000")
    
    if val_smape <= 6.0:
        print("STATUS: *** SUCCESS! TARGET ACHIEVED! ***")
    else:
        print(f"STATUS: Need {val_smape - 6.0:.4f} more improvement")
    
    print("Submission file: submission.csv")
    print(f"Features used: {len(common_features)}")
    print(f"Lag features used: {len(lag_features)}")

if __name__ == "__main__":
    main()