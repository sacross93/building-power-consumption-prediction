#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 간단 모델 - 유니코드 오류 없이
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
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
    print("Final Model Training")
    print("=" * 50)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    y_train = train_df['power_transformed']
    
    # 중요한 피처들만 선택
    important_features = [
        'hour', 'weekday', 'month', 'is_weekend', 'is_holiday',
        '기온(°C)', '습도(%)', '풍속(m/s)', '연면적(m2)', '냉방면적(m2)',
        'discomfort_index', 'apparent_temp', 'cooling_degree_days', 'heating_degree_days',
        'cooling_area_ratio', 'building_type_frequency'
    ]
    
    # 건물 유형 더미 변수들
    building_type_features = [col for col in train_df.columns if col.startswith('건물유형_')]
    
    all_features = important_features + building_type_features
    
    # 실제 존재하는 피처만 선택
    available_features = [col for col in all_features if col in train_df.columns]
    
    X_train = train_df[available_features].fillna(0)
    
    # Test 데이터도 동일한 피처로
    test_available_features = [col for col in available_features if col in test_df.columns]
    X_test = test_df[test_available_features].fillna(0)
    
    print(f"Train features: {X_train.shape[1]}, Test features: {X_test.shape[1]}")
    
    # Validation split (마지막 20%)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
    
    # LightGBM (최고 성능을 보인 모델)
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=120,
        learning_rate=0.08,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=2,
        lambda_l2=2,
        verbosity=-1,
        random_state=42,
        n_estimators=1000
    )
    
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    # Validation 성능 확인
    y_pred = lgb_model.predict(X_val)
    y_val_orig = safe_exp_transform(y_val.values)
    y_pred_orig = safe_exp_transform(y_pred)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    val_smape = smape(y_val_orig, y_pred_orig)
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 6.0:
        print("TARGET ACHIEVED! SMAPE <= 6.0")
    else:
        print(f"Need {val_smape - 6.0:.4f} more improvement to reach target")
    
    # 전체 데이터로 최종 모델 훈련
    print("Training final model on full data...")
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=120,
        learning_rate=0.08,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=2,
        lambda_l2=2,
        verbosity=-1,
        random_state=42,
        n_estimators=1500
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
    
    # 피처 중요도 확인
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    print(f"\nFinal Results:")
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 6.0:
        print("SUCCESS! Target achieved!")
    else:
        print(f"Close to target! Need {val_smape - 6.0:.4f} more improvement")
    
    print("Final submission file ready: submission.csv")

if __name__ == "__main__":
    main()