#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 테스트 - 피처 정렬 문제 해결
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

def test_single_version(data_version='iqr_log'):
    print(f"테스트: {data_version}")
    
    # 데이터 로드
    train_df = pd.read_csv(f'../processed_data/train_processed_{data_version}.csv')
    test_df = pd.read_csv(f'../processed_data/test_processed_{data_version}.csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Train 컬럼: {list(train_df.columns)}")
    print(f"Test 컬럼: {list(test_df.columns)}")
    
    # 공통 피처 찾기
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 'power_transformed']
    
    train_features = [col for col in train_df.columns if col not in exclude_cols]
    test_features = [col for col in test_df.columns if col not in exclude_cols]
    
    # 공통 피처만 사용
    common_features = [col for col in train_features if col in test_features]
    
    print(f"Train 피처: {len(train_features)}")
    print(f"Test 피처: {len(test_features)}")
    print(f"공통 피처: {len(common_features)}")
    
    if len(common_features) == 0:
        print("공통 피처가 없습니다!")
        return None
    
    # 데이터 준비
    X_train = train_df[common_features].fillna(0)
    y_train = train_df['power_transformed']
    X_test = test_df[common_features].fillna(0)
    
    # Validation split
    split_idx = int(len(X_train) * 0.85)
    X_train_split = X_train.iloc[:split_idx]
    y_train_split = y_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_val_split = y_train.iloc[split_idx:]
    
    print(f"Train split: {X_train_split.shape}, Val split: {X_val_split.shape}")
    
    # 모델 훈련
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=100,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbosity=-1,
        random_state=42,
        n_estimators=300
    )
    
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Validation 예측
    y_pred = model.predict(X_val_split)
    
    # SMAPE 계산
    y_val_original = safe_exp_transform(y_val_split.values)
    y_pred_original = safe_exp_transform(y_pred)
    y_pred_original = np.maximum(y_pred_original, 0)
    
    smape_score = smape(y_val_original, y_pred_original)
    print(f"Validation SMAPE: {smape_score:.4f}")
    
    # 전체 데이터로 최종 모델 훈련
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
        n_estimators=500
    )
    
    final_model.fit(X_train, y_train)
    
    # 테스트 예측
    test_pred_log = final_model.predict(X_test)
    test_pred = safe_exp_transform(test_pred_log)
    test_pred = np.maximum(test_pred, 0)
    
    print(f"예측값 범위: {test_pred.min():.2f} ~ {test_pred.max():.2f}")
    print(f"예측값 평균: {test_pred.mean():.2f}")
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission_path = f'submission_{data_version}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"제출 파일 생성: {submission_path}")
    
    if smape_score <= 6.0:
        print("🎉 목표 달성! SMAPE ≤ 6.0")
    else:
        print(f"목표까지 {smape_score - 6.0:.4f} 더 개선 필요")
    
    return smape_score

if __name__ == "__main__":
    print("빠른 모델 테스트")
    print("=" * 40)
    
    # 가장 좋은 성능을 보인 iqr_log 버전만 테스트
    score = test_single_version('iqr_log')
    
    if score and score <= 6.0:
        print(f"\n✅ 최종 결과: SMAPE {score:.4f} - 목표 달성!")
    elif score:
        print(f"\n📊 최종 결과: SMAPE {score:.4f}")
        print(f"목표까지 {score - 6.0:.4f} 더 개선 필요")
    else:
        print("모델 테스트 실패")