#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단하고 빠른 모델링 - SMAPE 측정
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# SMAPE 계산 함수
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

def test_data_version(data_version):
    print(f"\n=== 테스트: {data_version} ===")
    
    # 데이터 로드
    train_path = f'../processed_data/train_processed_{data_version}.csv'
    test_path = f'../processed_data/test_processed_{data_version}.csv'
    features_path = f'../processed_data/feature_columns_{data_version}.txt'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 피처 준비
    available_features = [col for col in feature_columns if col in train_df.columns]
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['power_transformed']
    
    test_available_features = [col for col in available_features if col in test_df.columns]
    X_test = test_df[test_available_features].fillna(0)
    
    print(f"Features: {len(available_features)}")
    
    # Validation split (마지막 15%)
    split_idx = int(len(X_train) * 0.85)
    X_train_split = X_train.iloc[:split_idx]
    y_train_split = y_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_val_split = y_train.iloc[split_idx:]
    
    # 간단한 LightGBM 모델
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 100,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'random_state': 42
    }
    
    model = lgb.LGBMRegressor(**params, n_estimators=500)
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # 예측
    y_pred = model.predict(X_val_split)
    
    # SMAPE 계산 (log 역변환 후)
    y_val_original = safe_exp_transform(y_val_split.values)
    y_pred_original = safe_exp_transform(y_pred)
    y_pred_original = np.maximum(y_pred_original, 0)
    
    smape_score = smape(y_val_original, y_pred_original)
    
    print(f"Validation SMAPE: {smape_score:.4f}")
    
    # 최종 모델로 테스트 예측
    final_model = lgb.LGBMRegressor(**params, n_estimators=1000)
    final_model.fit(X_train, y_train)
    
    test_pred_log = final_model.predict(X_test)
    test_pred = safe_exp_transform(test_pred_log)
    test_pred = np.maximum(test_pred, 0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission_path = f'submission_{data_version}_simple.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"제출 파일 생성: {submission_path}")
    print(f"예측값 범위: {test_pred.min():.2f} ~ {test_pred.max():.2f}")
    print(f"예측값 평균: {test_pred.mean():.2f}")
    
    return smape_score

def main():
    print("간단한 모델링으로 빠른 테스트")
    print("=" * 50)
    
    results = {}
    
    # 세 가지 데이터 버전 테스트
    for data_version in ['none_log', 'iqr_log', 'building_percentile_log']:
        try:
            score = test_data_version(data_version)
            results[data_version] = score
        except Exception as e:
            print(f"{data_version} 실패: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("최종 결과")
    print("=" * 50)
    
    for version, score in results.items():
        print(f"{version}: SMAPE {score:.4f}")
        if score <= 6.0:
            print(f"  ✅ 목표 달성! (≤ 6.0)")
        else:
            print(f"  📈 목표까지 {score - 6.0:.4f} 더 개선 필요")
    
    if results:
        best_version = min(results, key=results.get)
        best_score = results[best_version]
        
        print(f"\n🏆 최고 성능: {best_version} (SMAPE: {best_score:.4f})")
        
        if best_score <= 6.0:
            print("🎉 목표 달성!")
        else:
            print(f"목표까지 {best_score - 6.0:.4f} 더 개선 필요")

if __name__ == "__main__":
    main()