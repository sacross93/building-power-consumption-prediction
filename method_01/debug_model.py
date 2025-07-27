#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 디버깅 및 성능 확인
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

def analyze_data():
    print("데이터 분석 시작")
    
    # 원본 데이터 확인
    train_raw = pd.read_csv('../data/train.csv', encoding='utf-8-sig')
    test_raw = pd.read_csv('../data/test.csv', encoding='utf-8-sig')
    
    print(f"원본 Train: {train_raw.shape}")
    print(f"원본 Test: {test_raw.shape}")
    print(f"원본 Train 전력소비량 범위: {train_raw['전력소비량(kWh)'].min():.2f} ~ {train_raw['전력소비량(kWh)'].max():.2f}")
    print(f"원본 Train 전력소비량 평균: {train_raw['전력소비량(kWh)'].mean():.2f}")
    
    # 전처리된 데이터 확인
    train_processed = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_processed = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    print(f"전처리된 Train: {train_processed.shape}")
    print(f"전처리된 Test: {test_processed.shape}")
    
    if 'power_transformed' in train_processed.columns:
        print(f"Log 변환된 타겟 범위: {train_processed['power_transformed'].min():.2f} ~ {train_processed['power_transformed'].max():.2f}")
        print(f"Log 변환된 타겟 평균: {train_processed['power_transformed'].mean():.2f}")
        
        # Log 역변환해서 확인
        original_check = safe_exp_transform(train_processed['power_transformed'].values)
        print(f"Log 역변환 후 범위: {original_check.min():.2f} ~ {original_check.max():.2f}")
        print(f"Log 역변환 후 평균: {original_check.mean():.2f}")
    
    if '전력소비량(kWh)' in train_processed.columns:
        print(f"전처리된 원본 타겟 범위: {train_processed['전력소비량(kWh)'].min():.2f} ~ {train_processed['전력소비량(kWh)'].max():.2f}")
        print(f"전처리된 원본 타겟 평균: {train_processed['전력소비량(kWh)'].mean():.2f}")

def test_simple_baseline():
    print("\n간단한 베이스라인 테스트")
    
    # 전처리된 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    
    # 간단한 피처 선택 (시차 변수 제외)
    simple_features = ['hour', 'weekday', 'month', 'is_weekend', 'is_holiday', 
                      '기온(°C)', '습도(%)', '풍속(m/s)', '연면적(m2)', '냉방면적(m2)']
    
    available_features = [col for col in simple_features if col in train_df.columns]
    print(f"사용 가능한 간단한 피처: {available_features}")
    
    if len(available_features) == 0:
        print("사용 가능한 피처가 없습니다!")
        return
    
    X = train_df[available_features].fillna(0)
    
    # 타겟 확인
    if 'power_transformed' in train_df.columns:
        y = train_df['power_transformed']
        print("Log 변환된 타겟 사용")
    elif '전력소비량(kWh)' in train_df.columns:
        y = np.log1p(train_df['전력소비량(kWh)'])
        print("직접 Log 변환 적용")
    else:
        print("타겟 변수를 찾을 수 없습니다!")
        return
    
    print(f"피처 수: {X.shape[1]}")
    print(f"데이터 수: {len(X)}")
    print(f"타겟 범위: {y.min():.4f} ~ {y.max():.4f}")
    
    # 시계열 분할 (마지막 20%를 validation으로)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # 간단한 모델 훈련
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=31,
        learning_rate=0.1,
        verbosity=-1,
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred_log = model.predict(X_val)
    
    print(f"예측값 (log) 범위: {y_pred_log.min():.4f} ~ {y_pred_log.max():.4f}")
    
    # 원래 스케일로 변환
    y_val_original = safe_exp_transform(y_val.values)
    y_pred_original = safe_exp_transform(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0)
    
    print(f"실제값 범위: {y_val_original.min():.2f} ~ {y_val_original.max():.2f}")
    print(f"예측값 범위: {y_pred_original.min():.2f} ~ {y_pred_original.max():.2f}")
    
    # SMAPE 계산
    smape_score = smape(y_val_original, y_pred_original)
    print(f"SMAPE: {smape_score:.4f}")
    
    # 몇 개 샘플 확인
    print("\n샘플 확인:")
    for i in range(min(10, len(y_val_original))):
        print(f"실제: {y_val_original[i]:.2f}, 예측: {y_pred_original[i]:.2f}")
    
    return smape_score

def test_with_more_features():
    print("\n더 많은 피처로 테스트")
    
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    
    # 더 많은 피처 사용 (시차 변수 포함)
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 'power_deseasonalized']
    
    if 'power_transformed' in train_df.columns:
        exclude_cols.append('power_transformed')
        y = train_df['power_transformed']
    else:
        y = np.log1p(train_df['전력소비량(kWh)'])
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # NaN이 너무 많은 컬럼 제외
    feature_cols = [col for col in feature_cols if train_df[col].notna().sum() > len(train_df) * 0.5]
    
    print(f"사용할 피처 수: {len(feature_cols)}")
    print(f"피처 목록: {feature_cols[:10]}...")  # 처음 10개만 출력
    
    X = train_df[feature_cols].fillna(0)
    
    # 시계열 분할
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    # 모델 훈련
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=100,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        verbosity=-1,
        random_state=42,
        n_estimators=200
    )
    
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred_log = model.predict(X_val)
    
    y_val_original = safe_exp_transform(y_val.values)
    y_pred_original = safe_exp_transform(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0)
    
    smape_score = smape(y_val_original, y_pred_original)
    print(f"SMAPE (더 많은 피처): {smape_score:.4f}")
    
    return smape_score

if __name__ == "__main__":
    print("모델 디버깅 시작")
    print("=" * 50)
    
    # 1. 데이터 분석
    analyze_data()
    
    # 2. 간단한 베이스라인
    simple_score = test_simple_baseline()
    
    # 3. 더 많은 피처로 테스트
    complex_score = test_with_more_features()
    
    print("\n" + "=" * 50)
    print("디버깅 결과 요약")
    print("=" * 50)
    
    if simple_score:
        print(f"간단한 모델 SMAPE: {simple_score:.4f}")
        if simple_score <= 6.0:
            print("  -> 목표 달성!")
        else:
            print(f"  -> 목표까지 {simple_score - 6.0:.4f} 개선 필요")
    
    if complex_score:
        print(f"복잡한 모델 SMAPE: {complex_score:.4f}")
        if complex_score <= 6.0:
            print("  -> 목표 달성!")
        else:
            print(f"  -> 목표까지 {complex_score - 6.0:.4f} 개선 필요")