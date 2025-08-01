#!/usr/bin/env python3
"""
Simple improvements to the original high-performance solution
기존 7-8 SMAPE 성능을 5-6으로 개선하기 위한 간단한 개선사항들
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# 원본 함수들 import
from solution_backup import load_data, engineer_features, smape

def simple_improvements():
    """기존 고성능 솔루션에 간단한 개선사항만 적용."""
    print("=" * 60)
    print("SIMPLE IMPROVEMENTS FOR 7-8 → 5-6 SMAPE")
    print("=" * 60)
    
    # 1. 데이터 로드 (원본 방식)
    print("1. Loading data with original method...")
    base_dir = Path('../data')
    train_path = base_dir / 'train.csv'
    test_path = base_dir / 'test.csv'
    building_path = base_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    print(f"   Data loaded: Train {train_fe.shape}, Test {test_fe.shape}")
    
    # 2. 기존 피처에 몇 가지만 추가
    print("2. Adding minimal feature improvements...")
    
    for df in [train_fe, test_fe]:
        # 간단한 시간 피처 추가
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 온도 구간별 피처
        df['temp_category'] = pd.cut(df['temp'], bins=5, labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
        
        # 건물별 효율성 간단 지표
        df['efficiency_score'] = (df['pv_capacity'] + 1) / (df['total_area'] + 1) * 1000
    
    print(f"   Features added: {train_fe.shape[1]} total features")
    
    # 3. 모델 설정 (원본에서 약간만 조정)
    print("3. Training improved model...")
    
    # 피처 준비
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['num_date_time', '건물번호', '일시', 'datetime', '전력소비량(kWh)']]
    
    X = train_fe[feature_cols]
    y = train_fe['전력소비량(kWh)']
    X_test = test_fe[feature_cols]
    
    # 전처리 파이프라인 (원본과 동일)
    categorical_features = ['building_type', 'temp_category']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 원본 모델에서 약간만 조정 (과최적화 방지)
    model = xgb.XGBRegressor(
        max_depth=10,  # 12 → 10 (약간 단순화)
        n_estimators=1200,  # 1000 → 1200 (약간 증가)
        learning_rate=0.025,  # 0.03 → 0.025 (더 정교하게)
        subsample=0.85,  # 0.8 → 0.85
        colsample_bytree=0.85,  # 0.8 → 0.85
        reg_alpha=0.1,  # 0.0 → 0.1 (약간의 정규화)
        reg_lambda=1.5,  # 1.0 → 1.5 (더 강한 정규화)
        objective='reg:squarederror',
        tree_method='hist',  # CPU 사용 (로컬 테스트용)
        random_state=42,
    )
    
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    
    # 4. 검증 (원본 방식)
    cutoff = pd.Timestamp('2024-08-18')
    train_mask = train_fe['datetime'] < cutoff
    val_mask = ~train_mask
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    
    # 학습 및 검증
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    val_smape = smape(y_val.values, val_pred)
    
    print(f"4. Validation SMAPE: {val_smape:.4f}")
    
    # 5. 전체 데이터로 재학습 및 예측
    print("5. Final training and prediction...")
    pipeline.fit(X, y)
    test_pred = pipeline.predict(X_test)
    
    # 제출 파일 생성
    submission = test_fe[['num_date_time']].copy()
    submission['answer'] = test_pred
    submission.to_csv('submission_simple_improved.csv', index=False)
    
    print(f"✅ Simple improvement completed!")
    print(f"📊 Validation SMAPE: {val_smape:.4f}")
    print(f"💾 Submission saved: submission_simple_improved.csv")
    print(f"🎯 Target: Improve 7-8 SMAPE → 5-6 SMAPE")
    
    return val_smape

if __name__ == "__main__":
    result = simple_improvements()