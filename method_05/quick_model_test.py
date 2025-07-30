"""
Quick Model Performance Test
===========================

빠른 모델 성능 테스트 (1-fold 교차검증)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features, smape
from ts_validation import TimeSeriesCV
from improved_preprocessing import ImprovedPreprocessor


def quick_test():
    """빠른 성능 테스트."""
    print("=" * 60)
    print("QUICK MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    # 데이터 로드
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 1. 기존 전처리 테스트
    print("\n1. Testing Original Preprocessing...")
    train_fe, test_fe = engineer_features(train_df.copy(), test_df.copy())
    
    # 피처 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
    
    X_orig = train_fe[feature_cols].copy()
    y_orig = train_fe['전력소비량(kWh)']
    
    # 카테고리 인코딩
    categorical_cols = ['건물번호', 'building_type']
    for col in categorical_cols:
        if col in X_orig.columns:
            le = LabelEncoder()
            X_orig[col] = le.fit_transform(X_orig[col].astype(str))
    
    # 객체 타입 처리
    for col in X_orig.columns:
        if X_orig[col].dtype == 'object':
            le = LabelEncoder()
            X_orig[col] = le.fit_transform(X_orig[col].astype(str))
    
    # 간단한 시계열 분할 (마지막 7일)
    cutoff = pd.Timestamp('2024-08-18')
    train_mask = train_fe['datetime'] < cutoff
    val_mask = ~train_mask
    
    X_train_orig = X_orig.loc[train_mask]
    y_train_orig = y_orig.loc[train_mask]
    X_val_orig = X_orig.loc[val_mask]
    y_val_orig = y_orig.loc[val_mask]
    
    # XGBoost 모델 훈련
    model_orig = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=100,  # 빠른 테스트용
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = model_orig.predict(X_val_orig)
    smape_orig = smape(y_val_orig.values, y_pred_orig)
    
    print(f"Original Preprocessing SMAPE: {smape_orig:.4f}")
    
    # 2. 개선된 전처리 테스트
    print("\n2. Testing Improved Preprocessing...")
    preprocessor = ImprovedPreprocessor()
    X_improved, X_test_improved, y_improved = preprocessor.fit_transform(train_df, test_df)
    
    # 같은 분할 기준 적용
    train_processed, _ = engineer_features(train_df.copy(), test_df.copy())
    train_mask_improved = train_processed['datetime'] < cutoff
    val_mask_improved = ~train_mask_improved
    
    X_train_improved = X_improved.loc[train_mask_improved]
    y_train_improved = y_improved.loc[train_mask_improved]
    X_val_improved = X_improved.loc[val_mask_improved]
    y_val_improved = y_improved.loc[val_mask_improved]
    
    # XGBoost 모델 훈련
    model_improved = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=100,  # 빠른 테스트용
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model_improved.fit(X_train_improved, y_train_improved)
    y_pred_improved = model_improved.predict(X_val_improved)
    
    # 로그 변환 역변환
    y_val_original = preprocessor.inverse_transform_target(y_val_improved)
    y_pred_original = preprocessor.inverse_transform_target(y_pred_improved)
    
    smape_improved = smape(y_val_original.values, y_pred_original)
    
    print(f"Improved Preprocessing SMAPE: {smape_improved:.4f}")
    
    # 3. 결과 비교
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    improvement = (smape_orig - smape_improved) / smape_orig * 100
    
    print(f"Original SMAPE:   {smape_orig:.4f}")
    print(f"Improved SMAPE:   {smape_improved:.4f}")
    print(f"Improvement:      {improvement:+.2f}%")
    
    if improvement > 0:
        print("✅ Improved preprocessing shows better performance!")
    else:
        print("❌ Original preprocessing performs better")
    
    # 4. 피처 중요도 비교
    print(f"\nFeature count comparison:")
    print(f"Original features: {X_orig.shape[1]}")
    print(f"Improved features: {X_improved.shape[1]}")
    
    # 5. 간단한 시각화 정보
    print(f"\nTarget transformation effect:")
    print(f"Original target mean: {y_orig.mean():.2f}, std: {y_orig.std():.2f}")
    print(f"Improved target mean: {y_improved.mean():.2f}, std: {y_improved.std():.2f}")
    
    return {
        'original_smape': smape_orig,
        'improved_smape': smape_improved,
        'improvement_pct': improvement,
        'original_features': X_orig.shape[1],
        'improved_features': X_improved.shape[1]
    }


if __name__ == "__main__":
    results = quick_test()
    print(f"\n🎯 Quick test completed!")
    
    if results['improvement_pct'] > 0:
        print(f"🚀 Improved preprocessing is {results['improvement_pct']:.2f}% better!")
    else:
        print(f"⚠️  Original preprocessing is {abs(results['improvement_pct']):.2f}% better")