#!/usr/bin/env python3
"""
정규화 방법들의 성능 비교 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from advanced_feature_engineering import AdvancedFeatureEngineer
from improved_preprocessing import ImprovedPreprocessor
import warnings
warnings.filterwarnings('ignore')

def smape_score(y_true, y_pred):
    """SMAPE 계산."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def test_normalization_methods():
    """다양한 정규화 방법 테스트."""
    print("=" * 60)
    print("NORMALIZATION METHODS COMPARISON")
    print("=" * 60)
    
    # 1. 데이터 로드 및 고급 피처 엔지니어링
    print("1. Loading data and applying advanced FE...")
    from solution import load_data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 고급 피처 엔지니어링 적용
    engineer = AdvancedFeatureEngineer()
    train_advanced, test_advanced, _ = engineer.apply_advanced_feature_engineering(
        train_df, test_df
    )
    
    print(f"   Advanced FE completed: {train_advanced.shape[1]} features")
    
    # 2. 베이스라인 (현재 RobustScaler)
    print("\n2. Testing normalization methods...")
    
    scalers = {
        'RobustScaler (Current)': RobustScaler(),
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    results = {}
    
    for scaler_name, scaler in scalers.items():
        print(f"\n   Testing {scaler_name}...")
        
        # ImprovedPreprocessor에 scaler 적용
        preprocessor = ImprovedPreprocessor()
        preprocessor.scaler = scaler  # scaler 교체
        
        X_train, X_test, y_train = preprocessor.fit_transform(
            train_advanced.copy(), test_advanced.copy()
        )
        
        # 빠른 XGBoost 모델로 테스트
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,  # 빠른 테스트용
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            # 로그 역변환
            y_val_orig = np.expm1(y_val)
            y_pred_orig = np.expm1(y_pred)
            
            smape = smape_score(y_val_orig, y_pred_orig)
            cv_scores.append(smape)
        
        avg_smape = np.mean(cv_scores)
        std_smape = np.std(cv_scores)
        results[scaler_name] = {'mean': avg_smape, 'std': std_smape}
        
        print(f"      SMAPE: {avg_smape:.3f} ± {std_smape:.3f}")
    
    # 3. 결과 정리
    print("\n" + "=" * 60)
    print("NORMALIZATION COMPARISON RESULTS")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for i, (method, score) in enumerate(sorted_results):
        status = "🥇 BEST" if i == 0 else f"#{i+1}"
        print(f"{status:>8} {method:<25} SMAPE: {score['mean']:.3f} ± {score['std']:.3f}")
    
    best_method = sorted_results[0][0]
    improvement = results['RobustScaler (Current)']['mean'] - sorted_results[0][1]['mean']
    
    print(f"\n🎯 Best method: {best_method}")
    if improvement > 0:
        print(f"📈 Improvement over current: -{improvement:.3f} SMAPE")
        print(f"💡 Recommendation: Switch to {best_method}")
    else:
        print(f"📊 Current RobustScaler is optimal")
    
    return best_method, results

if __name__ == "__main__":
    best_method, results = test_normalization_methods()