#!/usr/bin/env python3
"""
정규화 방법 빠른 비교 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

def smape_score(y_true, y_pred):
    """SMAPE 계산."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def quick_scaler_test():
    """빠른 정규화 방법 테스트."""
    print("=" * 50)
    print("QUICK SCALER COMPARISON TEST")
    print("=" * 50)
    
    from ultimate_tuning_solution import UltimateTuningSolution
    from improved_preprocessing import ImprovedPreprocessor
    
    # 1. 데이터 로드
    print("1. Loading data...")
    solution = UltimateTuningSolution(quick_mode=True, max_trials=10)
    
    from solution import load_data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 고급 피처 엔지니어링 (한 번만)
    from advanced_feature_engineering import AdvancedFeatureEngineer
    engineer = AdvancedFeatureEngineer()
    train_advanced, test_advanced, _ = engineer.apply_advanced_feature_engineering(
        train_df, test_df
    )
    
    # 2. 각 scaler 테스트
    scalers = ['robust', 'standard', 'minmax', 'quantile']
    results = {}
    
    for scaler_name in scalers:
        print(f"\n2. Testing {scaler_name} scaler...")
        
        try:
            # 전처리
            preprocessor = ImprovedPreprocessor(scaler_type=scaler_name)
            X_train, _, y_train = preprocessor.fit_transform(
                train_advanced.copy(), test_advanced.copy()
            )
            
            # 빠른 XGBoost 모델
            model = xgb.XGBRegressor(
                max_depth=6,
                n_estimators=50,  # 매우 빠른 테스트
                learning_rate=0.1,
                tree_method='gpu_hist',
                gpu_id=0,
                random_state=42,
                verbosity=0
            )
            
            # 시계열 교차검증 (2 folds만)
            tscv = TimeSeriesSplit(n_splits=2)
            
            def neg_smape(y_true, y_pred):
                y_true_orig = np.expm1(y_true)
                y_pred_orig = np.expm1(y_pred)
                return -smape_score(y_true_orig, y_pred_orig)
            
            scorer = make_scorer(neg_smape)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scorer, n_jobs=1)
            
            avg_smape = -np.mean(scores)
            std_smape = np.std(scores)
            
            results[scaler_name] = {'mean': avg_smape, 'std': std_smape}
            print(f"   {scaler_name:<10} SMAPE: {avg_smape:.3f} ± {std_smape:.3f}")
            
        except Exception as e:
            print(f"   {scaler_name:<10} ERROR: {e}")
            results[scaler_name] = {'mean': float('inf'), 'std': 0}
    
    # 3. 결과 정리
    print("\n" + "=" * 50)
    print("SCALER COMPARISON RESULTS")
    print("=" * 50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for i, (method, score) in enumerate(sorted_results):
        if score['mean'] == float('inf'):
            status = "❌ ERROR"
            smape_str = "FAILED"
        else:
            status = "🥇 BEST" if i == 0 else f"#{i+1}"
            smape_str = f"{score['mean']:.3f} ± {score['std']:.3f}"
        
        print(f"{status:>8} {method:<12} SMAPE: {smape_str}")
    
    if sorted_results[0][1]['mean'] != float('inf'):
        best_method = sorted_results[0][0]
        print(f"\n🎯 Best scaler: {best_method}")
        print(f"💡 Recommendation: Use '{best_method}' in ultimate_tuning_solution.py")
        
        # ultimate_tuning_solution.py 수정 제안
        print(f"\n📝 To apply: change scaler_type='{best_method}' in ultimate_tuning_solution.py")
    
    return sorted_results

if __name__ == "__main__":
    results = quick_scaler_test()