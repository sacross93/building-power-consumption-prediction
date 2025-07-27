#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 모델 - 빠른 최적화와 앙상블
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

def prepare_data():
    """데이터 준비"""
    print("데이터 준비 중...")
    
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
    
    # 시차 변수들 (가장 중요)
    lag_features = [
        'power_lag_1h', 'power_lag_24h', 'power_lag_168h',
        'power_rolling_mean_24h', 'power_rolling_std_24h', 'power_rolling_mean_7d'
    ]
    
    # 건물 유형 더미 변수들
    building_type_features = [col for col in train_df.columns if col.startswith('건물유형_')]
    
    all_features = important_features + lag_features + building_type_features
    
    # 실제 존재하는 피처만 선택
    available_features = [col for col in all_features if col in train_df.columns]
    
    X_train = train_df[available_features].fillna(0)
    
    # Test 데이터도 동일한 피처로
    test_available_features = [col for col in available_features if col in test_df.columns]
    X_test = test_df[test_available_features].fillna(0)
    
    print(f"Train 피처: {X_train.shape[1]}, Test 피처: {X_test.shape[1]}")
    print(f"사용된 피처: {test_available_features}")
    
    return X_train, y_train, X_test, test_df['num_date_time']

def test_model_performance(X_train, y_train):
    """모델 성능 테스트"""
    print("모델 성능 테스트 중...")
    
    # 시계열 분할 (마지막 20%를 validation으로)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    models = {}
    scores = {}
    
    # LightGBM (최적화된 파라미터)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 120,
        'learning_rate': 0.08,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 2,
        'lambda_l2': 2,
        'verbosity': -1,
        'random_state': 42
    }
    
    models['lgb'] = lgb.LGBMRegressor(**lgb_params, n_estimators=800)
    models['lgb'].fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    # XGBoost (최적화된 파라미터)
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 2,
        'reg_lambda': 2,
        'random_state': 42,
        'verbosity': 0
    }
    
    models['xgb'] = xgb.XGBRegressor(**xgb_params, n_estimators=800)
    models['xgb'].fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    # CatBoost
    models['cat'] = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    models['cat'].fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    # 각 모델 성능 평가
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        y_val_orig = safe_exp_transform(y_val.values)
        y_pred_orig = safe_exp_transform(y_pred)
        y_pred_orig = np.maximum(y_pred_orig, 0)
        
        score = smape(y_val_orig, y_pred_orig)
        scores[name] = score
        predictions[name] = y_pred_orig
        
        print(f"{name.upper()} SMAPE: {score:.4f}")
    
    # 앙상블
    ensemble_pred = (predictions['lgb'] * 0.4 + 
                    predictions['xgb'] * 0.4 + 
                    predictions['cat'] * 0.2)
    
    ensemble_score = smape(y_val_orig, ensemble_pred)
    scores['ensemble'] = ensemble_score
    
    print(f"Ensemble SMAPE: {ensemble_score:.4f}")
    
    return models, scores

def train_final_models_and_predict(X_train, y_train, X_test):
    """최종 모델 훈련 및 예측"""
    print("최종 모델 훈련 중...")
    
    predictions = {}
    
    # LightGBM
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
        n_estimators=1200
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = safe_exp_transform(lgb_model.predict(X_test))
    predictions['lgb'] = np.maximum(lgb_pred, 0)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=2,
        reg_lambda=2,
        random_state=42,
        verbosity=0,
        n_estimators=1200
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = safe_exp_transform(xgb_model.predict(X_test))
    predictions['xgb'] = np.maximum(xgb_pred, 0)
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1200,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_pred = safe_exp_transform(cat_model.predict(X_test))
    predictions['cat'] = np.maximum(cat_pred, 0)
    
    # 앙상블
    ensemble_pred = (predictions['lgb'] * 0.4 + 
                    predictions['xgb'] * 0.4 + 
                    predictions['cat'] * 0.2)
    
    return ensemble_pred, predictions

def main():
    print("최종 모델링 시작")
    print("=" * 50)
    
    # 데이터 준비
    X_train, y_train, X_test, test_ids = prepare_data()
    
    # 모델 성능 테스트
    models, scores = test_model_performance(X_train, y_train)
    
    print(f"\n성능 테스트 결과:")
    for name, score in scores.items():
        print(f"{name.upper()}: {score:.4f}")
        if score <= 6.0:
            print(f"  ✅ 목표 달성!")
        else:
            print(f"  📈 목표까지 {score - 6.0:.4f} 개선 필요")
    
    best_score = min(scores.values())
    print(f"\n최고 성능: {best_score:.4f}")
    
    # 최종 예측
    ensemble_pred, individual_preds = train_final_models_and_predict(X_train, y_train, X_test)
    
    # 제출 파일들 생성
    print(f"\n제출 파일 생성 중...")
    
    # 앙상블 제출 파일
    submission = pd.DataFrame({
        'num_date_time': test_ids,
        'answer': ensemble_pred
    })
    submission.to_csv('submission.csv', index=False)
    print(f"앙상블 제출 파일: submission.csv")
    
    # 개별 모델 제출 파일들
    for name, pred in individual_preds.items():
        individual_submission = pd.DataFrame({
            'num_date_time': test_ids,
            'answer': pred
        })
        individual_submission.to_csv(f'submission_{name}.csv', index=False)
        print(f"{name.upper()} 제출 파일: submission_{name}.csv")
    
    print(f"\n최종 결과:")
    print(f"앙상블 예측값 범위: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f}")
    print(f"앙상블 예측값 평균: {ensemble_pred.mean():.2f}")
    
    if best_score <= 6.0:
        print("🎉 목표 달성! SMAPE ≤ 6.0")
        print("제출 파일이 준비되었습니다!")
    else:
        print(f"목표까지 {best_score - 6.0:.4f} 더 개선이 필요하지만,")
        print("현재 성능도 경쟁력 있는 수준입니다!")

if __name__ == "__main__":
    main()