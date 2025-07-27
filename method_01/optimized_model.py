#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 모델 - SMAPE 6 이하 달성
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import warnings
from sklearn.model_selection import KFold

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
    """데이터 준비 및 피처 선택"""
    print("데이터 준비 중...")
    
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    # 타겟 설정
    y_train = train_df['power_transformed']
    
    # 피처 선택 (중요한 피처들 우선)
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # NaN이 많은 컬럼 제거
    good_features = []
    for col in feature_cols:
        if train_df[col].notna().sum() > len(train_df) * 0.7:  # 70% 이상 valid
            good_features.append(col)
    
    print(f"선택된 피처 수: {len(good_features)}")
    
    # Train 데이터
    X_train = train_df[good_features].fillna(0)
    
    # Test 데이터 (Train과 공통 피처만)
    test_features = [col for col in good_features if col in test_df.columns]
    X_test = test_df[test_features].fillna(0)
    
    print(f"Train 피처: {X_train.shape[1]}, Test 피처: {X_test.shape[1]}")
    
    return X_train, y_train, X_test, test_df['num_date_time']

def optimize_lightgbm(X_train, y_train, n_trials=50):
    """LightGBM 최적화"""
    print("LightGBM 최적화 중...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'verbosity': -1,
            'random_state': 42
        }
        
        # 간단한 time series cross validation
        scores = []
        kf = KFold(n_splits=3, shuffle=False)
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params, n_estimators=500)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_val)
            
            # SMAPE 계산
            y_val_orig = safe_exp_transform(y_val.values)
            y_pred_orig = safe_exp_transform(y_pred)
            y_pred_orig = np.maximum(y_pred_orig, 0)
            
            score = smape(y_val_orig, y_pred_orig)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=1200)  # 20분 제한
    
    print(f"LightGBM 최적 SMAPE: {study.best_value:.4f}")
    return study.best_params, study.best_value

def optimize_xgboost(X_train, y_train, n_trials=50):
    """XGBoost 최적화"""
    print("XGBoost 최적화 중...")
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': 0
        }
        
        scores = []
        kf = KFold(n_splits=3, shuffle=False)
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params, n_estimators=500)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50, verbose=False)
            
            y_pred = model.predict(X_val)
            
            y_val_orig = safe_exp_transform(y_val.values)
            y_pred_orig = safe_exp_transform(y_pred)
            y_pred_orig = np.maximum(y_pred_orig, 0)
            
            score = smape(y_val_orig, y_pred_orig)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=1200)
    
    print(f"XGBoost 최적 SMAPE: {study.best_value:.4f}")
    return study.best_params, study.best_value

def train_and_predict(X_train, y_train, X_test, lgb_params, xgb_params):
    """최종 모델 훈련 및 예측"""
    print("최종 모델 훈련 중...")
    
    predictions = {}
    
    # LightGBM
    print("LightGBM 훈련...")
    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    predictions['lgb'] = safe_exp_transform(lgb_pred)
    
    # XGBoost
    print("XGBoost 훈련...")
    xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1000, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    predictions['xgb'] = safe_exp_transform(xgb_pred)
    
    # CatBoost (간단한 파라미터)
    print("CatBoost 훈련...")
    cat_model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    predictions['cat'] = safe_exp_transform(cat_pred)
    
    # 앙상블 (가중 평균)
    ensemble_pred = (predictions['lgb'] * 0.4 + 
                    predictions['xgb'] * 0.4 + 
                    predictions['cat'] * 0.2)
    
    ensemble_pred = np.maximum(ensemble_pred, 0)
    
    print("예측 완료!")
    return ensemble_pred, predictions

def main():
    print("최적화된 모델링 시작")
    print("=" * 50)
    
    # 데이터 준비
    X_train, y_train, X_test, test_ids = prepare_data()
    
    # 모델 최적화
    lgb_params, lgb_score = optimize_lightgbm(X_train, y_train, n_trials=30)
    xgb_params, xgb_score = optimize_xgboost(X_train, y_train, n_trials=30)
    
    print(f"\n최적화 결과:")
    print(f"LightGBM SMAPE: {lgb_score:.4f}")
    print(f"XGBoost SMAPE: {xgb_score:.4f}")
    
    best_score = min(lgb_score, xgb_score)
    print(f"최고 성능: {best_score:.4f}")
    
    if best_score <= 6.0:
        print("🎉 목표 달성! SMAPE ≤ 6.0")
    else:
        print(f"목표까지 {best_score - 6.0:.4f} 더 개선 필요")
    
    # 최종 예측
    ensemble_pred, individual_preds = train_and_predict(
        X_train, y_train, X_test, lgb_params, xgb_params
    )
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_ids,
        'answer': ensemble_pred
    })
    
    submission_path = 'submission_optimized.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n제출 파일 생성: {submission_path}")
    print(f"예측값 범위: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f}")
    print(f"예측값 평균: {ensemble_pred.mean():.2f}")
    
    # 개별 모델 예측도 저장
    for name, pred in individual_preds.items():
        individual_submission = pd.DataFrame({
            'num_date_time': test_ids,
            'answer': pred
        })
        individual_submission.to_csv(f'submission_{name}.csv', index=False)
        print(f"개별 모델 {name} 제출 파일: submission_{name}.csv")

if __name__ == "__main__":
    main()