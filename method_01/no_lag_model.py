#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시차 변수 없는 모델 - Test 데이터에 맞춤
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
    print("Model without Lag Features")
    print("=" * 40)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    y_train = train_df['power_transformed']
    
    # Test에 있는 피처만 사용 (시차 변수 제외)
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed']
    
    # 시차 변수도 제외
    lag_cols = ['power_lag_1h', 'power_lag_24h', 'power_lag_168h',
               'power_rolling_mean_24h', 'power_rolling_std_24h', 'power_rolling_mean_7d']
    
    exclude_cols.extend(lag_cols)
    
    # Test에 실제로 있는 피처만 선택
    available_features = []
    for col in train_df.columns:
        if col not in exclude_cols and col in test_df.columns:
            available_features.append(col)
    
    print(f"Available features: {len(available_features)}")
    print("Features used:")
    for feature in available_features:
        print(f"  {feature}")
    
    X_train = train_df[available_features].fillna(0)
    X_test = test_df[available_features].fillna(0)
    
    print(f"\nData shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Validation split (마지막 20%)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    print(f"Split - Train: {X_tr.shape}, Val: {X_val.shape}")
    
    models = {}
    scores = {}
    predictions = {}
    
    # LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=150,
        learning_rate=0.05,
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
    
    y_pred_lgb = lgb_model.predict(X_val)
    y_val_orig = safe_exp_transform(y_val.values)
    y_pred_orig_lgb = safe_exp_transform(y_pred_lgb)
    y_pred_orig_lgb = np.maximum(y_pred_orig_lgb, 0)
    
    scores['lgb'] = smape(y_val_orig, y_pred_orig_lgb)
    models['lgb'] = lgb_model
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=2,
        reg_lambda=2,
        random_state=42,
        verbosity=0,
        n_estimators=1000
    )
    
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred_xgb = xgb_model.predict(X_val)
    y_pred_orig_xgb = safe_exp_transform(y_pred_xgb)
    y_pred_orig_xgb = np.maximum(y_pred_orig_xgb, 0)
    
    scores['xgb'] = smape(y_val_orig, y_pred_orig_xgb)
    models['xgb'] = xgb_model
    
    # CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_pred_cat = cat_model.predict(X_val)
    y_pred_orig_cat = safe_exp_transform(y_pred_cat)
    y_pred_orig_cat = np.maximum(y_pred_orig_cat, 0)
    
    scores['cat'] = smape(y_val_orig, y_pred_orig_cat)
    models['cat'] = cat_model
    
    # 결과 출력
    print(f"\nValidation Results:")
    for name, score in scores.items():
        print(f"{name.upper()}: {score:.4f}")
        if score <= 6.0:
            print(f"  *** TARGET ACHIEVED! ***")
        else:
            print(f"  Need {score - 6.0:.4f} more")
    
    best_model_name = min(scores, key=scores.get)
    best_score = scores[best_model_name]
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name.upper()} with SMAPE {best_score:.4f}")
    
    # 전체 데이터로 최종 모델 훈련
    print(f"Training final {best_model_name.upper()} model...")
    
    if best_model_name == 'lgb':
        final_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            num_leaves=150,
            learning_rate=0.05,
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
    elif best_model_name == 'xgb':
        final_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=2,
            reg_lambda=2,
            random_state=42,
            verbosity=0,
            n_estimators=1500
        )
    else:  # catboost
        final_model = CatBoostRegressor(
            iterations=1500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
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
    
    print(f"\n{'='*40}")
    print("FINAL RESULTS")
    print(f"{'='*40}")
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Validation SMAPE: {best_score:.4f}")
    print(f"Target SMAPE: 6.0000")
    
    if best_score <= 6.0:
        print("STATUS: *** TARGET ACHIEVED! ***")
    else:
        print(f"STATUS: Need {best_score - 6.0:.4f} more improvement")
    
    print("Submission file: submission.csv")
    print(f"Features used: {len(available_features)} (no lag features)")
    
    # 피처 중요도
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.0f}")

if __name__ == "__main__":
    main()