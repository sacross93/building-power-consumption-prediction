#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 앙상블 모델 - Phase A1 간소화 버전
목표: SMAPE 7.95 → 6.5~7.0 (빠른 검증)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')

def smape(y_true, y_pred, epsilon=1e-8):
    """SMAPE 계산"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > epsilon
    smape_values = np.zeros_like(numerator, dtype=float)
    smape_values[mask] = numerator[mask] / denominator[mask]
    
    return 100.0 * np.mean(smape_values)

def load_and_prepare_data():
    """데이터 로드 및 기본 준비"""
    print("데이터 로드 중...")
    
    # 데이터 로드
    train_df = pd.read_csv('../data/train.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('../data/test.csv', encoding='utf-8-sig')
    building_info = pd.read_csv('../data/building_info.csv', encoding='utf-8-sig')
    
    # 컬럼명 정리
    for df in [train_df, test_df, building_info]:
        df.columns = df.columns.str.strip()
    
    # 건물 정보 병합
    train_df = train_df.merge(building_info, on='건물번호', how='left')
    test_df = test_df.merge(building_info, on='건물번호', how='left')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def quick_feature_engineering(train_df, test_df):
    """method_05 핵심 기법 적용"""
    print("핵심 피처 엔지니어링 중...")
    
    # 1. 시간 관련 피처 생성
    for df in [train_df, test_df]:
        df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
        df['year'] = df['일시'].dt.year
        df['month'] = df['일시'].dt.month
        df['day'] = df['일시'].dt.day
        df['hour'] = df['일시'].dt.hour
        df['weekday'] = df['일시'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # 2. 건물 정보 전처리
    numeric_building_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 
                           'ESS저장용량(kWh)', 'PCS용량(kW)']
    
    for col in numeric_building_cols:
        for df in [train_df, test_df]:
            df[col] = df[col].replace('-', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = train_df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # 3. 건물별 통계 피처 생성 (method_05의 핵심)
    print("건물별 통계 피처 생성...")
    
    # 전체 건물별 평균
    building_mean = train_df.groupby('건물번호')['전력소비량(kWh)'].mean()
    
    # 건물별 시간대별 평균
    bld_hour_mean = (
        train_df.groupby(['건물번호', 'hour'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_hour_mean'})
    )
    train_df = train_df.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test_df = test_df.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test_df['bld_hour_mean'] = test_df['bld_hour_mean'].fillna(
        test_df['건물번호'].map(building_mean)
    )
    
    # 건물별 요일별 평균
    bld_weekday_mean = (
        train_df.groupby(['건물번호', 'weekday'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_weekday_mean'})
    )
    train_df = train_df.merge(bld_weekday_mean, on=['건물번호', 'weekday'], how='left')
    test_df = test_df.merge(bld_weekday_mean, on=['건물번호', 'weekday'], how='left')
    test_df['bld_weekday_mean'] = test_df['bld_weekday_mean'].fillna(
        test_df['건물번호'].map(building_mean)
    )
    
    # 건물별 월별 평균
    bld_month_mean = (
        train_df.groupby(['건물번호', 'month'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_month_mean'})
    )
    train_df = train_df.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test_df = test_df.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test_df['bld_month_mean'] = test_df['bld_month_mean'].fillna(
        test_df['건물번호'].map(building_mean)
    )
    
    # 4. 누락 데이터 대체 (method_05 기법)
    print("누락 데이터 대체...")
    
    # 8월 데이터의 시간대별 평균으로 일조/일사량 추정
    train_august = train_df[train_df['month'] == 8]
    avg_sunshine = train_august.groupby('hour')['일조(hr)'].mean()
    avg_solar = train_august.groupby('hour')['일사(MJ/m2)'].mean()
    
    # Train에는 원본 데이터 사용, Test에는 추정값 사용
    train_df['sunshine_est'] = train_df['일조(hr)']
    train_df['solar_est'] = train_df['일사(MJ/m2)']
    test_df['sunshine_est'] = test_df['hour'].map(avg_sunshine)
    test_df['solar_est'] = test_df['hour'].map(avg_solar)
    
    # 5. 상호작용 피처 생성
    print("상호작용 피처 생성...")
    
    for df in [train_df, test_df]:
        # 기상 상호작용
        df['humidity_temp'] = df['습도(%)'] * df['기온(°C)']
        df['rain_wind'] = df['강수량(mm)'] * df['풍속(m/s)']
        df['temp_wind'] = df['기온(°C)'] * df['풍속(m/s)']
        
        # 건물 관련 비율
        df['cooling_area_ratio'] = df['냉방면적(m2)'] / df['연면적(m2)']
        df['pv_per_area'] = df['태양광용량(kW)'] / df['연면적(m2)']
        df['ess_per_area'] = df['ESS저장용량(kWh)'] / df['연면적(m2)']
        
        # 기상과 건물 상호작용
        df['temp_area'] = df['기온(°C)'] * df['연면적(m2)']
        df['humidity_cooling_area'] = df['습도(%)'] * df['냉방면적(m2)']
    
    print(f"피처 엔지니어링 완료 - Train: {train_df.shape[1]}개 컬럼, Test: {test_df.shape[1]}개 컬럼")
    
    return train_df, test_df

def prepare_features_for_modeling(train_df, test_df):
    """모델링을 위한 피처 준비"""
    print("모델링용 피처 준비...")
    
    # 제외할 컬럼들
    exclude_cols = ['전력소비량(kWh)', '일시', 'num_date_time']
    
    # 공통 피처 선택
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols and col in test_df.columns]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['전력소비량(kWh)']
    X_test = test_df[feature_cols].copy()
    
    # NaN 처리 먼저
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # 범주형 변수 처리 (LightGBM용)
    categorical_cols = ['건물번호', '건물유형']
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    # 수치형 버전 (XGBoost, CatBoost용)
    X_train_numeric = X_train.copy()
    X_test_numeric = X_test.copy()
    
    for col in categorical_cols:
        if col in X_train_numeric.columns:
            if X_train_numeric[col].dtype.name == 'category':
                X_train_numeric[col] = X_train_numeric[col].cat.codes
                X_test_numeric[col] = X_test_numeric[col].cat.codes
    
    print(f"최종 피처 수: {len(feature_cols)}")
    print(f"범주형 피처: {[col for col in categorical_cols if col in feature_cols]}")
    
    return X_train, y_train, X_test, X_train_numeric, X_test_numeric, feature_cols, categorical_cols

def create_chronological_split(train_df, X_train, y_train, validation_days=7):
    """시계열 고려한 chronological split"""
    print(f"Chronological split - 마지막 {validation_days}일을 validation으로 사용")
    
    train_df_sorted = train_df.sort_values('일시')
    cutoff_date = train_df_sorted['일시'].max() - pd.Timedelta(days=validation_days)
    
    train_mask = train_df_sorted['일시'] < cutoff_date
    val_mask = ~train_mask
    
    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    
    print(f"Split - Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Cutoff date: {cutoff_date}")
    
    return X_tr, y_tr, X_val, y_val, train_mask, val_mask

def main():
    print("빠른 앙상블 모델 - Phase A1 간소화")
    print("=" * 60)
    print("목표: SMAPE 7.95 → 6.5~7.0 (빠른 검증)")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 데이터 로드 및 피처 엔지니어링
    train_df, test_df = load_and_prepare_data()
    train_df, test_df = quick_feature_engineering(train_df, test_df)
    X_train, y_train, X_test, X_train_numeric, X_test_numeric, feature_cols, categorical_cols = prepare_features_for_modeling(train_df, test_df)
    
    # 2. Chronological split
    X_tr, y_tr, X_val, y_val, train_mask, val_mask = create_chronological_split(train_df, X_train, y_train)
    
    # 수치형 버전도 split
    X_tr_numeric = X_train_numeric[train_mask]
    X_val_numeric = X_train_numeric[val_mask]
    
    categorical_features = [col for col in categorical_cols if col in X_train.columns]
    
    models = {}
    predictions = {}
    val_scores = {}
    
    # 3. LightGBM (최적화된 파라미터 사용)
    print("\\nLightGBM 훈련...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=350,
        learning_rate=0.06,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=15,
        lambda_l1=1.5,
        lambda_l2=1.5,
        max_depth=12,
        verbosity=-1,
        random_state=42,
        n_estimators=2000
    )
    
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    y_pred_lgb = lgb_model.predict(X_val)
    y_pred_lgb = np.maximum(y_pred_lgb, 0)
    val_scores['lgb'] = smape(y_val.values, y_pred_lgb)
    models['lgb'] = lgb_model
    
    print(f"LightGBM Validation SMAPE: {val_scores['lgb']:.4f}")
    
    # 4. XGBoost (조정된 파라미터)
    print("XGBoost 훈련...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=10,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=1,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42,
        verbosity=0,
        n_estimators=2000
    )
    
    xgb_model.fit(X_tr_numeric, y_tr, eval_set=[(X_val_numeric, y_val)], verbose=False)
    
    y_pred_xgb = xgb_model.predict(X_val_numeric)
    y_pred_xgb = np.maximum(y_pred_xgb, 0)
    val_scores['xgb'] = smape(y_val.values, y_pred_xgb)
    models['xgb'] = xgb_model
    
    print(f"XGBoost Validation SMAPE: {val_scores['xgb']:.4f}")
    
    # 5. CatBoost (조정된 파라미터)
    print("CatBoost 훈련...")
    cat_model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=3,
        bootstrap_type='Bayesian',
        bagging_temperature=0.5,
        random_seed=42,
        verbose=False
    )
    
    cat_model.fit(X_tr_numeric, y_tr, eval_set=(X_val_numeric, y_val))
    
    y_pred_cat = cat_model.predict(X_val_numeric)
    y_pred_cat = np.maximum(y_pred_cat, 0)
    val_scores['cat'] = smape(y_val.values, y_pred_cat)
    models['cat'] = cat_model
    
    print(f"CatBoost Validation SMAPE: {val_scores['cat']:.4f}")
    
    # 6. 최종 모델 훈련 및 예측
    print("\\n최종 모델들 훈련...")
    
    # LightGBM
    final_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=350,
        learning_rate=0.06,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=15,
        lambda_l1=1.5,
        lambda_l2=1.5,
        max_depth=12,
        verbosity=-1,
        random_state=42,
        n_estimators=3000
    )
    final_lgb.fit(X_train, y_train, categorical_feature=categorical_features)
    predictions['lgb'] = np.maximum(final_lgb.predict(X_test), 0)
    
    # XGBoost
    final_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=10,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=1,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42,
        verbosity=0,
        n_estimators=3000
    )
    final_xgb.fit(X_train_numeric, y_train)
    predictions['xgb'] = np.maximum(final_xgb.predict(X_test_numeric), 0)
    
    # CatBoost
    final_cat = CatBoostRegressor(
        iterations=3000,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=3,
        bootstrap_type='Bayesian',
        bagging_temperature=0.5,
        random_seed=42,
        verbose=False
    )
    final_cat.fit(X_train_numeric, y_train)
    predictions['cat'] = np.maximum(final_cat.predict(X_test_numeric), 0)
    
    # 7. 가중 앙상블 생성
    print("가중 앙상블 생성...")
    
    # 역수 가중치 (낮은 SMAPE일수록 높은 가중치)
    total_weight = sum(1/score for score in val_scores.values())
    weights = {name: (1/score)/total_weight for name, score in val_scores.items()}
    
    ensemble_pred = np.zeros(len(predictions['lgb']))
    for model_name, weight in weights.items():
        ensemble_pred += weight * predictions[model_name]
    
    # 8. 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': ensemble_pred
    })
    
    submission.to_csv('submission_quick_ensemble.csv', index=False)
    
    # 9. 결과 출력
    elapsed_time = time.time() - start_time
    best_individual_score = min(val_scores.values())
    
    print(f"\\n{'='*60}")
    print("빠른 앙상블 결과")
    print(f"{'='*60}")
    print(f"모델별 Validation SMAPE:")
    for name, score in val_scores.items():
        print(f"  {name.upper()}: {score:.4f}")
    
    print(f"\\n앙상블 가중치:")
    for name, weight in weights.items():
        print(f"  {name.upper()}: {weight:.4f}")
    
    print(f"\\n최고 개별 모델 성능: {best_individual_score:.4f}")
    print(f"예측값 통계:")
    print(f"  범위: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f}")
    print(f"  평균: {ensemble_pred.mean():.2f}")
    print(f"  표준편차: {ensemble_pred.std():.2f}")
    
    print(f"\\n실행 시간: {elapsed_time/60:.1f}분")
    print(f"사용된 피처 수: {len(feature_cols)}")
    print("제출 파일: submission_quick_ensemble.csv")
    
    if best_individual_score <= 7.0:
        print("\\n*** Phase A1 목표 달성! ***")
    else:
        print(f"\\n추가 최적화 필요: {best_individual_score - 7.0:.4f}")
    
    return best_individual_score

if __name__ == "__main__":
    final_score = main()