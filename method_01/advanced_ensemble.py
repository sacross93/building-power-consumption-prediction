#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 앙상블 모델 - Phase A1: 다중 모델 + Optuna 최적화
목표: SMAPE 7.95 → 6.5~7.0 (1-1.5점 개선)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    """데이터 로드 및 기본 준비 (fast_improved.py와 동일)"""
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
    """method_05 핵심 기법 적용 (fast_improved.py와 동일)"""
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
    
    # 범주형 변수 처리
    categorical_cols = ['건물번호', '건물유형']
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    print(f"최종 피처 수: {len(feature_cols)}")
    print(f"범주형 피처: {[col for col in categorical_cols if col in feature_cols]}")
    
    return X_train, y_train, X_test, feature_cols, categorical_cols

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
    
    return X_tr, y_tr, X_val, y_val

class AdvancedEnsemble:
    """고급 앙상블 모델 클래스"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = {}
        self.models = {}
        self.model_scores = {}
        
    def optimize_lightgbm(self, X_tr, y_tr, X_val, y_val, categorical_features, n_trials=100):
        """LightGBM 하이퍼파라미터 최적화"""
        print(f"LightGBM 최적화 시작 (trials: {n_trials})...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': trial.suggest_int('num_leaves', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'verbosity': -1,
                'random_state': self.random_state,
                'n_estimators': 1000
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                categorical_feature=categorical_features,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)
            return smape(y_val.values, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['lgb'] = study.best_params
        self.model_scores['lgb'] = study.best_value
        
        print(f"LightGBM 최적화 완료 - Best SMAPE: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_xgboost(self, X_tr, y_tr, X_val, y_val, n_trials=100):
        """XGBoost 하이퍼파라미터 최적화"""
        print(f"XGBoost 최적화 시작 (trials: {n_trials})...")
        
        # XGBoost용 데이터 준비 (범주형 변수를 수치형으로 변환)
        X_tr_xgb = X_tr.copy()
        X_val_xgb = X_val.copy()
        
        for col in ['건물번호', '건물유형']:
            if col in X_tr_xgb.columns:
                if X_tr_xgb[col].dtype.name == 'category':
                    X_tr_xgb[col] = X_tr_xgb[col].cat.codes
                    X_val_xgb[col] = X_val_xgb[col].cat.codes
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': self.random_state,
                'verbosity': 0,
                'n_estimators': 1000
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr_xgb, y_tr,
                eval_set=[(X_val_xgb, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val_xgb)
            y_pred = np.maximum(y_pred, 0)
            return smape(y_val.values, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['xgb'] = study.best_params
        self.model_scores['xgb'] = study.best_value
        
        print(f"XGBoost 최적화 완료 - Best SMAPE: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_catboost(self, X_tr, y_tr, X_val, y_val, n_trials=100):
        """CatBoost 하이퍼파라미터 최적화"""
        print(f"CatBoost 최적화 시작 (trials: {n_trials})...")
        
        # CatBoost용 데이터 준비 (범주형 변수를 수치형으로 변환)
        X_tr_cat = X_tr.copy()
        X_val_cat = X_val.copy()
        
        for col in ['건물번호', '건물유형']:
            if col in X_tr_cat.columns:
                if X_tr_cat[col].dtype.name == 'category':
                    X_tr_cat[col] = X_tr_cat[col].cat.codes
                    X_val_cat[col] = X_val_cat[col].cat.codes
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'depth': trial.suggest_int('depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'random_seed': self.random_state,
                'verbose': False
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            else:
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_tr_cat, y_tr,
                eval_set=(X_val_cat, y_val),
                verbose=False
            )
            
            y_pred = model.predict(X_val_cat)
            y_pred = np.maximum(y_pred, 0)
            return smape(y_val.values, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['cat'] = study.best_params
        self.model_scores['cat'] = study.best_value
        
        print(f"CatBoost 최적화 완료 - Best SMAPE: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def train_final_models(self, X_train, y_train, X_test, categorical_features):
        """최적 파라미터로 최종 모델들 훈련"""
        print("최종 모델들 훈련 중...")
        
        # XGBoost와 CatBoost용 데이터 준비
        X_train_numeric = X_train.copy()
        X_test_numeric = X_test.copy()
        
        for col in ['건물번호', '건물유형']:
            if col in X_train_numeric.columns:
                if X_train_numeric[col].dtype.name == 'category':
                    X_train_numeric[col] = X_train_numeric[col].cat.codes
                    X_test_numeric[col] = X_test_numeric[col].cat.codes
        
        predictions = {}
        
        # LightGBM
        print("LightGBM 최종 훈련...")
        lgb_params = self.best_params['lgb'].copy()
        lgb_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_estimators': 3000
        })
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train, categorical_feature=categorical_features)
        predictions['lgb'] = np.maximum(lgb_model.predict(X_test), 0)
        self.models['lgb'] = lgb_model
        
        # XGBoost
        print("XGBoost 최종 훈련...")
        xgb_params = self.best_params['xgb'].copy()
        xgb_params.update({
            'random_state': self.random_state,
            'n_estimators': 3000
        })
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_numeric, y_train)
        predictions['xgb'] = np.maximum(xgb_model.predict(X_test_numeric), 0)
        self.models['xgb'] = xgb_model
        
        # CatBoost
        print("CatBoost 최종 훈련...")
        cat_params = self.best_params['cat'].copy()
        cat_params.update({
            'random_seed': self.random_state,
            'iterations': 3000,
            'verbose': False
        })
        
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train_numeric, y_train)
        predictions['cat'] = np.maximum(cat_model.predict(X_test_numeric), 0)
        self.models['cat'] = cat_model
        
        return predictions
    
    def create_weighted_ensemble(self, predictions):
        """성능 기반 가중 앙상블 생성"""
        print("가중 앙상블 생성...")
        
        # 역수 가중치 (낮은 SMAPE일수록 높은 가중치)
        total_weight = sum(1/score for score in self.model_scores.values())
        weights = {name: (1/score)/total_weight for name, score in self.model_scores.items()}
        
        ensemble_pred = np.zeros(len(predictions['lgb']))
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        print(f"앙상블 가중치:")
        for name, weight in weights.items():
            print(f"  {name.upper()}: {weight:.4f} (SMAPE: {self.model_scores[name]:.4f})")
        
        return ensemble_pred, weights

def main():
    print("고급 앙상블 모델 - Phase A1")
    print("=" * 60)
    print("목표: SMAPE 7.95 → 6.5~7.0 (1-1.5점 개선)")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 데이터 로드 및 피처 엔지니어링
    train_df, test_df = load_and_prepare_data()
    train_df, test_df = quick_feature_engineering(train_df, test_df)
    X_train, y_train, X_test, feature_cols, categorical_cols = prepare_features_for_modeling(train_df, test_df)
    
    # 2. Chronological split
    X_tr, y_tr, X_val, y_val = create_chronological_split(train_df, X_train, y_train)
    
    categorical_features = [col for col in categorical_cols if col in X_train.columns]
    
    # 3. 앙상블 모델 초기화 및 최적화
    ensemble = AdvancedEnsemble(random_state=42)
    
    # Optuna 최적화 (각 모델당 25 trials로 조정 - 빠른 테스트)
    ensemble.optimize_lightgbm(X_tr, y_tr, X_val, y_val, categorical_features, n_trials=25)
    ensemble.optimize_xgboost(X_tr, y_tr, X_val, y_val, n_trials=25)
    ensemble.optimize_catboost(X_tr, y_tr, X_val, y_val, n_trials=25)
    
    # 4. 최종 모델 훈련
    predictions = ensemble.train_final_models(X_train, y_train, X_test, categorical_features)
    
    # 5. 가중 앙상블 생성
    ensemble_pred, weights = ensemble.create_weighted_ensemble(predictions)
    
    # 6. 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': ensemble_pred
    })
    
    submission.to_csv('submission_advanced_ensemble.csv', index=False)
    
    # 7. 결과 출력
    elapsed_time = time.time() - start_time
    best_individual_score = min(ensemble.model_scores.values())
    
    print(f"\\n{'='*60}")
    print("Phase A1 완료 - 고급 앙상블 결과")
    print(f"{'='*60}")
    print(f"최고 개별 모델 성능: {best_individual_score:.4f}")
    print(f"모델별 성능:")
    for name, score in ensemble.model_scores.items():
        print(f"  {name.upper()}: {score:.4f}")
    
    print(f"\\n예측값 통계:")
    print(f"  범위: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f}")
    print(f"  평균: {ensemble_pred.mean():.2f}")
    print(f"  표준편차: {ensemble_pred.std():.2f}")
    
    print(f"\\n실행 시간: {elapsed_time/60:.1f}분")
    print(f"사용된 피처 수: {len(feature_cols)}")
    print("제출 파일: submission_advanced_ensemble.csv")
    
    if best_individual_score <= 7.0:
        print("\\n*** Phase A1 목표 달성! ***")
    else:
        print(f"\\n추가 최적화 필요: {best_individual_score - 7.0:.4f}")
    
    return best_individual_score

if __name__ == "__main__":
    final_score = main()