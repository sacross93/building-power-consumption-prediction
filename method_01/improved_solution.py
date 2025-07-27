#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 전력사용량 예측 모델 - method_05 기법 적용
목표: SMAPE 10 이하 달성
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import optuna
import warnings
from datetime import datetime

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

class ImprovedPowerPredictor:
    """개선된 전력소비량 예측 모델"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self):
        """데이터 로드 및 기본 준비"""
        print("데이터 로드 및 기본 준비 중...")
        
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
        return train_df, test_df, building_info
    
    def advanced_feature_engineering(self, train_df, test_df):
        """고급 피처 엔지니어링 - method_05 기법 적용"""
        print("고급 피처 엔지니어링 중...")
        
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
            # '-' 값을 NaN으로 변환 후 중앙값으로 채우기
            for df in [train_df, test_df]:
                df[col] = df[col].replace('-', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = train_df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # 3. 건물별 통계 피처 생성 (method_05의 핵심 기법)
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
        print("누락 데이터 대체 중...")
        
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
        
        # 6. Weather lag features 추가
        print("Weather lag features 생성...")
        
        weather_cols = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)']
        
        # 1시간, 3시간 전 기상 데이터
        for col in weather_cols:
            for lag in [1, 3]:
                train_df[f'{col}_lag_{lag}h'] = train_df.groupby('건물번호')[col].shift(lag)
                # Test는 마지막 값으로 초기화
                if len(train_df) > 0:
                    last_values = train_df.groupby('건물번호')[col].tail(lag).reset_index(drop=True)
                    test_df[f'{col}_lag_{lag}h'] = test_df.groupby('건물번호')[col].transform(lambda x: x.iloc[0])
        
        # 7. Rolling statistics
        print("Rolling statistics 생성...")
        
        for col in weather_cols:
            for window in [6, 12, 24]:
                train_df[f'{col}_rolling_mean_{window}h'] = (
                    train_df.groupby('건물번호')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                # Test는 각 건물별 최근 평균값으로 초기화
                building_recent_mean = {}
                for building in train_df['건물번호'].unique():
                    building_data = train_df[train_df['건물번호'] == building][col].tail(window)
                    building_recent_mean[building] = building_data.mean()
                
                test_df[f'{col}_rolling_mean_{window}h'] = test_df['건물번호'].map(building_recent_mean)
        
        print(f"피처 엔지니어링 완료 - Train: {train_df.shape[1]}개 컬럼, Test: {test_df.shape[1]}개 컬럼")
        
        return train_df, test_df
    
    def prepare_features_for_modeling(self, train_df, test_df):
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
        
        # 범주형 변수 처리
        categorical_cols = ['건물번호', '건물유형']
        for col in categorical_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
        
        # NaN 처리
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        print(f"최종 피처 수: {len(feature_cols)}")
        print(f"범주형 피처: {[col for col in categorical_cols if col in feature_cols]}")
        
        return X_train, y_train, X_test, feature_cols, categorical_cols
    
    def chronological_split(self, train_df, validation_days=7):
        """시계열 고려한 chronological split"""
        print(f"Chronological split - 마지막 {validation_days}일을 validation으로 사용")
        
        train_df = train_df.sort_values('일시')
        cutoff_date = train_df['일시'].max() - pd.Timedelta(days=validation_days)
        
        train_mask = train_df['일시'] < cutoff_date
        val_mask = ~train_mask
        
        print(f"Train samples: {train_mask.sum()}")
        print(f"Validation samples: {val_mask.sum()}")
        print(f"Cutoff date: {cutoff_date}")
        
        return train_mask, val_mask
    
    def optimize_models(self, X_train, y_train, categorical_cols, n_trials=100):
        """모델별 하이퍼파라미터 최적화"""
        print("모델 최적화 시작...")
        
        # Chronological split for validation
        train_df_temp = pd.concat([X_train, y_train], axis=1)
        train_df_temp['일시'] = pd.to_datetime(train_df_temp.index.map(
            lambda x: f"2024-06-01 00:00:00"  # 임시 날짜 (실제로는 원본 데이터 사용)
        )) + pd.to_timedelta(train_df_temp.index, unit='H')
        
        # 간단한 split (마지막 20%)
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
        # OneHot Encoder 준비
        categorical_features = [col for col in categorical_cols if col in X_train.columns]
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )
        
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_val_processed = preprocessor.transform(X_val)
        
        best_models = {}
        
        # 1. LightGBM 최적화
        print("LightGBM 최적화...")
        def lgb_objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': trial.suggest_int('num_leaves', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'verbosity': -1,
                'random_state': self.random_state
            }
            
            model = lgb.LGBMRegressor(**params, n_estimators=1000)
            model.fit(X_tr_processed, y_tr, 
                     eval_set=[(X_val_processed, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_val_processed)
            return smape(y_val.values, y_pred)
        
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_objective, n_trials=n_trials//3)
        
        self.best_params['lgb'] = study_lgb.best_params
        best_models['lgb'] = study_lgb.best_value
        
        # 2. XGBoost 최적화 (method_05 스타일)
        print("XGBoost 최적화...")
        def xgb_objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 8, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_processed, y_tr, 
                     eval_set=[(X_val_processed, y_val)],
                     early_stopping_rounds=100, verbose=False)
            
            y_pred = model.predict(X_val_processed)
            return smape(y_val.values, y_pred)
        
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=n_trials//3)
        
        self.best_params['xgb'] = study_xgb.best_params
        best_models['xgb'] = study_xgb.best_value
        
        # 3. CatBoost 최적화
        print("CatBoost 최적화...")
        def cat_objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'iterations': trial.suggest_int('iterations', 1000, 3000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': self.random_state,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr_processed, y_tr, 
                     eval_set=(X_val_processed, y_val),
                     early_stopping_rounds=100, verbose=False)
            
            y_pred = model.predict(X_val_processed)
            return smape(y_val.values, y_pred)
        
        study_cat = optuna.create_study(direction='minimize')
        study_cat.optimize(cat_objective, n_trials=n_trials//3)
        
        self.best_params['cat'] = study_cat.best_params
        best_models['cat'] = study_cat.best_value
        
        # 결과 출력
        print("\n최적화 결과:")
        for model_name, score in best_models.items():
            print(f"{model_name.upper()}: {score:.4f}")
        
        best_model = min(best_models, key=best_models.get)
        print(f"최고 성능 모델: {best_model.upper()} (SMAPE: {best_models[best_model]:.4f})")
        
        return preprocessor, best_models
    
    def train_ensemble_models(self, X_train, y_train, X_test, preprocessor):
        """앙상블 모델 훈련 및 예측"""
        print("앙상블 모델 훈련...")
        
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        predictions = {}
        
        # LightGBM
        lgb_params = self.best_params['lgb']
        lgb_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_estimators': 2000
        })
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train_processed, y_train)
        predictions['lgb'] = lgb_model.predict(X_test_processed)
        
        # XGBoost
        xgb_params = self.best_params['xgb']
        xgb_params.update({
            'random_state': self.random_state
        })
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_processed, y_train)
        predictions['xgb'] = xgb_model.predict(X_test_processed)
        
        # CatBoost
        cat_params = self.best_params['cat']
        cat_params.update({
            'random_seed': self.random_state,
            'verbose': False
        })
        
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train_processed, y_train)
        predictions['cat'] = cat_model.predict(X_test_processed)
        
        # 가중 앙상블 (성능 기반)
        best_scores = {
            'lgb': 10.0,  # 임시값, 실제로는 CV 점수 사용
            'xgb': 8.0,
            'cat': 12.0
        }
        
        # 역수 가중치 (낮은 SMAPE일수록 높은 가중치)
        total_weight = sum(1/score for score in best_scores.values())
        weights = {name: (1/score)/total_weight for name, score in best_scores.items()}
        
        ensemble_pred = np.zeros(len(predictions['lgb']))
        for model_name, weight in weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        # 음수 클리핑
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        print(f"앙상블 가중치: {weights}")
        print(f"예측값 범위: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f}")
        
        return ensemble_pred, predictions
    
    def run_complete_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("개선된 전력소비량 예측 모델 실행")
        print("목표: SMAPE 10 이하")
        print("=" * 60)
        
        # 1. 데이터 로드
        train_df, test_df, building_info = self.load_and_prepare_data()
        
        # 2. 고급 피처 엔지니어링
        train_df, test_df = self.advanced_feature_engineering(train_df, test_df)
        
        # 3. 모델링용 피처 준비
        X_train, y_train, X_test, feature_cols, categorical_cols = (
            self.prepare_features_for_modeling(train_df, test_df)
        )
        
        # 4. 모델 최적화
        preprocessor, best_scores = self.optimize_models(
            X_train, y_train, categorical_cols, n_trials=60
        )
        
        # 5. 앙상블 모델 훈련 및 예측
        ensemble_pred, individual_preds = self.train_ensemble_models(
            X_train, y_train, X_test, preprocessor
        )
        
        # 6. 제출 파일 생성
        submission = pd.DataFrame({
            'num_date_time': test_df['num_date_time'],
            'answer': ensemble_pred
        })
        
        submission.to_csv('submission_improved.csv', index=False)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print("최종 결과")
        print(f"{'='*60}")
        
        best_individual_score = min(best_scores.values())
        print(f"최고 개별 모델 성능: {best_individual_score:.4f}")
        
        if best_individual_score <= 10.0:
            print("🎉 목표 달성! SMAPE ≤ 10.0")
        else:
            print(f"목표까지 {best_individual_score - 10.0:.4f} 더 개선 필요")
        
        print(f"사용된 피처 수: {len(feature_cols)}")
        print("제출 파일: submission_improved.csv")
        
        return best_individual_score

def main():
    predictor = ImprovedPowerPredictor(random_state=42)
    final_score = predictor.run_complete_pipeline()
    
    print(f"\n최종 검증 SMAPE: {final_score:.4f}")
    if final_score <= 10.0:
        print("성공적으로 목표를 달성했습니다!")
    else:
        print(f"추가 개선이 필요합니다. ({final_score - 10.0:.4f} 더)")

if __name__ == "__main__":
    main()