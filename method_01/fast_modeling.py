#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠르고 안정적인 전력사용량 예측 모델링 - SMAPE 6 이하 달성 목표
Author: Claude  
Date: 2025-07-27
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoostRegressor
import optuna
import warnings
import pickle
import os
import gc
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# GPU 지원 확인
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU 감지됨: {torch.cuda.get_device_name(0)}")
except ImportError:
    GPU_AVAILABLE = False
    print("PyTorch 없음 - GPU 사용 불가")

# SMAPE 계산 함수
def smape(y_true, y_pred, epsilon=1e-8):
    """SMAPE (Symmetric Mean Absolute Percentage Error) 계산"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 분모가 0에 가까운 경우 처리
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape_val

def safe_exp_transform(y_log):
    """안전한 exp 역변환"""
    # 너무 큰 값 클리핑하여 overflow 방지
    y_log_clipped = np.clip(y_log, -10, 10)
    return np.expm1(y_log_clipped)


class FastPowerPredictor:
    """빠르고 안정적인 전력소비량 예측 모델 클래스"""
    
    def __init__(self, use_gpu=True, random_state=42):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
        # 결과 저장 폴더
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        
        print(f"GPU 사용: {self.use_gpu}")
        
    def load_processed_data(self, data_version='iqr_log'):
        """전처리된 데이터 로드"""
        print(f"데이터 로드 중: {data_version}")
        
        # 전처리된 데이터 로드
        train_path = f'../processed_data/train_processed_{data_version}.csv'
        test_path = f'../processed_data/test_processed_{data_version}.csv'
        features_path = f'../processed_data/feature_columns_{data_version}.txt'
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # 피처 컬럼 로드
        with open(features_path, 'r', encoding='utf-8') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        print(f"Train: {self.train_df.shape}, Test: {self.test_df.shape}")
        print(f"Features: {len(self.feature_columns)}개")
        
        return self.train_df, self.test_df, self.feature_columns
    
    def prepare_features(self):
        """피처 준비"""
        print("피처 준비 중...")
        
        # 사용 가능한 피처만 선택
        available_features = [col for col in self.feature_columns if col in self.train_df.columns]
        
        self.X_train = self.train_df[available_features]
        self.y_train = self.train_df['power_transformed']  # log 변환된 target
        
        # Test 피처 (train과 동일한 순서로)
        test_available_features = [col for col in available_features if col in self.test_df.columns]
        self.X_test = self.test_df[test_available_features]
        
        print(f"Train features: {self.X_train.shape[1]}")
        print(f"Test features: {self.X_test.shape[1]}")
        
        # NaN 처리
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        return self.X_train, self.y_train, self.X_test
    
    def create_validation_split(self, test_ratio=0.15):
        """시계열 고려한 validation split 생성"""
        # 시간 순서대로 정렬
        if '일시' in self.train_df.columns:
            sorted_indices = self.train_df.sort_values('일시').index
        else:
            sorted_indices = self.train_df.index
        
        # 마지막 15%를 validation으로 사용
        split_idx = int(len(sorted_indices) * (1 - test_ratio))
        
        train_indices = sorted_indices[:split_idx]
        val_indices = sorted_indices[split_idx:]
        
        X_train_split = self.X_train.loc[train_indices]
        y_train_split = self.y_train.loc[train_indices]
        X_val_split = self.X_train.loc[val_indices]
        y_val_split = self.y_train.loc[val_indices]
        
        print(f"Train split: {len(X_train_split)}, Val split: {len(X_val_split)}")
        
        return X_train_split, X_val_split, y_train_split, y_val_split
    
    def optimize_lightgbm(self, n_trials=50):
        """LightGBM 하이퍼파라미터 최적화 (간단한 버전)"""
        print("LightGBM 최적화 시작...")
        
        X_train_split, X_val_split, y_train_split, y_val_split = self.create_validation_split()
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 31, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                'verbosity': -1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # GPU 설정 (선택사항)
            if self.use_gpu:
                try:
                    params.update({
                        'device_type': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                    })
                except:
                    # GPU 실패시 CPU로 fallback
                    pass
            
            try:
                # 간단한 train/validation
                model = lgb.LGBMRegressor(**params, n_estimators=1000)
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val_split)
                
                # SMAPE 계산 (log 역변환 후)
                y_val_original = safe_exp_transform(y_val_split.values)
                y_pred_original = safe_exp_transform(y_pred)
                y_pred_original = np.maximum(y_pred_original, 0)
                
                smape_score = smape(y_val_original, y_pred_original)
                
                return smape_score
                
            except Exception as e:
                print(f"Trial 에러: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30분 제한
        
        self.best_params['lightgbm'] = study.best_params
        self.cv_scores['lightgbm'] = study.best_value
        
        print(f"\nLightGBM 최적 SMAPE: {self.cv_scores['lightgbm']:.4f}")
        print(f"최적 파라미터: {self.best_params['lightgbm']}")
        
        return study.best_params
    
    def optimize_xgboost(self, n_trials=50):
        """XGBoost 하이퍼파라미터 최적화"""
        print("XGBoost 최적화 시작...")
        
        X_train_split, X_val_split, y_train_split, y_val_split = self.create_validation_split()
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': 1000,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # GPU 설정 (선택사항)
            if self.use_gpu:
                try:
                    params.update({
                        'tree_method': 'hist',
                        'device': 'cuda'
                    })
                except:
                    # GPU 실패시 CPU로 fallback
                    pass
            
            try:
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    early_stopping_rounds=100,
                    verbose=False
                )
                
                y_pred = model.predict(X_val_split)
                
                # SMAPE 계산 (log 역변환 후)
                y_val_original = safe_exp_transform(y_val_split.values)
                y_pred_original = safe_exp_transform(y_pred)
                y_pred_original = np.maximum(y_pred_original, 0)
                
                smape_score = smape(y_val_original, y_pred_original)
                
                return smape_score
                
            except Exception as e:
                print(f"XGBoost Trial 에러: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30분 제한
        
        self.best_params['xgboost'] = study.best_params
        self.cv_scores['xgboost'] = study.best_value
        
        print(f"\nXGBoost 최적 SMAPE: {self.cv_scores['xgboost']:.4f}")
        print(f"최적 파라미터: {self.best_params['xgboost']}")
        
        return study.best_params
    
    def optimize_catboost(self, n_trials=50):
        """CatBoost 하이퍼파라미터 최적화"""
        print("CatBoost 최적화 시작...")
        
        X_train_split, X_val_split, y_train_split, y_val_split = self.create_validation_split()
        
        def objective(trial):
            params = {
                'iterations': 1000,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
                'random_seed': self.random_state,
                'verbose': False,
                'early_stopping_rounds': 100
            }
            
            # GPU 설정 (선택사항)
            if self.use_gpu:
                try:
                    params.update({
                        'task_type': 'GPU',
                        'devices': '0'
                    })
                except:
                    # GPU 실패시 CPU로 fallback
                    pass
            
            try:
                model = CatBoostRegressor(**params)
                
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=(X_val_split, y_val_split),
                    verbose=False
                )
                
                y_pred = model.predict(X_val_split)
                
                # SMAPE 계산 (log 역변환 후)
                y_val_original = safe_exp_transform(y_val_split.values)
                y_pred_original = safe_exp_transform(y_pred)
                y_pred_original = np.maximum(y_pred_original, 0)
                
                smape_score = smape(y_val_original, y_pred_original)
                
                return smape_score
                
            except Exception as e:
                print(f"CatBoost Trial 에러: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30분 제한
        
        self.best_params['catboost'] = study.best_params
        self.cv_scores['catboost'] = study.best_value
        
        print(f"\nCatBoost 최적 SMAPE: {self.cv_scores['catboost']:.4f}")
        print(f"최적 파라미터: {self.best_params['catboost']}")
        
        return study.best_params
    
    def train_final_models(self):
        """최적 하이퍼파라미터로 최종 모델들 훈련"""
        print("최종 모델들 훈련 중...")
        
        # LightGBM
        if 'lightgbm' in self.best_params:
            print("LightGBM 최종 모델 훈련...")
            params = self.best_params['lightgbm'].copy()
            params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'n_estimators': 2000
            })
            
            if self.use_gpu:
                try:
                    params.update({
                        'device_type': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
                except:
                    pass
            
            self.models['lightgbm'] = lgb.LGBMRegressor(**params)
            self.models['lightgbm'].fit(self.X_train, self.y_train)
        
        # XGBoost
        if 'xgboost' in self.best_params:
            print("XGBoost 최종 모델 훈련...")
            params = self.best_params['xgboost'].copy()
            params.update({
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                'n_jobs': -1,
                'n_estimators': 2000
            })
            
            if self.use_gpu:
                try:
                    params.update({
                        'tree_method': 'hist',
                        'device': 'cuda'
                    })
                except:
                    pass
            
            self.models['xgboost'] = xgb.XGBRegressor(**params)
            self.models['xgboost'].fit(self.X_train, self.y_train)
        
        # CatBoost
        if 'catboost' in self.best_params:
            print("CatBoost 최종 모델 훈련...")
            params = self.best_params['catboost'].copy()
            params.update({
                'random_seed': self.random_state,
                'verbose': False,
                'iterations': 2000
            })
            
            if self.use_gpu:
                try:
                    params.update({
                        'task_type': 'GPU',
                        'devices': '0'
                    })
                except:
                    pass
            
            self.models['catboost'] = CatBoostRegressor(**params)
            self.models['catboost'].fit(self.X_train, self.y_train)
        
        print("모든 모델 훈련 완료!")
    
    def predict_ensemble(self, weights=None):
        """앙상블 예측"""
        print("앙상블 예측 수행 중...")
        
        predictions = {}
        
        # 각 모델로 예측
        for model_name, model in self.models.items():
            y_pred_log = model.predict(self.X_test)
            
            # Log 역변환
            y_pred_final = safe_exp_transform(y_pred_log)
            y_pred_final = np.maximum(y_pred_final, 0)
            
            predictions[model_name] = y_pred_final
            
            print(f"{model_name} 예측 완료 - 범위: {y_pred_final.min():.2f} ~ {y_pred_final.max():.2f}")
        
        # 앙상블 (가중평균)
        if weights is None:
            # 성능 기반 가중치 계산
            total_score = sum(1/score for score in self.cv_scores.values())
            weights = {name: (1/score)/total_score for name, score in self.cv_scores.items()}
        
        print(f"앙상블 가중치: {weights}")
        
        ensemble_pred = np.zeros(len(self.X_test))
        for model_name, weight in weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        self.predictions = ensemble_pred
        self.individual_predictions = predictions
        
        print(f"앙상블 예측 완료: {len(self.predictions)}개")
        print(f"예측값 범위: {self.predictions.min():.2f} ~ {self.predictions.max():.2f}")
        print(f"예측값 평균: {self.predictions.mean():.2f}")
        
        return self.predictions
    
    def create_submission(self, suffix=""):
        """제출 파일 생성"""
        if not hasattr(self, 'predictions'):
            print("예측이 먼저 수행되어야 합니다.")
            return None
        
        submission = pd.DataFrame({
            'num_date_time': self.test_df['num_date_time'],
            'answer': self.predictions
        })
        
        submission_path = f'submission{suffix}.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"제출 파일 생성: {submission_path}")
        return submission


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("빠른 전력사용량 예측 모델링 시작")
    print("목표: SMAPE ≤ 6.0")
    print("=" * 60)
    
    # 모델 초기화
    predictor = FastPowerPredictor(use_gpu=True, random_state=42)
    
    # 데이터 로드 - 세가지 버전 모두 테스트
    best_score = float('inf')
    best_version = None
    
    for data_version in ['none_log', 'iqr_log', 'building_percentile_log']:
        print(f"\n{'='*60}")
        print(f"데이터 버전 테스트: {data_version}")
        print(f"{'='*60}")
        
        try:
            # 데이터 로드
            predictor.load_processed_data(data_version)
            predictor.prepare_features()
            
            # LightGBM만 빠르게 테스트
            predictor.optimize_lightgbm(n_trials=30)
            
            current_score = predictor.cv_scores['lightgbm']
            print(f"{data_version} - SMAPE: {current_score:.4f}")
            
            if current_score < best_score:
                best_score = current_score
                best_version = data_version
                
        except Exception as e:
            print(f"{data_version} 실패: {e}")
            continue
    
    print(f"\n최고 성능 데이터 버전: {best_version} (SMAPE: {best_score:.4f})")
    
    # 최고 성능 버전으로 전체 모델 훈련
    print(f"\n{'='*60}")
    print(f"최고 성능 버전으로 전체 모델 최적화: {best_version}")
    print(f"{'='*60}")
    
    predictor = FastPowerPredictor(use_gpu=True, random_state=42)
    predictor.load_processed_data(best_version)
    predictor.prepare_features()
    
    # 모든 모델 최적화
    predictor.optimize_lightgbm(n_trials=100)
    predictor.optimize_xgboost(n_trials=100)
    predictor.optimize_catboost(n_trials=100)
    
    # 최종 모델 훈련
    predictor.train_final_models()
    
    # 앙상블 예측
    predictor.predict_ensemble()
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("최종 결과")
    print(f"{'='*60}")
    print("모델별 CV SMAPE:")
    for model_name, score in predictor.cv_scores.items():
        print(f"  {model_name}: {score:.4f}")
    
    best_individual_score = min(predictor.cv_scores.values())
    print(f"\n최고 개별 모델 성능: {best_individual_score:.4f}")
    
    if best_individual_score <= 6.0:
        print("🎉 목표 달성! SMAPE ≤ 6.0")
    else:
        print(f"목표까지 {best_individual_score - 6.0:.4f} 더 개선 필요")
    
    # 제출 파일 생성
    submission = predictor.create_submission(f"_{best_version}_final")
    
    print("\n모델링 완료!")
    print(f"제출 파일: submission_{best_version}_final.csv")


if __name__ == "__main__":
    main()