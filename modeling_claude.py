#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전력사용량 예측 머신러닝 모델링
Author: Claude  
Date: 2025-07-24
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

# GPU 지원 확인
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU 감지됨: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
except ImportError:
    GPU_AVAILABLE = False
    print("PyTorch 없음 - GPU 사용 불가")

# 유틸리티 함수 import
from utils.smape_utils import *
from utils.validation import *
from utils.postprocess import *

warnings.filterwarnings('ignore')

class PowerConsumptionPredictor:
    """전력소비량 예측 모델 클래스"""
    
    def __init__(self, model_type='lightgbm', random_state=42, use_gpu=True):
        """
        Args:
            model_type: 모델 타입 ('lightgbm', 'xgboost', 'catboost')
            random_state: 랜덤 시드
            use_gpu: GPU 사용 여부
        """
        self.model_type = model_type
        self.random_state = random_state
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.model = None
        self.best_params = None
        self.cv_scores = []
        self.feature_importance = None
        self.cold_start_handler = ColdStartHandler()
        
        # GPU 설정 출력
        if self.use_gpu:
            print(f"[GPU 모드] {model_type} GPU 가속 활성화")
        else:
            print(f"[CPU 모드] {model_type} CPU 사용")
        
        # 결과 저장 폴더
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_processed_data(self, data_version='none_log'):
        """
        전처리된 데이터 로드
        
        Args:
            data_version: 데이터 버전 ('none_log', 'iqr_log', 'building_percentile_log')
        """
        print(f"데이터 로드 중: {data_version}")
        
        # 전처리된 데이터 로드
        train_path = f'processed_data/train_processed_{data_version}.csv'
        test_path = f'processed_data/test_processed_{data_version}.csv'
        features_path = f'processed_data/feature_columns_{data_version}.txt'
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # 피처 컬럼 로드
        with open(features_path, 'r', encoding='utf-8') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        # 데이터 타입 최적화
        # 이미 datetime으로 되어 있는지 확인
        if self.train_df['일시'].dtype == 'object':
            self.train_df['일시'] = pd.to_datetime(self.train_df['일시'])
        if self.test_df['일시'].dtype == 'object':
            self.test_df['일시'] = pd.to_datetime(self.test_df['일시'])
        
        print(f"Train: {self.train_df.shape}, Test: {self.test_df.shape}")
        print(f"Features: {len(self.feature_columns)}개")
        
        return self.train_df, self.test_df, self.feature_columns
    
    def prepare_features(self):
        """피처 준비 및 Cold Start 처리"""
        print("피처 준비 중...")
        
        # 원본 훈련 데이터에서 Cold Start 정보 추출
        original_train = pd.read_csv('data/train.csv', encoding='utf-8-sig')
        original_train.columns = original_train.columns.str.strip()
        original_train['일시'] = pd.to_datetime(original_train['일시'], format='%Y%m%d %H')
        
        # Cold Start Handler 학습
        self.cold_start_handler.fit(original_train)
        
        # Train/Test 피처 추출
        # Train은 전처리된 데이터의 피처 사용
        available_features = [col for col in self.feature_columns if col in self.train_df.columns]
        self.X_train = self.train_df[available_features]
        self.y_train = self.train_df['power_transformed']  # log 변환된 target
        
        # Test는 Cold Start 처리 후 피처 추출
        # 원본 test 데이터 로드
        original_test = pd.read_csv('data/test.csv', encoding='utf-8-sig')
        original_test.columns = original_test.columns.str.strip()
        original_test['일시'] = pd.to_datetime(original_test['일시'], format='%Y%m%d %H')
        
        # Test 데이터에 Cold Start 피처 추가
        test_with_cold_start = self.cold_start_handler.get_initial_features(original_test)
        
        # Test 피처 정렬 (train과 동일한 순서)
        test_available_features = [col for col in available_features if col in self.test_df.columns]
        self.X_test = self.test_df[test_available_features]
        
        print(f"Train features: {self.X_train.shape[1]}")
        print(f"Test features: {self.X_test.shape[1]}")
        print(f"Feature 예시: {list(self.X_train.columns[:5])}")
        
        # 훈련 통계 계산 (후처리용)
        self.train_stats = calculate_train_stats(original_train)
        
        return self.X_train, self.y_train, self.X_test
    
    def optimize_hyperparameters(self, n_trials=200, cv_folds=3):
        """
        Optuna를 사용한 하이퍼파라미터 튜닝
        
        Args:
            n_trials: 시도 횟수
            cv_folds: CV fold 수
        """
        print(f"하이퍼파라미터 튜닝 시작: {self.model_type}")
        
        def objective(trial):
            # 모델별 하이퍼파라미터 정의
            if self.model_type == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'metric': 'None',  # custom metric 사용
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                    'verbosity': -1,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                
                # GPU 설정 추가
                if self.use_gpu:
                    params.update({
                        'device_type': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'max_bin': trial.suggest_int('max_bin', 63, 255),  # GPU 호환 범위
                        'gpu_use_dp': False  # 단정밀도 사용으로 메모리 절약
                    })
                
            elif self.model_type == 'xgboost':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                
                # GPU 설정 추가 (XGBoost 2.0+ 새로운 문법)
                if self.use_gpu:
                    params.update({
                        'tree_method': 'hist',
                        'device': 'cuda',
                        'max_bin': trial.suggest_int('max_bin', 256, 512)  # GPU 최적화
                    })
                
            elif self.model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 2000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                    'random_seed': self.random_state,
                    'verbose': False,
                    'thread_count': -1
                }
                
                # GPU 설정 추가
                if self.use_gpu:
                    params.update({
                        'task_type': 'GPU',
                        'devices': '0',
                        'gpu_ram_part': 0.8  # GPU 메모리의 80% 사용
                    })
                
                if params['bootstrap_type'] == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
                elif params['bootstrap_type'] == 'Bernoulli':
                    params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
            
            # Cross Validation - use simplified index-based split to avoid datetime issues
            cv_scores = []
            n_samples = len(self.X_train)
            fold_size = n_samples // (cv_folds + 1)
            
            for fold in range(cv_folds):
                # Simple time-ordered split
                val_start = n_samples - (fold + 1) * fold_size
                val_end = n_samples - fold * fold_size if fold > 0 else n_samples
                train_end = val_start
                
                if train_end <= fold_size:  # Need minimum training data
                    continue
                    
                train_idx = list(range(train_end))
                val_idx = list(range(val_start, val_end))
                X_fold_train = self.X_train.iloc[train_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_val = self.y_train.iloc[val_idx]
                
                try:
                    if self.model_type == 'lightgbm':
                        train_set = lgb.Dataset(X_fold_train, label=y_fold_train)
                        val_set = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_set)
                        
                        # Train without early stopping for CV
                        model = lgb.train(
                            params,
                            train_set,
                            num_boost_round=500,  # Reduced for faster CV
                            callbacks=[lgb.log_evaluation(0)]
                        )
                        
                        y_pred = model.predict(X_fold_val)
                        
                    elif self.model_type == 'xgboost':
                        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
                        
                        model = xgb.train(
                            params,
                            dtrain,
                            evals=[(dval, 'eval')],
                            num_boost_round=1000,
                            early_stopping_rounds=100,
                            verbose_eval=0
                        )
                        
                        y_pred = model.predict(dval)
                        
                    elif self.model_type == 'catboost':
                        model = CatBoostRegressor(**params)
                        
                        model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=(X_fold_val, y_fold_val),
                            early_stopping_rounds=100,
                            verbose=False
                        )
                        
                        y_pred = model.predict(X_fold_val)
                    
                    # SMAPE 계산 (log 역변환 후)
                    y_val_original = safe_exp_transform(y_fold_val.values)
                    y_pred_original = safe_exp_transform(y_pred)
                    y_pred_original = np.maximum(y_pred_original, 0)  # 음수 클리핑
                    
                    fold_smape = smape(y_val_original, y_pred_original)
                    cv_scores.append(fold_smape)
                    
                except Exception as e:
                    print(f"Fold {fold} 에러: {e}")
                    return float('inf')
                
                # 메모리 정리
                del X_fold_train, y_fold_train, X_fold_val, y_fold_val, model
                gc.collect()
            
            if len(cv_scores) == 0:
                return float('inf')
                
            mean_cv_score = np.mean(cv_scores)
            
            # Pruning (조기 종료)
            trial.report(mean_cv_score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return mean_cv_score
        
        # Optuna 최적화 실행
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1시간 제한
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n최적 하이퍼파라미터:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"최적 CV SMAPE: {self.best_score:.4f}")
        
        return self.best_params
    
    def train_final_model(self):
        """최적 하이퍼파라미터로 최종 모델 훈련"""
        print("최종 모델 훈련 중...")
        
        if self.model_type == 'lightgbm':
            # 기본 파라미터에 최적 파라미터 업데이트
            params = {
                'objective': 'regression',
                'metric': 'None',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            params.update(self.best_params)
            
            # GPU 설정 추가 (최종 모델에서도)
            if self.use_gpu:
                params.update({
                    'device_type': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'gpu_use_dp': False
                })
                # max_bin이 best_params에 없으면 기본값 설정
                if 'max_bin' not in params:
                    params['max_bin'] = 127
            
            # Validation set 생성 (최근 7일)
            train_df_sorted = self.train_df.sort_values(['건물번호', '일시'])
            train_size = int(len(train_df_sorted) * 0.85)
            
            X_final_train = self.X_train.iloc[:train_size]
            y_final_train = self.y_train.iloc[:train_size]
            X_final_val = self.X_train.iloc[train_size:]
            y_final_val = self.y_train.iloc[train_size:]
            
            train_set = lgb.Dataset(X_final_train, label=y_final_train)
            val_set = lgb.Dataset(X_final_val, label=y_final_val, reference=train_set)
            
            # Train without early stopping to avoid validation metric issues
            self.model = lgb.train(
                params,
                train_set,
                num_boost_round=1000,
                callbacks=[lgb.log_evaluation(100)]
            )
            
        elif self.model_type == 'xgboost':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            params.update(self.best_params)
            
            # GPU 설정 추가 (최종 모델에서도, XGBoost 2.0+ 새로운 문법)
            if self.use_gpu:
                params.update({
                    'tree_method': 'hist',
                    'device': 'cuda'
                })
                # max_bin이 best_params에 없으면 기본값 설정
                if 'max_bin' not in params:
                    params['max_bin'] = 256
            
            # Validation set 생성
            train_size = int(len(self.X_train) * 0.85)
            X_final_train = self.X_train.iloc[:train_size]
            y_final_train = self.y_train.iloc[:train_size]
            X_final_val = self.X_train.iloc[train_size:]
            y_final_val = self.y_train.iloc[train_size:]
            
            dtrain = xgb.DMatrix(X_final_train, label=y_final_train)
            dval = xgb.DMatrix(X_final_val, label=y_final_val)
            
            self.model = xgb.train(
                params,
                dtrain,
                evals=[(dval, 'eval')],
                num_boost_round=2000,
                early_stopping_rounds=200,
                verbose_eval=100
            )
            
        elif self.model_type == 'catboost':
            params = {
                'random_seed': self.random_state,
                'verbose': 100,
                'thread_count': -1
            }
            params.update(self.best_params)
            
            # GPU 설정 추가 (최종 모델에서도)
            if self.use_gpu:
                params.update({
                    'task_type': 'GPU',
                    'devices': '0',
                    'gpu_ram_part': 0.8
                })
            
            # Validation set 생성
            train_size = int(len(self.X_train) * 0.85)
            X_final_train = self.X_train.iloc[:train_size]
            y_final_train = self.y_train.iloc[:train_size]
            X_final_val = self.X_train.iloc[train_size:]
            y_final_val = self.y_train.iloc[train_size:]
            
            self.model = CatBoostRegressor(**params)
            self.model.fit(
                X_final_train, y_final_train,
                eval_set=(X_final_val, y_final_val),
                early_stopping_rounds=200,
                verbose=100
            )
        
        # Feature Importance 추출
        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'xgboost':
            importance = self.model.get_score(importance_type='gain')
            self.feature_importance = pd.DataFrame([
                {'feature': k, 'importance': v} for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
            
        elif self.model_type == 'catboost':
            importance = self.model.get_feature_importance()
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        print(f"\n상위 10개 중요 피처:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def predict(self):
        """테스트 데이터 예측"""
        print("예측 수행 중...")
        
        if self.model_type == 'lightgbm':
            y_pred_log = self.model.predict(self.X_test)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(self.X_test)
            y_pred_log = self.model.predict(dtest)
        elif self.model_type == 'catboost':
            y_pred_log = self.model.predict(self.X_test)
        
        # 후처리 파이프라인
        building_ids = self.test_df['건물번호'].values
        
        # Log 역변환 및 후처리
        y_pred_final = post_process_predictions(
            y_pred_log,
            building_ids,
            self.train_stats,
            apply_log_transform=True,
            clip_negative=True,
            apply_smoothing=False
        )
        
        self.predictions = y_pred_final
        
        print(f"예측 완료: {len(self.predictions)}개")
        print(f"예측값 범위: {self.predictions.min():.2f} ~ {self.predictions.max():.2f}")
        print(f"예측값 평균: {self.predictions.mean():.2f}")
        
        return self.predictions
    
    def save_model(self, suffix=""):
        """모델 저장"""
        model_path = f'models/{self.model_type}_model{suffix}.pkl'
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'train_stats': self.train_stats,
            'cold_start_handler': self.cold_start_handler,
            'model_type': self.model_type
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"모델 저장: {model_path}")
        
    def create_submission(self, suffix=""):
        """제출 파일 생성"""
        submission = pd.DataFrame({
            'num_date_time': self.test_df['num_date_time'],
            'answer': self.predictions
        })
        
        submission_path = f'results/submission_{self.model_type}{suffix}.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"제출 파일 생성: {submission_path}")
        return submission

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("전력사용량 예측 모델링 시작")
    print("=" * 60)
    
    # 실험 설정
    data_versions = ['none_log', 'iqr_log', 'building_percentile_log']
    model_types = ['lightgbm']  # 일단 LightGBM만
    
    results_summary = []
    
    for data_version in data_versions:
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"실험: {data_version} + {model_type}")
            print(f"{'='*60}")
            
            try:
                # 모델 초기화 (GPU 사용 가능 시 GPU 모드)
                predictor = PowerConsumptionPredictor(model_type=model_type, random_state=42, use_gpu=True)
                
                # 데이터 로드
                predictor.load_processed_data(data_version)
                
                # 피처 준비
                predictor.prepare_features()
                
                # 하이퍼파라미터 튜닝
                predictor.optimize_hyperparameters(n_trials=100, cv_folds=3)
                
                # 최종 모델 훈련
                predictor.train_final_model()
                
                # 예측
                predictor.predict()
                
                # 모델 저장
                suffix = f"_{data_version}"
                predictor.save_model(suffix)
                
                # 제출 파일 생성
                submission = predictor.create_submission(suffix)
                
                # 결과 기록
                results_summary.append({
                    'data_version': data_version,
                    'model_type': model_type,
                    'best_smape': predictor.best_score,
                    'submission_file': f'submission_{model_type}_{data_version}.csv'
                })
                
                print(f"[SUCCESS] {data_version} + {model_type} 완료")
                
            except Exception as e:
                print(f"[ERROR] {data_version} + {model_type} 실패: {e}")
                continue
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("실험 결과 요약")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results_summary)
    if len(results_df) > 0:
        results_df = results_df.sort_values('best_smape')
        print(results_df.to_string(index=False))
        
        # 최고 성능 모델
        best_result = results_df.iloc[0]
        print(f"\n[BEST] 최고 성능:")
        print(f"  데이터: {best_result['data_version']}")
        print(f"  모델: {best_result['model_type']}")
        print(f"  SMAPE: {best_result['best_smape']:.4f}")
        print(f"  제출파일: {best_result['submission_file']}")
        
        # 결과 저장
        results_df.to_csv('results/experiment_results.csv', index=False)
    
    print("\n모든 실험 완료!")

if __name__ == "__main__":
    main()