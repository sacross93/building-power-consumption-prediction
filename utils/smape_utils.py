#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMAPE (Symmetric Mean Absolute Percentage Error) 관련 유틸리티
Author: Claude
Date: 2025-07-24
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

def smape(y_true, y_pred, epsilon=1e-8):
    """
    SMAPE (Symmetric Mean Absolute Percentage Error) 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값  
        epsilon: 0으로 나누기 방지를 위한 작은 값
    
    Returns:
        SMAPE 점수 (0-100 범위)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 분모가 0에 가까운 경우 처리
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    return smape_val

def smape_loss_lgb(y_pred, y_true):
    """
    LightGBM용 SMAPE custom loss function
    
    Args:
        y_pred: 예측값
        y_true: LightGBM Dataset의 실제값
        
    Returns:
        grad, hess: gradient와 hessian
    """
    y_true = y_true.get_label()
    
    epsilon = 1e-8
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    
    # SMAPE의 gradient 계산
    grad = np.where(y_pred >= y_true,
                   1.0 / denominator,
                   -1.0 / denominator)
    
    # Hessian은 0으로 근사 (2차 도함수가 복잡함)
    hess = np.ones_like(y_pred) * 1e-6
    
    return grad, hess

def smape_eval_lgb(y_pred, y_true):
    """
    LightGBM용 SMAPE evaluation metric
    
    Args:
        y_pred: 예측값
        y_true: LightGBM Dataset의 실제값
        
    Returns:
        eval_name, eval_result, is_higher_better
    """
    y_true = y_true.get_label()
    smape_val = smape(y_true, y_pred)
    
    return 'smape', smape_val, False  # False: 낮을수록 좋음

def smape_loss_xgb(y_pred, y_true):
    """
    XGBoost용 SMAPE custom loss function
    
    Args:
        y_pred: 예측값
        y_true: 실제값
        
    Returns:
        grad, hess: gradient와 hessian
    """
    epsilon = 1e-8
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    
    # SMAPE의 gradient 계산
    grad = np.where(y_pred >= y_true,
                   1.0 / denominator,
                   -1.0 / denominator)
    
    # Hessian은 0으로 근사
    hess = np.ones_like(y_pred) * 1e-6
    
    return grad, hess

def smape_eval_xgb(y_pred, y_true):
    """
    XGBoost용 SMAPE evaluation metric
    """
    smape_val = smape(y_true, y_pred)
    return 'smape', smape_val

def safe_log_transform(y, epsilon=1e-8):
    """
    안전한 log 변환 (음수값 처리)
    
    Args:
        y: 변환할 값
        epsilon: 최소값
        
    Returns:
        log 변환된 값
    """
    y = np.maximum(y, epsilon)
    return np.log1p(y)

def safe_exp_transform(y_log):
    """
    안전한 exp 역변환
    
    Args:
        y_log: log 변환된 값
        
    Returns:
        원본 스케일로 복원된 값
    """
    return np.expm1(y_log)

def clip_predictions(y_pred, y_train_min=0, y_train_max=None):
    """
    예측값을 합리적 범위로 클리핑
    
    Args:
        y_pred: 예측값
        y_train_min: 최소값 (기본: 0)
        y_train_max: 최대값 (None이면 클리핑 안함)
        
    Returns:
        클리핑된 예측값
    """
    y_pred_clipped = np.maximum(y_pred, y_train_min)
    
    if y_train_max is not None:
        y_pred_clipped = np.minimum(y_pred_clipped, y_train_max)
    
    return y_pred_clipped

def evaluate_predictions(y_true, y_pred, prefix=""):
    """
    예측 결과에 대한 종합적 평가
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        prefix: 출력 접두사
        
    Returns:
        평가 결과 딕셔너리
    """
    # 음수 예측값 처리
    y_pred_positive = np.maximum(y_pred, 0)
    
    results = {
        'smape': smape(y_true, y_pred_positive),
        'mae': mean_absolute_error(y_true, y_pred_positive),
        'mape': np.mean(np.abs((y_true - y_pred_positive) / (y_true + 1e-8))) * 100,
        'negative_predictions': (y_pred < 0).sum(),
        'zero_predictions': (y_pred_positive == 0).sum(),
        'mean_pred': np.mean(y_pred_positive),
        'std_pred': np.std(y_pred_positive),
        'min_pred': np.min(y_pred_positive),
        'max_pred': np.max(y_pred_positive)
    }
    
    print(f"\n{prefix} 예측 결과 평가:")
    print(f"SMAPE: {results['smape']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MAPE: {results['mape']:.4f}")
    print(f"음수 예측: {results['negative_predictions']}개")
    print(f"0 예측: {results['zero_predictions']}개")
    print(f"예측값 범위: {results['min_pred']:.2f} ~ {results['max_pred']:.2f}")
    print(f"예측값 평균: {results['mean_pred']:.2f} ± {results['std_pred']:.2f}")
    
    return results

def calculate_building_smape(y_true, y_pred, building_ids):
    """
    건물별 SMAPE 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        building_ids: 건물 ID
        
    Returns:
        건물별 SMAPE DataFrame
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'building_id': building_ids
    })
    
    building_smape = []
    for building_id in df['building_id'].unique():
        building_data = df[df['building_id'] == building_id]
        building_smape_val = smape(building_data['y_true'], building_data['y_pred'])
        building_smape.append({
            'building_id': building_id,
            'smape': building_smape_val,
            'count': len(building_data)
        })
    
    building_smape_df = pd.DataFrame(building_smape)
    building_smape_df = building_smape_df.sort_values('smape', ascending=False)
    
    print(f"\n건물별 SMAPE (상위 10개):")
    print(building_smape_df.head(10).to_string(index=False))
    print(f"\n전체 평균 SMAPE: {building_smape_df['smape'].mean():.4f}")
    print(f"SMAPE 표준편차: {building_smape_df['smape'].std():.4f}")
    
    return building_smape_df

if __name__ == "__main__":
    # 테스트 코드
    y_true = np.array([10, 20, 30, 0.1, 0.01])
    y_pred = np.array([12, 18, 35, 0.05, 0.02])
    
    print("SMAPE 테스트:")
    print(f"SMAPE: {smape(y_true, y_pred):.4f}")
    
    # 평가 함수 테스트
    evaluate_predictions(y_true, y_pred, "테스트")