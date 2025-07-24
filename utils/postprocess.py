#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
예측 후처리 및 Cold Start 문제 해결
Author: Claude
Date: 2025-07-24
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def restore_seasonality(predictions: np.ndarray, 
                       building_ids: np.ndarray,
                       hours: np.ndarray,
                       seasonal_patterns: Dict[int, Dict[int, float]]) -> np.ndarray:
    """
    계절성 복원 (시간대별 패턴 추가)
    
    Args:
        predictions: 예측값 (계절성 제거된 상태)
        building_ids: 건물 ID 배열
        hours: 시간 배열 (0-23)
        seasonal_patterns: 건물별 시간대별 평균 패턴
        
    Returns:
        계절성이 복원된 예측값
    """
    restored_predictions = predictions.copy()
    
    for i, (building_id, hour) in enumerate(zip(building_ids, hours)):
        if building_id in seasonal_patterns and hour in seasonal_patterns[building_id]:
            restored_predictions[i] += seasonal_patterns[building_id][hour]
            
    return restored_predictions

def extract_seasonal_patterns(train_df: pd.DataFrame,
                            building_col: str = '건물번호',
                            datetime_col: str = '일시', 
                            target_col: str = '전력소비량(kWh)') -> Dict[int, Dict[int, float]]:
    """
    훈련 데이터에서 건물별 시간대별 계절성 패턴 추출
    
    Args:
        train_df: 훈련 데이터
        building_col: 건물 컬럼명
        datetime_col: 날짜 컬럼명
        target_col: 타겟 컬럼명
        
    Returns:
        건물별 시간대별 평균 패턴 딕셔너리
    """
    seasonal_patterns = {}
    
    # 시간 추출
    if 'hour' not in train_df.columns:
        train_df['hour'] = pd.to_datetime(train_df[datetime_col]).dt.hour
    
    for building_id in train_df[building_col].unique():
        building_data = train_df[train_df[building_col] == building_id]
        hourly_patterns = building_data.groupby('hour')[target_col].mean().to_dict()
        seasonal_patterns[building_id] = hourly_patterns
        
    return seasonal_patterns

class ColdStartHandler:
    """Cold Start 문제 해결을 위한 클래스"""
    
    def __init__(self):
        self.last_values = {}  # 건물별 마지막 값들
        self.hourly_patterns = {}  # 건물별 시간대별 패턴
        self.trend_factors = {}  # 건물별 트렌드 요인
        
    def fit(self, train_df: pd.DataFrame, 
            building_col: str = '건물번호',
            datetime_col: str = '일시',
            target_col: str = '전력소비량(kWh)'):
        """
        훈련 데이터에서 Cold Start 해결을 위한 정보 추출
        
        Args:
            train_df: 훈련 데이터
            building_col: 건물 컬럼명
            datetime_col: 날짜 컬럼명
            target_col: 타겟 컬럼명
        """
        # 시간 순 정렬
        train_sorted = train_df.sort_values([building_col, datetime_col])
        
        for building_id in train_sorted[building_col].unique():
            building_data = train_sorted[train_sorted[building_col] == building_id]
            
            # 마지막 168시간(1주일) 값 저장
            last_168_values = building_data[target_col].tail(168).values
            self.last_values[building_id] = last_168_values
            
            # 시간대별 패턴 계산
            if 'hour' not in building_data.columns:
                building_data = building_data.copy()
                building_data['hour'] = pd.to_datetime(building_data[datetime_col]).dt.hour
            
            hourly_pattern = building_data.groupby('hour')[target_col].mean().to_dict()
            self.hourly_patterns[building_id] = hourly_pattern
            
            # 최근 7일간 트렌드 계산
            recent_week = building_data.tail(168)
            if len(recent_week) >= 168:
                early_week = recent_week.head(84)[target_col].mean()
                late_week = recent_week.tail(84)[target_col].mean()
                trend = (late_week - early_week) / early_week if early_week > 0 else 0
                self.trend_factors[building_id] = trend
            else:
                self.trend_factors[building_id] = 0
    
    def get_initial_features(self, test_df: pd.DataFrame,
                           building_col: str = '건물번호',
                           datetime_col: str = '일시') -> pd.DataFrame:
        """
        테스트 데이터의 초기 피처 생성 (시차 변수 대체)
        
        Args:
            test_df: 테스트 데이터
            building_col: 건물 컬럼명
            datetime_col: 날짜 컬럼명
            
        Returns:
            시차 변수가 채워진 테스트 데이터
        """
        test_with_features = test_df.copy()
        
        # 시간 정보 추가
        if 'hour' not in test_with_features.columns:
            test_with_features['hour'] = pd.to_datetime(test_with_features[datetime_col]).dt.hour
        
        # 시차 변수 초기화
        lag_columns = ['power_lag_1h', 'power_lag_24h', 'power_lag_168h',
                      'power_rolling_mean_24h', 'power_rolling_std_24h', 'power_rolling_mean_7d']
        
        for col in lag_columns:
            if col not in test_with_features.columns:
                test_with_features[col] = 0.0
        
        # 건물별로 초기값 설정
        for building_id in test_with_features[building_col].unique():
            building_mask = test_with_features[building_col] == building_id
            building_test = test_with_features[building_mask]
            
            if building_id in self.last_values:
                last_vals = self.last_values[building_id]
                
                # 각 시간대에 대해 초기값 설정
                for idx, (_, row) in enumerate(building_test.iterrows()):
                    hour = row['hour']
                    
                    # 1시간 전 값 (마지막 값 기준)
                    if len(last_vals) > 0:
                        test_with_features.loc[row.name, 'power_lag_1h'] = last_vals[-1]
                    
                    # 24시간 전 값 (같은 시간대 값)
                    if len(last_vals) >= 24:
                        same_hour_idx = -(24 - (hour % 24))
                        if abs(same_hour_idx) <= len(last_vals):
                            test_with_features.loc[row.name, 'power_lag_24h'] = last_vals[same_hour_idx]
                    
                    # 168시간 전 값 (1주일 전 같은 시간)
                    if len(last_vals) >= 168:
                        week_ago_idx = -(168 - (hour % 24))
                        if abs(week_ago_idx) <= len(last_vals):
                            test_with_features.loc[row.name, 'power_lag_168h'] = last_vals[week_ago_idx]
                    
                    # Rolling 통계 (최근 값들 기준)
                    if len(last_vals) >= 24:
                        test_with_features.loc[row.name, 'power_rolling_mean_24h'] = np.mean(last_vals[-24:])
                        test_with_features.loc[row.name, 'power_rolling_std_24h'] = np.std(last_vals[-24:])
                    
                    if len(last_vals) >= 168:
                        test_with_features.loc[row.name, 'power_rolling_mean_7d'] = np.mean(last_vals[-168:])
            
            # 패턴 기반 보정
            if building_id in self.hourly_patterns:
                for idx, (_, row) in enumerate(building_test.iterrows()):
                    hour = row['hour']
                    if hour in self.hourly_patterns[building_id]:
                        pattern_val = self.hourly_patterns[building_id][hour]
                        
                        # 패턴 값으로 보정
                        if test_with_features.loc[row.name, 'power_lag_1h'] == 0:
                            test_with_features.loc[row.name, 'power_lag_1h'] = pattern_val
                        if test_with_features.loc[row.name, 'power_lag_24h'] == 0:
                            test_with_features.loc[row.name, 'power_lag_24h'] = pattern_val
                        if test_with_features.loc[row.name, 'power_lag_168h'] == 0:
                            test_with_features.loc[row.name, 'power_lag_168h'] = pattern_val
        
        return test_with_features

def post_process_predictions(predictions: np.ndarray,
                           building_ids: np.ndarray,
                           train_stats: Dict[int, Dict[str, float]],
                           apply_log_transform: bool = True,
                           clip_negative: bool = True,
                           apply_smoothing: bool = False) -> np.ndarray:
    """
    예측값 후처리 파이프라인
    
    Args:
        predictions: 원시 예측값
        building_ids: 건물 ID 배열
        train_stats: 건물별 훈련 통계 (min, max, mean, std)
        apply_log_transform: log 역변환 적용 여부
        clip_negative: 음수 클리핑 여부
        apply_smoothing: 평활화 적용 여부
        
    Returns:
        후처리된 예측값
    """
    processed_preds = predictions.copy()
    
    # 1. Log 역변환
    if apply_log_transform:
        processed_preds = np.expm1(processed_preds)
    
    # 2. 음수 클리핑
    if clip_negative:
        processed_preds = np.maximum(processed_preds, 0)
    
    # 3. 건물별 범위 제한
    for i, building_id in enumerate(building_ids):
        if building_id in train_stats:
            stats = train_stats[building_id]
            
            # 극단적 값 클리핑 (평균 ± 4 * 표준편차)
            upper_bound = stats['mean'] + 4 * stats['std']
            lower_bound = max(0, stats['mean'] - 2 * stats['std'])
            
            processed_preds[i] = np.clip(processed_preds[i], lower_bound, upper_bound)
    
    # 4. 평활화 (선택사항)
    if apply_smoothing:
        processed_preds = apply_temporal_smoothing(processed_preds, building_ids)
    
    return processed_preds

def apply_temporal_smoothing(predictions: np.ndarray, 
                           building_ids: np.ndarray,
                           window_size: int = 3) -> np.ndarray:
    """
    시간적 평활화 적용 (각 건물별로)
    
    Args:
        predictions: 예측값
        building_ids: 건물 ID 배열
        window_size: 평활화 윈도우 크기
        
    Returns:
        평활화된 예측값
    """
    smoothed_preds = predictions.copy()
    
    # 건물별로 평활화
    for building_id in np.unique(building_ids):
        mask = building_ids == building_id
        building_preds = predictions[mask]
        
        # 이동 평균 적용
        if len(building_preds) >= window_size:
            for i in range(window_size//2, len(building_preds) - window_size//2):
                window_start = i - window_size//2
                window_end = i + window_size//2 + 1
                smoothed_preds[mask][i] = np.mean(building_preds[window_start:window_end])
    
    return smoothed_preds

def calculate_train_stats(train_df: pd.DataFrame,
                        building_col: str = '건물번호',
                        target_col: str = '전력소비량(kWh)') -> Dict[int, Dict[str, float]]:
    """
    훈련 데이터에서 건물별 통계 계산
    
    Args:
        train_df: 훈련 데이터
        building_col: 건물 컬럼명
        target_col: 타겟 컬럼명
        
    Returns:
        건물별 통계 딕셔너리
    """
    train_stats = {}
    
    for building_id in train_df[building_col].unique():
        building_data = train_df[train_df[building_col] == building_id][target_col]
        
        train_stats[building_id] = {
            'mean': building_data.mean(),
            'std': building_data.std(),
            'min': building_data.min(),
            'max': building_data.max(),
            'median': building_data.median(),
            'q25': building_data.quantile(0.25),
            'q75': building_data.quantile(0.75)
        }
    
    return train_stats

if __name__ == "__main__":
    # 테스트 코드
    print("후처리 유틸리티 테스트")
    
    # 더미 데이터로 테스트
    predictions = np.array([5.2, 6.1, 4.8, 0.1, -0.5])
    building_ids = np.array([1, 1, 1, 2, 2])
    
    train_stats = {
        1: {'mean': 5.0, 'std': 1.0, 'min': 2.0, 'max': 8.0},
        2: {'mean': 2.0, 'std': 0.5, 'min': 0.5, 'max': 4.0}
    }
    
    processed = post_process_predictions(
        predictions, building_ids, train_stats,
        apply_log_transform=False, clip_negative=True
    )
    
    print(f"원본 예측값: {predictions}")
    print(f"후처리 예측값: {processed}")