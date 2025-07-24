#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시계열 데이터를 위한 검증 전략
Author: Claude
Date: 2025-07-24
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from datetime import datetime, timedelta

class TimeSeriesSplitBuilding(BaseCrossValidator):
    """
    건물별 시계열 데이터를 위한 교차검증 분할기
    각 건물의 시간 순서를 보존하면서 분할
    """
    
    def __init__(self, n_splits=5, test_size_days=7, gap_days=1):
        """
        Args:
            n_splits: 분할 수
            test_size_days: 테스트 기간 (일)
            gap_days: train-test 사이 간격 (일)
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        
    def split(self, X, y=None, groups=None):
        """
        데이터를 시간 순서를 보존하면서 분할
        
        Args:
            X: DataFrame with datetime column
            y: target (not used)
            groups: building_id column
            
        Yields:
            train_indices, test_indices
        """
        if not hasattr(X, 'reset_index'):
            X = pd.DataFrame(X)
            
        # 날짜 컬럼 찾기
        datetime_col = None
        for col in X.columns:
            if '일시' in str(col) or 'datetime' in str(col).lower() or 'date' in str(col).lower():
                datetime_col = col
                break
                
        if datetime_col is None:
            raise ValueError("날짜/시간 컬럼을 찾을 수 없습니다")
        
        # 건물별로 분할
        building_col = None
        for col in X.columns:
            if '건물' in str(col) or 'building' in str(col).lower():
                building_col = col
                break
        
        if building_col is None:
            # 건물별 분할이 불가능한 경우 전체 데이터로 시계열 분할
            return self._split_global(X, datetime_col)
        else:
            # 건물별 분할
            return self._split_by_building(X, datetime_col, building_col)
    
    def _split_global(self, X, datetime_col):
        """전체 데이터에 대한 시계열 분할"""
        # 시간 순 정렬
        X_sorted = X.sort_values(datetime_col)
        # 이미 datetime 타입인지 확인
        try:
            if X_sorted[datetime_col].dtype == 'object':
                datetime_series = pd.to_datetime(X_sorted[datetime_col])
            else:
                datetime_series = X_sorted[datetime_col]
            unique_dates = datetime_series.dt.date.unique()
        except Exception as e:
            print(f"날짜 변환 에러 (global): {e}")
            # fallback: 인덱스 기반 분할
            return self._split_by_index(X_sorted)
        unique_dates = sorted(unique_dates)
        
        total_days = len(unique_dates)
        
        for i in range(self.n_splits):
            # 테스트 기간 계산
            test_end_idx = total_days - i * (self.test_size_days + self.gap_days)
            test_start_idx = test_end_idx - self.test_size_days
            train_end_idx = test_start_idx - self.gap_days
            
            if train_end_idx <= 0:
                continue
                
            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]
            
            # 인덱스 찾기
            train_mask = pd.to_datetime(X_sorted[datetime_col]).dt.date.isin(train_dates)
            test_mask = pd.to_datetime(X_sorted[datetime_col]).dt.date.isin(test_dates)
            
            train_indices = X_sorted[train_mask].index.tolist()
            test_indices = X_sorted[test_mask].index.tolist()
            
            yield train_indices, test_indices
    
    def _split_by_building(self, X, datetime_col, building_col):
        """건물별 시계열 분할"""
        # 각 건물별로 시간 순 정렬
        X_sorted = X.sort_values([building_col, datetime_col])
        
        # 전체 날짜 범위 계산
        # 이미 datetime 타입인지 확인
        try:
            if X_sorted[datetime_col].dtype == 'object':
                # 문자열인 경우 datetime으로 변환
                datetime_series = pd.to_datetime(X_sorted[datetime_col])
            else:
                # 이미 datetime인 경우 그대로 사용
                datetime_series = X_sorted[datetime_col]
            all_dates = datetime_series.dt.date.unique()
        except Exception as e:
            print(f"날짜 변환 에러: {e}")
            # fallback: 인덱스 기반 분할
            return self._split_by_index(X_sorted)
        all_dates = sorted(all_dates)
        total_days = len(all_dates)
        
        for i in range(self.n_splits):
            # 테스트 기간 계산
            test_end_idx = total_days - i * (self.test_size_days + self.gap_days)
            test_start_idx = test_end_idx - self.test_size_days
            train_end_idx = test_start_idx - self.gap_days
            
            if train_end_idx <= 0:
                continue
                
            train_dates = all_dates[:train_end_idx]
            test_dates = all_dates[test_start_idx:test_end_idx]
            
            # 모든 건물에서 해당 날짜 찾기
            train_mask = pd.to_datetime(X_sorted[datetime_col]).dt.date.isin(train_dates)
            test_mask = pd.to_datetime(X_sorted[datetime_col]).dt.date.isin(test_dates)
            
            train_indices = X_sorted[train_mask].index.tolist()
            test_indices = X_sorted[test_mask].index.tolist()
            
            # 빈 분할 제외
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def _split_by_index(self, X_sorted):
        """인덱스 기반 간단한 분할 (날짜 파싱 실패시 fallback)"""
        total_samples = len(X_sorted)
        test_size = total_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = total_samples - (i + 1) * test_size
            test_end = total_samples - i * test_size if i > 0 else total_samples
            train_end = test_start
            
            if train_end <= 0:
                continue
                
            train_indices = X_sorted.iloc[:train_end].index.tolist()
            test_indices = X_sorted.iloc[test_start:test_end].index.tolist()
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

def create_validation_split(df, datetime_col='일시', building_col='건물번호', 
                          test_ratio=0.15, gap_days=1):
    """
    단일 validation split 생성 (최근 데이터를 test로)
    
    Args:
        df: 데이터프레임
        datetime_col: 날짜 컬럼명
        building_col: 건물 컬럼명  
        test_ratio: 테스트 비율
        gap_days: train-test 간격 (일)
        
    Returns:
        train_df, val_df
    """
    # 시간 순 정렬
    df_sorted = df.sort_values([building_col, datetime_col])
    
    # 전체 날짜 범위
    all_dates = pd.to_datetime(df_sorted[datetime_col]).dt.date.unique()
    all_dates = sorted(all_dates)
    
    # 분할점 계산
    total_days = len(all_dates)
    test_days = int(total_days * test_ratio)
    
    train_end_idx = total_days - test_days - gap_days
    test_start_idx = total_days - test_days
    
    train_dates = all_dates[:train_end_idx]
    test_dates = all_dates[test_start_idx:]
    
    # 분할
    train_mask = pd.to_datetime(df_sorted[datetime_col]).dt.date.isin(train_dates)
    test_mask = pd.to_datetime(df_sorted[datetime_col]).dt.date.isin(test_dates)
    
    train_df = df_sorted[train_mask].reset_index(drop=True)
    val_df = df_sorted[test_mask].reset_index(drop=True)
    
    print(f"Train 기간: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}일)")
    print(f"Validation 기간: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}일)")
    print(f"Train 샘플: {len(train_df)}, Validation 샘플: {len(val_df)}")
    
    return train_df, val_df

def get_test_period_validation(df, datetime_col='일시', building_col='건물번호'):
    """
    실제 테스트 기간과 유사한 validation set 생성
    (마지막 7일을 validation으로 사용)
    
    Args:
        df: 데이터프레임
        datetime_col: 날짜 컬럼명
        building_col: 건물 컬럼명
        
    Returns:
        train_df, val_df
    """
    return create_validation_split(
        df, datetime_col, building_col, 
        test_ratio=7/85,  # 약 7일/전체기간
        gap_days=1
    )

def validate_time_split(train_df, val_df, datetime_col='일시'):
    """
    시계열 분할이 올바른지 검증
    
    Args:
        train_df: 훈련 데이터
        val_df: 검증 데이터  
        datetime_col: 날짜 컬럼명
    """
    train_max_date = pd.to_datetime(train_df[datetime_col]).max()
    val_min_date = pd.to_datetime(val_df[datetime_col]).min()
    
    print(f"\n시계열 분할 검증:")
    print(f"Train 최대 날짜: {train_max_date}")
    print(f"Validation 최소 날짜: {val_min_date}")
    print(f"Gap: {(val_min_date - train_max_date).days}일")
    
    # 데이터 리키지 체크
    if train_max_date >= val_min_date:
        print("⚠️  경고: 데이터 리키지 발생! Train과 Val 기간이 겹칩니다.")
        return False
    else:
        print("✅ 시계열 분할이 올바릅니다.")
        return True

if __name__ == "__main__":
    # 테스트 코드
    print("TimeSeriesSplitBuilding 테스트")
    
    # 더미 데이터 생성
    dates = pd.date_range('2024-06-01', '2024-08-24', freq='H')
    buildings = [1, 2, 3]
    
    test_data = []
    for building in buildings:
        for date in dates:
            test_data.append({
                '건물번호': building,
                '일시': date,
                'value': np.random.rand()
            })
    
    df_test = pd.DataFrame(test_data)
    
    # TimeSeriesSplit 테스트
    tscv = TimeSeriesSplitBuilding(n_splits=3, test_size_days=7, gap_days=1)
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(df_test)):
        print(f"\nFold {i+1}:")
        print(f"Train samples: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")
        
        train_dates = pd.to_datetime(df_test.iloc[train_idx]['일시'])
        test_dates = pd.to_datetime(df_test.iloc[test_idx]['일시'])
        
        print(f"Train period: {train_dates.min()} ~ {train_dates.max()}")
        print(f"Test period: {test_dates.min()} ~ {test_dates.max()}")
        
        if i >= 1:  # 처음 2개 fold만 출력
            break