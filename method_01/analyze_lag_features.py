#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시차 변수 문제 분석
"""

import pandas as pd
import numpy as np

def analyze_lag_features():
    print("Analyzing Lag Features")
    print("=" * 40)
    
    # 데이터 로드
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    test_df = pd.read_csv('../processed_data/test_processed_iqr_log.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 시차 변수들 찾기
    lag_cols = [col for col in train_df.columns if 'lag' in col or 'rolling' in col]
    
    print(f"\nLag features in train: {len(lag_cols)}")
    for col in lag_cols:
        print(f"  {col}")
    
    # Test에도 있는지 확인
    test_lag_cols = [col for col in lag_cols if col in test_df.columns]
    
    print(f"\nLag features in test: {len(test_lag_cols)}")
    for col in test_lag_cols:
        print(f"  {col}")
    
    # 시차 변수 통계
    if lag_cols:
        print(f"\nLag features statistics (train):")
        for col in lag_cols:
            if col in train_df.columns:
                values = train_df[col]
                non_zero = (values != 0).sum()
                print(f"{col}:")
                print(f"  Non-zero values: {non_zero}/{len(values)} ({non_zero/len(values)*100:.1f}%)")
                print(f"  Mean: {values.mean():.4f}")
                print(f"  Range: {values.min():.2f} ~ {values.max():.2f}")
    
    # Test에서 시차 변수 상태
    if test_lag_cols:
        print(f"\nLag features statistics (test):")
        for col in test_lag_cols:
            if col in test_df.columns:
                values = test_df[col]
                non_zero = (values != 0).sum()
                print(f"{col}:")
                print(f"  Non-zero values: {non_zero}/{len(values)} ({non_zero/len(values)*100:.1f}%)")
                print(f"  Mean: {values.mean():.4f}")
    
    # 원본 데이터 확인
    print(f"\nOriginal data check:")
    
    if 'power_transformed' in train_df.columns:
        power_transformed = train_df['power_transformed']
        print(f"power_transformed range: {power_transformed.min():.4f} ~ {power_transformed.max():.4f}")
        print(f"power_transformed mean: {power_transformed.mean():.4f}")
    
    if '전력소비량(kWh)' in train_df.columns:
        power_original = train_df['전력소비량(kWh)']
        print(f"전력소비량(kWh) range: {power_original.min():.2f} ~ {power_original.max():.2f}")
        print(f"전력소비량(kWh) mean: {power_original.mean():.2f}")
    
    # 시간 정보 확인
    if '일시' in train_df.columns:
        print(f"\nTime info:")
        print(f"Train time range: {train_df['일시'].min()} ~ {train_df['일시'].max()}")
    
    if '일시' in test_df.columns:
        print(f"Test time range: {test_df['일시'].min()} ~ {test_df['일시'].max()}")
    
    # 건물별 데이터 확인
    if '건물번호' in train_df.columns:
        print(f"\nBuilding info:")
        print(f"Train buildings: {train_df['건물번호'].nunique()}")
        print(f"Train samples per building: {len(train_df) / train_df['건물번호'].nunique():.1f}")
    
    if '건물번호' in test_df.columns:
        print(f"Test buildings: {test_df['건물번호'].nunique()}")
        print(f"Test samples per building: {len(test_df) / test_df['건물번호'].nunique():.1f}")

def test_different_approaches():
    print(f"\n{'='*40}")
    print("Testing Different Approaches")
    print("=" * 40)
    
    train_df = pd.read_csv('../processed_data/train_processed_iqr_log.csv')
    
    # 1. 시차 변수 없이 테스트
    basic_features = ['hour', 'weekday', 'month', 'is_weekend', 'is_holiday',
                     '기온(°C)', '습도(%)', '풍속(m/s)', '연면적(m2)', '냉방면적(m2)']
    
    available_basic = [col for col in basic_features if col in train_df.columns]
    print(f"\nBasic features available: {len(available_basic)}")
    print(f"Basic features: {available_basic}")
    
    # 2. 모든 피처 리스트
    exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                   'power_deseasonalized', 'power_transformed']
    all_features = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"\nAll features available: {len(all_features)}")
    
    # 시차 변수만 확인
    lag_only = [col for col in all_features if 'lag' in col or 'rolling' in col]
    print(f"Lag features only: {len(lag_only)}")
    print(f"Lag features: {lag_only}")
    
    # 중요도 높은 피처들
    important_features = ['hour', 'weekday', 'month', '기온(°C)', '습도(%)', 
                         'discomfort_index', 'apparent_temp', 'cooling_degree_days',
                         'heating_degree_days', 'cooling_area_ratio']
    
    available_important = [col for col in important_features if col in train_df.columns]
    print(f"\nImportant features available: {len(available_important)}")
    
    # 건물 유형 더미 변수들
    building_dummies = [col for col in train_df.columns if col.startswith('건물유형_')]
    print(f"Building type dummies: {len(building_dummies)}")

if __name__ == "__main__":
    analyze_lag_features()
    test_different_approaches()