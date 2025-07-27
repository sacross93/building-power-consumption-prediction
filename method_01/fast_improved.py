#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 개선 모델 - method_05 기법 적용 (빠른 테스트용)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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
    
    # 범주형 변수 처리
    categorical_cols = ['건물번호', '건물유형']
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    print(f"최종 피처 수: {len(feature_cols)}")
    print(f"범주형 피처: {[col for col in categorical_cols if col in feature_cols]}")
    
    return X_train, y_train, X_test, feature_cols

def main():
    print("빠른 개선 모델 - method_05 핵심 기법")
    print("=" * 50)
    
    # 1. 데이터 로드
    train_df, test_df = load_and_prepare_data()
    
    # 2. 피처 엔지니어링
    train_df, test_df = quick_feature_engineering(train_df, test_df)
    
    # 3. 모델링용 피처 준비
    X_train, y_train, X_test, feature_cols = prepare_features_for_modeling(train_df, test_df)
    
    # 4. Chronological split (마지막 7일)
    train_df_sorted = train_df.sort_values('일시')
    cutoff_date = train_df_sorted['일시'].max() - pd.Timedelta(days=7)
    
    train_mask = train_df_sorted['일시'] < cutoff_date
    val_mask = ~train_mask
    
    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    
    print(f"Split - Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Cutoff date: {cutoff_date}")
    
    # 5. LightGBM 모델 훈련 (범주형 변수 지원)
    print("LightGBM 훈련...")
    
    categorical_features = [col for col in ['건물번호', '건물유형'] if col in X_train.columns]
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=300,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=1000
    )
    
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    # 6. 검증
    y_pred = lgb_model.predict(X_val)
    y_pred = np.maximum(y_pred, 0)  # 음수 클리핑
    
    val_smape = smape(y_val.values, y_pred)
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    if val_smape <= 10.0:
        print("*** 목표 달성! SMAPE <= 10.0 ***")
    else:
        print(f"목표까지 {val_smape - 10.0:.4f} 더 개선 필요")
    
    # 7. 최종 모델 훈련 및 예측
    print("최종 모델 훈련...")
    
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=300,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_child_samples=20,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=-1,
        random_state=42,
        n_estimators=1500
    )
    
    final_model.fit(X_train, y_train, categorical_feature=categorical_features)
    
    # 8. 테스트 예측
    test_pred = final_model.predict(X_test)
    test_pred = np.maximum(test_pred, 0)
    
    # 9. 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_df['num_date_time'],
        'answer': test_pred
    })
    
    submission.to_csv('submission_fast_improved.csv', index=False)
    
    print(f"\\n{'='*50}")
    print("최종 결과")
    print(f"{'='*50}")
    print(f"Validation SMAPE: {val_smape:.4f}")
    print(f"목표 SMAPE: 10.0000")
    print(f"사용된 피처 수: {len(feature_cols)}")
    print(f"예측값 범위: {test_pred.min():.2f} ~ {test_pred.max():.2f}")
    print("제출 파일: submission_fast_improved.csv")
    
    if val_smape <= 10.0:
        print("*** 목표 달성! ***")
    else:
        print(f"추가 개선 필요: {val_smape - 10.0:.4f}")
    
    # 10. 피처 중요도 출력
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\nTop 10 Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return val_smape

if __name__ == "__main__":
    final_score = main()