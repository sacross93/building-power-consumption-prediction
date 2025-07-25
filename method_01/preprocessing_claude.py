#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건물 전력사용량 예측 - 데이터 전처리
Author: Claude
Date: 2025-07-24
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.tsa.seasonal import STL
import holidays

warnings.filterwarnings('ignore')

class PowerConsumptionPreprocessor:
    """전력소비량 예측을 위한 데이터 전처리 클래스"""
    
    def __init__(self):
        self.scalers = {}
        self.outlier_bounds = {}
        self.pca_weather = None
        self.building_clusters = {}
        
        # 결과 저장 폴더 생성
        os.makedirs('processed_data', exist_ok=True)
        
    def load_data(self):
        """데이터 로드"""
        print("데이터 로드 중...")
        
        train = pd.read_csv('data/train.csv', encoding='utf-8-sig')
        building_info = pd.read_csv('data/building_info.csv', encoding='utf-8-sig') 
        test = pd.read_csv('data/test.csv', encoding='utf-8-sig')
        
        # 컬럼명 정리 (BOM 문자 제거)
        train.columns = train.columns.str.strip()
        building_info.columns = building_info.columns.str.strip()
        test.columns = test.columns.str.strip()
        
        print("컬럼명:", train.columns.tolist())
        
        # 일시 컬럼을 datetime으로 변환
        train['일시'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
        test['일시'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')
        
        print(f"Train: {train.shape}, Building Info: {building_info.shape}, Test: {test.shape}")
        return train, building_info, test
    
    def basic_preprocessing(self, train, building_info, test):
        """기본 전처리 및 결측치 처리"""
        print("\n=== 기본 전처리 ===")
        
        # Building info의 '-' 값 처리
        print("Building info '-' 값 처리...")
        
        # 태양광/ESS 관련 컬럼들을 숫자형으로 변환하고 '-'는 0으로 처리
        numeric_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
        for col in numeric_cols:
            building_info[col] = pd.to_numeric(building_info[col], errors='coerce').fillna(0)
        
        # 재생에너지 설비 보유 여부 이진 변수 생성
        building_info['has_solar'] = (building_info['태양광용량(kW)'] > 0).astype(int)
        building_info['has_ess'] = (building_info['ESS저장용량(kWh)'] > 0).astype(int)
        building_info['has_renewable'] = ((building_info['has_solar'] == 1) | 
                                        (building_info['has_ess'] == 1)).astype(int)
        
        # Train/Test 데이터에 건물 정보 병합
        train = train.merge(building_info, on='건물번호', how='left')
        test = test.merge(building_info, on='건물번호', how='left')
        
        print(f"병합 후 - Train: {train.shape}, Test: {test.shape}")
        
        return train, test, building_info
    
    def create_time_features(self, df):
        """시간 관련 피처 생성"""
        print("시간 관련 피처 생성...")
        
        # 기본 시간 피처
        df['year'] = df['일시'].dt.year
        df['month'] = df['일시'].dt.month
        df['day'] = df['일시'].dt.day
        df['hour'] = df['일시'].dt.hour
        df['weekday'] = df['일시'].dt.dayofweek
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # 한국 공휴일 정보
        kr_holidays = holidays.SouthKorea(years=range(2024, 2025))
        df['is_holiday'] = df['일시'].dt.date.apply(lambda x: x in kr_holidays).astype(int)
        
        # 근무시간 변수 (건물 유형별 고려)
        def get_work_hours(row):
            building_type = row['건물유형']
            hour = row['hour']
            weekday = row['weekday']
            
            if building_type in ['아파트']:
                return 1  # 아파트는 24시간
            elif building_type in ['백화점', '상용']:
                return 1 if (10 <= hour <= 22) else 0
            elif building_type in ['학교']:
                return 1 if (weekday < 5 and 8 <= hour <= 18) else 0
            elif building_type in ['병원']:
                return 1  # 병원은 24시간
            elif building_type in ['호텔']:
                return 1  # 호텔은 24시간  
            elif building_type in ['IDC(전화국)']:
                return 1  # IDC는 24시간
            else:
                return 1 if (9 <= hour <= 18) else 0
        
        df['is_work_hours'] = df.apply(get_work_hours, axis=1)
        
        # 시간 순환 인코딩 (sin/cos 변환)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_weather_composite_features(self, df):
        """기상 변수 합성 피처 생성"""
        print("기상 합성 변수 생성...")
        
        # 불쾌지수 (Discomfort Index)
        df['discomfort_index'] = (0.81 * df['기온(°C)'] + 
                                 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3)
        
        # 체감온도 (Wind Chill / Heat Index)
        def apparent_temperature(temp, humidity, wind_speed):
            # 간단한 체감온도 공식
            return temp + 0.33 * (humidity / 100.0) - 0.70 * wind_speed - 4.00
        
        df['apparent_temp'] = apparent_temperature(df['기온(°C)'], df['습도(%)'], df['풍속(m/s)'])
        
        # 냉방도일 (Cooling Degree Days) - 기준온도 18도
        df['cooling_degree_days'] = np.maximum(0, df['기온(°C)'] - 18)
        
        # 난방도일 (Heating Degree Days) - 기준온도 18도  
        df['heating_degree_days'] = np.maximum(0, 18 - df['기온(°C)'])
        
        # 일사열지수 (Solar Heat Index) - 안전한 계산
        if '일사(MJ/m2)' in df.columns:
            df['solar_heat_index'] = df['일사(MJ/m2)'] * (df['기온(°C)'] / 30.0)
        else:
            df['solar_heat_index'] = 0
        
        # 습도 카테고리
        df['humidity_category'] = pd.cut(df['습도(%)'], 
                                       bins=[0, 30, 60, 80, 100], 
                                       labels=['dry', 'comfortable', 'humid', 'very_humid'])
        
        return df
    
    def create_lag_features(self, df):
        """시차 변수 생성"""
        print("시차 변수 생성...")
        
        if '전력소비량(kWh)' not in df.columns:
            print("Target 변수가 없어 시차 변수 생성 스킵")
            return df
        
        # 건물별로 시차 변수 생성
        lag_features = []
        
        for building_id in df['건물번호'].unique():
            building_data = df[df['건물번호'] == building_id].copy()
            building_data = building_data.sort_values('일시')
            
            # 1시간, 24시간, 168시간(1주) 전 소비량
            building_data['power_lag_1h'] = building_data['전력소비량(kWh)'].shift(1)
            building_data['power_lag_24h'] = building_data['전력소비량(kWh)'].shift(24)
            building_data['power_lag_168h'] = building_data['전력소비량(kWh)'].shift(168)
            
            # Rolling 통계 (과거 24시간)
            building_data['power_rolling_mean_24h'] = (building_data['전력소비량(kWh)']
                                                     .rolling(window=24, min_periods=1).mean().shift(1))
            building_data['power_rolling_std_24h'] = (building_data['전력소비량(kWh)']
                                                    .rolling(window=24, min_periods=1).std().shift(1))
            
            # Rolling 통계 (과거 7일)
            building_data['power_rolling_mean_7d'] = (building_data['전력소비량(kWh)']
                                                    .rolling(window=168, min_periods=1).mean().shift(1))
            
            lag_features.append(building_data)
        
        df_with_lags = pd.concat(lag_features, ignore_index=True)
        df_with_lags = df_with_lags.sort_values(['건물번호', '일시']).reset_index(drop=True)
        
        return df_with_lags
    
    def remove_seasonality(self, df):
        """시계열 계절성 제거 - 간단한 방법 사용"""
        print("시계열 계절성 제거...")
        
        if '전력소비량(kWh)' not in df.columns:
            print("Target 변수가 없어 계절성 제거 스킵")
            return df
        
        # 건물별로 단순한 계절성 제거 (시간대별 평균 제거)
        deseasonalized_data = []
        
        for building_id in df['건물번호'].unique():
            building_data = df[df['건물번호'] == building_id].copy()
            building_data = building_data.sort_values('일시')
            
            try:
                # 시간대별 평균 계산
                hourly_avg = building_data.groupby('hour')['전력소비량(kWh)'].mean()
                
                # 계절성 제거 = 원본 - 시간대별 평균
                building_data['power_deseasonalized'] = (
                    building_data['전력소비량(kWh)'] - 
                    building_data['hour'].map(hourly_avg)
                )
                    
            except Exception as e:
                print(f"건물 {building_id} 계절성 제거 실패: {e}")
                building_data['power_deseasonalized'] = building_data['전력소비량(kWh)']
            
            deseasonalized_data.append(building_data)
        
        df_deseasonalized = pd.concat(deseasonalized_data, ignore_index=True)
        df_deseasonalized = df_deseasonalized.sort_values(['건물번호', '일시']).reset_index(drop=True)
        
        return df_deseasonalized
    
    def handle_outliers(self, df, method='iqr'):
        """이상치 처리"""
        print(f"이상치 처리 방법: {method}")
        
        if '전력소비량(kWh)' not in df.columns:
            print("Target 변수가 없어 이상치 처리 스킵")
            return df
        
        df_processed = df.copy()
        
        if method == 'iqr':
            # 전체 데이터에서 IQR 기반 이상치 제거
            Q1 = df['전력소비량(kWh)'].quantile(0.25)
            Q3 = df['전력소비량(kWh)'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = ((df['전력소비량(kWh)'] >= lower_bound) & 
                           (df['전력소비량(kWh)'] <= upper_bound))
            df_processed = df_processed[outlier_mask].reset_index(drop=True)
            
            print(f"IQR 방법으로 {len(df) - len(df_processed)}개 이상치 제거")
            
        elif method == 'building_percentile':
            # 건물별 상위/하위 5% 제거
            valid_indices = []
            
            for building_id in df['건물번호'].unique():
                building_data = df[df['건물번호'] == building_id]
                lower_bound = building_data['전력소비량(kWh)'].quantile(0.05)
                upper_bound = building_data['전력소비량(kWh)'].quantile(0.95)
                
                building_mask = ((building_data['전력소비량(kWh)'] >= lower_bound) & 
                               (building_data['전력소비량(kWh)'] <= upper_bound))
                valid_indices.extend(building_data[building_mask].index.tolist())
            
            df_processed = df_processed.loc[valid_indices].reset_index(drop=True)
            print(f"건물별 percentile 방법으로 {len(df) - len(df_processed)}개 이상치 제거")
            
        elif method == 'none':
            print("이상치 제거하지 않음")
            
        return df_processed
    
    def create_building_features(self, df):
        """건물 관련 피처 생성"""
        print("건물 관련 피처 생성...")
        
        # 연면적 대비 전력소비량 (있는 경우만)
        if '전력소비량(kWh)' in df.columns:
            df['power_per_area'] = df['전력소비량(kWh)'] / df['연면적(m2)']
            df['power_per_cooling_area'] = df['전력소비량(kWh)'] / df['냉방면적(m2)']
        
        # 건물 규모 카테고리
        area_bins = [0, 50000, 100000, 200000, np.inf]
        area_labels = ['small', 'medium', 'large', 'very_large']
        df['building_size_category'] = pd.cut(df['연면적(m2)'], bins=area_bins, labels=area_labels)
        
        # 냉방면적 비율
        df['cooling_area_ratio'] = df['냉방면적(m2)'] / df['연면적(m2)']
        
        # 건물 유형 인코딩 (빈도 인코딩)
        building_type_counts = df['건물유형'].value_counts()
        df['building_type_frequency'] = df['건물유형'].map(building_type_counts)
        
        return df
    
    def scale_features(self, train_df, test_df=None):
        """피처 스케일링 - 간단한 방법"""
        print("피처 스케일링...")
        
        # 숫자형 컬럼만 스케일링
        exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                       'power_deseasonalized', 'power_transformed', 'year', 'month', 'day', 'hour', 'weekday',
                       'is_weekend', 'is_holiday', 'is_work_hours', 'has_solar', 'has_ess', 'has_renewable']
        
        # 더미 변수도 제외
        exclude_cols.extend([col for col in train_df.columns if col.startswith('건물유형_') or 
                           col.startswith('humidity_category_') or col.startswith('building_size_category_')])
        
        # 스케일링할 컬럼 선택
        numeric_cols = []
        for col in train_df.columns:
            if col not in exclude_cols and train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                numeric_cols.append(col)
        
        print(f"스케일링 대상 컬럼 ({len(numeric_cols)}개): {numeric_cols[:10]}...")  # 처음 10개만 출력
        
        # RobustScaler 사용 (이상치에 강함)
        if numeric_cols:
            self.scalers['numeric'] = RobustScaler()
            train_df[numeric_cols] = self.scalers['numeric'].fit_transform(train_df[numeric_cols].fillna(0))
            
            if test_df is not None:
                # test 데이터에 있는 컬럼만 스케일링
                test_numeric_cols = [col for col in numeric_cols if col in test_df.columns]
                print(f"Test 스케일링 컬럼 ({len(test_numeric_cols)}개): {test_numeric_cols[:5]}...")
                
                if test_numeric_cols:
                    # train에서 fit한 스케일러로 test의 해당 컬럼만 변환
                    test_scaler = RobustScaler()
                    test_scaler.mean_ = self.scalers['numeric'].center_[:len(test_numeric_cols)]
                    test_scaler.scale_ = self.scalers['numeric'].scale_[:len(test_numeric_cols)]
                    
                    # 간단하게 각 컬럼별로 개별 스케일링
                    for i, col in enumerate(test_numeric_cols):
                        if i < len(self.scalers['numeric'].center_):
                            train_col_idx = numeric_cols.index(col)
                            test_df[col] = ((test_df[col].fillna(0) - self.scalers['numeric'].center_[train_col_idx]) / 
                                          self.scalers['numeric'].scale_[train_col_idx])
        
        return train_df, test_df
    
    def transform_target(self, df, method='log'):
        """Target 변수 변환"""
        if '전력소비량(kWh)' not in df.columns:
            return df
            
        print(f"Target 변환 방법: {method}")
        
        if method == 'log':
            df['power_transformed'] = np.log1p(df['전력소비량(kWh)'])
        elif method == 'sqrt':
            df['power_transformed'] = np.sqrt(df['전력소비량(kWh)'])
        elif method == 'boxcox':
            # Box-Cox 변환 (양수 값만)
            positive_mask = df['전력소비량(kWh)'] > 0
            if positive_mask.sum() > 0:
                df.loc[positive_mask, 'power_transformed'], _ = stats.boxcox(
                    df.loc[positive_mask, '전력소비량(kWh)'])
                df.loc[~positive_mask, 'power_transformed'] = 0
            else:
                df['power_transformed'] = df['전력소비량(kWh)']
        else:
            df['power_transformed'] = df['전력소비량(kWh)']
        
        return df
    
    def get_feature_columns(self, df):
        """모델링에 사용할 피처 컬럼 선정"""
        # 제외할 컬럼들
        exclude_cols = ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', 
                       'power_deseasonalized', 'power_transformed']
        
        # 카테고리 변수들은 더미 변수로 변환 (실제 존재하는 컬럼만)
        potential_categorical_cols = ['건물유형', 'humidity_category', 'building_size_category']
        categorical_cols = [col for col in potential_categorical_cols if col in df.columns]
        
        feature_cols = []
        df_processed = df.copy()
        
        for col in df.columns:
            if col not in exclude_cols:
                if col in categorical_cols:
                    # 더미 변수 생성
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    feature_cols.extend(dummies.columns.tolist())
                else:
                    feature_cols.append(col)
        
        # 카테고리 원본 컬럼 제거
        df_processed = df_processed.drop(columns=categorical_cols, errors='ignore')
        
        return df_processed, feature_cols
    
    def preprocess_pipeline(self, outlier_method='iqr', target_transform='log'):
        """전체 전처리 파이프라인"""
        print("=" * 60)
        print(f"데이터 전처리 시작 - 이상치: {outlier_method}, Target: {target_transform}")
        print("=" * 60)
        
        # 1. 데이터 로드
        train, building_info, test = self.load_data()
        
        # 2. 기본 전처리
        train, test, building_info = self.basic_preprocessing(train, building_info, test)
        
        # 3. 시간 피처 생성
        train = self.create_time_features(train)
        test = self.create_time_features(test)
        
        # 4. 기상 합성 변수 생성
        train = self.create_weather_composite_features(train)
        test = self.create_weather_composite_features(test)
        
        # 5. 건물 피처 생성
        train = self.create_building_features(train)
        test = self.create_building_features(test)
        
        # 6. 시차 변수 생성 (train만)
        train = self.create_lag_features(train)
        
        # 7. 계절성 제거 (train만)
        train = self.remove_seasonality(train)
        
        # 8. 이상치 처리 (train만)
        train = self.handle_outliers(train, method=outlier_method)
        
        # 9. Target 변환 (train만)
        train = self.transform_target(train, method=target_transform)
        
        # 10. 피처 컬럼 정리
        train, train_feature_cols = self.get_feature_columns(train)
        test, test_feature_cols = self.get_feature_columns(test)
        
        print(f"Train 컬럼 (피처 정리 후): {train.columns.tolist()}")
        
        # 11. 스케일링
        train, test = self.scale_features(train, test)
        
        print(f"\n전처리 완료!")
        print(f"Train 최종 크기: {train.shape}")
        print(f"Test 최종 크기: {test.shape}")
        print(f"피처 수: {len(train_feature_cols)}")
        
        return train, test, train_feature_cols
    
    def save_processed_data(self, train, test, feature_cols, suffix=""):
        """전처리된 데이터 저장"""
        train_file = f'processed_data/train_processed{suffix}.csv'
        test_file = f'processed_data/test_processed{suffix}.csv'
        feature_file = f'processed_data/feature_columns{suffix}.txt'
        
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
        
        with open(feature_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(feature_cols))
        
        print(f"\n저장 완료:")
        print(f"- {train_file}")
        print(f"- {test_file}")
        print(f"- {feature_file}")

def main():
    """메인 실행 함수"""
    preprocessor = PowerConsumptionPreprocessor()
    
    # 이상치 처리 방법별로 전처리 수행
    outlier_methods = ['none', 'iqr', 'building_percentile']
    target_transforms = ['log']
    
    for outlier_method in outlier_methods:
        for target_transform in target_transforms:
            suffix = f"_{outlier_method}_{target_transform}"
            
            try:
                train, test, feature_cols = preprocessor.preprocess_pipeline(
                    outlier_method=outlier_method, 
                    target_transform=target_transform
                )
                
                preprocessor.save_processed_data(train, test, feature_cols, suffix)
                
            except Exception as e:
                print(f"전처리 실패 ({suffix}): {e}")
                continue
    
    print("\n" + "=" * 60)
    print("모든 전처리 완료!")

if __name__ == "__main__":
    main()