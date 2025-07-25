

import pandas as pd
import numpy as np
import os

def preprocess_data(data_dir, output_dir):
    """
    Loads, preprocesses, and feature-engineers the power consumption data.

    Args:
        data_dir (str): The directory containing the raw data files (train.csv, building_info.csv).
        output_dir (str): The directory where the preprocessed file will be saved.
    """
    # 1. 데이터 로딩
    print("1. Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    building_info_df = pd.read_csv(os.path.join(data_dir, 'building_info.csv'))

    # Rename columns to English
    train_df.columns = ['num_date_time', 'building_number', 'datetime', 'temperature', 'precipitation', 
                        'windspeed', 'humidity', 'sunshine', 'solar_radiation', 'power_consumption']
    building_info_df.columns = ['building_number', 'building_type', 'total_area', 'cooling_area', 
                                'solar_capacity', 'ess_capacity', 'pcs_capacity']

    # 2. 기본 데이터 정리 및 타입 변환
    print("2. Cleaning and converting data types...")
    # datetime 컬럼 타입 변환
    # The 'datetime' column in train_df is already in string format 'YYYYMMDD HH', 
    # so we can convert it directly to datetime objects.
    train_df['datetime'] = pd.to_datetime(train_df['datetime'], format='%Y%m%d %H')

    # 용량 관련 컬럼의 비숫자 문자 처리 및 숫자 타입 변환
    for col in ['solar_capacity', 'ess_capacity', 'pcs_capacity']:
        building_info_df[col] = building_info_df[col].replace('-', 0).astype(float)

    # 3. 데이터 병합
    print("3. Merging datasets...")
    df = pd.merge(train_df, building_info_df, on='building_number', how='left')

    # 4. 피처 엔지니어링
    print("4. Performing feature engineering...")
    # 시간 관련 특성
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek  # 0: Monday, 6: Sunday
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # 공휴일 특성 (2024년 6월~8월 기준: 6/6 현충일, 8/15 광복절)
    holidays = [pd.Timestamp('2024-06-06'), pd.Timestamp('2024-08-15')]
    df['is_holiday'] = df['datetime'].dt.date.isin([d.date() for d in holidays]).astype(int)
    # 공휴일이 주말과 겹치는 경우도 있으므로, 주말이 아니면서 공휴일인 경우를 특별히 고려할 수 있음
    # 여기서는 단순히 공휴일 여부만 표시
    
    # 불쾌지수(Discomfort Index) 생성
    # DI = T - 0.55 * (1 - 0.01 * RH) * (T - 14.5)
    df['discomfort_index'] = df['temperature'] - 0.55 * (1 - 0.01 * df['humidity']) * (df['temperature'] - 14.5)

    # 건물 유형 원-핫 인코딩
    df = pd.get_dummies(df, columns=['building_type'], prefix='building')

    # 5. 이상치 처리
    print("5. Handling outliers...")
    # outlier_analysis.txt 기반으로 상위 이상치 기준값 설정
    upper_bound = 7552.73
    df['power_consumption'] = df['power_consumption'].clip(upper=upper_bound)

    # 6. 최종 데이터셋 준비 및 저장
    print("6. Saving preprocessed data...")
    # 불필요한 컬럼 제거
    df = df.drop(columns=['num_date_time', 'datetime', 'solar_capacity', 'ess_capacity', 'pcs_capacity'])
    
    # 출력 디렉토리 확인 및 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'preprocessed_gemini.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing complete. Saved to {output_path}")
    print("\nFinal DataFrame columns:")
    print(df.columns)
    print("\nFinal DataFrame head:")
    print(df.head())

if __name__ == '__main__':
    # 실행 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, '..', 'data')
    output_directory = os.path.join(current_dir, '..', 'preprocessed_data_gemini')
    
    preprocess_data(data_directory, output_directory)

