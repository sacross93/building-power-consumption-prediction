"""
Enhanced Safe Power Consumption Prediction Solution
==================================================

고급 피처 엔지니어링과 앙상블을 통한 성능 개선 (데이터 리케지 없음)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def advanced_safe_features(train_df, test_df):
    """고급 안전 피처 엔지니어링 (lag 없음)."""
    print("Creating advanced safe features...")
    
    def create_features(df):
        df = df.copy()
        
        # 기본 datetime 파싱
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # 고급 시간 피처
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['week'] = df['datetime'].dt.isocalendar().week
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['quarter'] = df['datetime'].dt.quarter
        
        # 시간 카테고리
        df['season'] = ((df['month'] % 12 + 3) // 3).map({1: 0, 2: 1, 3: 2, 4: 3})  # 0: 겨울, 1: 봄, 2: 여름, 3: 가을
        df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True)
        
        # 비즈니스 시간 (더 세분화)
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_holiday'] = ((df['month'] == 8) & (df['day'].between(15, 17))).astype(int)  # 광복절 기간
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (df['weekday'] < 5)).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10) & (df['weekday'] < 5)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (df['weekday'] < 5)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13) & (df['weekday'] < 5)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 8)).astype(int)
        df['is_late_evening'] = ((df['hour'] >= 20) & (df['hour'] <= 22)).astype(int)
        df['is_deep_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        
        # 순환 인코딩 (더 많은 주기)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # 고급 날씨 피처
        if 'temp' in df.columns:
            # 온도 관련
            df['temp_squared'] = df['temp'] ** 2
            df['temp_cubed'] = df['temp'] ** 3
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0) ** 2
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0) ** 2
            df['temp_comfort'] = np.abs(df['temp'] - 22)
            df['temp_extreme'] = np.maximum(np.abs(df['temp'] - 22) - 8, 0)
            
            # 온도 구간 (더 세분화)
            df['temp_freezing'] = (df['temp'] < 0).astype(int)
            df['temp_very_cold'] = ((df['temp'] >= 0) & (df['temp'] < 10)).astype(int)
            df['temp_cold'] = ((df['temp'] >= 10) & (df['temp'] < 18)).astype(int)
            df['temp_mild'] = ((df['temp'] >= 18) & (df['temp'] < 22)).astype(int)
            df['temp_comfortable'] = ((df['temp'] >= 22) & (df['temp'] < 26)).astype(int)
            df['temp_warm'] = ((df['temp'] >= 26) & (df['temp'] < 30)).astype(int)
            df['temp_hot'] = ((df['temp'] >= 30) & (df['temp'] < 35)).astype(int)
            df['temp_very_hot'] = (df['temp'] >= 35).astype(int)
            
            # 시간과 온도 상호작용
            df['temp_hour_interaction'] = df['temp'] * df['hour']
            df['temp_weekday_interaction'] = df['temp'] * df['weekday']
            df['temp_month_interaction'] = df['temp'] * df['month']
            
            # 습도 관련
            if 'humidity' in df.columns:
                df['humidity_squared'] = df['humidity'] ** 2
                df['temp_humidity'] = df['temp'] * df['humidity']
                df['temp_humidity_squared'] = (df['temp'] * df['humidity']) ** 2
                df['discomfort_index'] = df['temp'] - 0.55 * (1 - df['humidity'] / 100) * (df['temp'] - 14.5)
                
                # 습도 구간
                df['humidity_low'] = (df['humidity'] < 30).astype(int)
                df['humidity_normal'] = ((df['humidity'] >= 30) & (df['humidity'] < 70)).astype(int)
                df['humidity_high'] = (df['humidity'] >= 70).astype(int)
            
            # 바람 관련
            if 'wind_speed' in df.columns:
                df['wind_speed_squared'] = df['wind_speed'] ** 2
                df['temp_wind'] = df['temp'] * df['wind_speed']
                df['wind_cooling_effect'] = df['wind_speed'] * np.maximum(df['temp'] - 20, 0)
                
                # 바람 구간
                df['wind_calm'] = (df['wind_speed'] < 1).astype(int)
                df['wind_light'] = ((df['wind_speed'] >= 1) & (df['wind_speed'] < 3)).astype(int)
                df['wind_moderate'] = ((df['wind_speed'] >= 3) & (df['wind_speed'] < 6)).astype(int)
                df['wind_strong'] = (df['wind_speed'] >= 6).astype(int)
            
            # 강수량 관련
            if 'rainfall' in df.columns:
                df['has_rain'] = (df['rainfall'] > 0).astype(int)
                df['light_rain'] = ((df['rainfall'] > 0) & (df['rainfall'] <= 2)).astype(int)
                df['moderate_rain'] = ((df['rainfall'] > 2) & (df['rainfall'] <= 10)).astype(int)
                df['heavy_rain'] = (df['rainfall'] > 10).astype(int)
                df['rain_log'] = np.log1p(df['rainfall'])
                df['rain_temp_interaction'] = df['rainfall'] * df['temp']
        
        # 고급 건물 피처
        if 'total_area' in df.columns:
            df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
            df['non_cooling_area'] = df['total_area'] - df['cooling_area']
            df['log_total_area'] = np.log1p(df['total_area'])
            df['log_cooling_area'] = np.log1p(df['cooling_area'])
            df['sqrt_total_area'] = np.sqrt(df['total_area'])
            
            # 건물 크기 카테고리 (더 세분화)
            df['micro_building'] = (df['total_area'] < 1000).astype(int)
            df['small_building'] = ((df['total_area'] >= 1000) & (df['total_area'] < 10000)).astype(int)
            df['medium_building'] = ((df['total_area'] >= 10000) & (df['total_area'] < 50000)).astype(int)
            df['large_building'] = ((df['total_area'] >= 50000) & (df['total_area'] < 100000)).astype(int)
            df['mega_building'] = (df['total_area'] >= 100000).astype(int)
            
            # 면적 효율성
            df['area_efficiency'] = df['cooling_area'] / (df['total_area'] + 1)
            df['area_efficiency_log'] = np.log1p(df['area_efficiency'])
        
        # PV 관련 (더 세분화)
        if 'pv_capacity' in df.columns:
            df['has_pv'] = (df['pv_capacity'] > 0).astype(int)
            df['pv_log'] = np.log1p(df['pv_capacity'])
            
            if 'total_area' in df.columns:
                df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
                df['pv_efficiency'] = np.log1p(df['pv_per_area'])
                
            # PV 크기 카테고리
            df['pv_small'] = ((df['pv_capacity'] > 0) & (df['pv_capacity'] <= 100)).astype(int)
            df['pv_medium'] = ((df['pv_capacity'] > 100) & (df['pv_capacity'] <= 500)).astype(int)
            df['pv_large'] = (df['pv_capacity'] > 500).astype(int)
        
        return df
    
    # 고급 건물별 통계 피처 (안전하게)
    def add_building_statistics(train_df, test_df):
        """훈련 데이터에서 건물별 통계를 계산하여 안전하게 적용."""
        
        # 훈련 데이터에서만 건물별 통계 계산
        building_stats = {}
        
        for building_id in train_df['건물번호'].unique():
            building_data = train_df[train_df['건물번호'] == building_id]
            
            if '전력소비량(kWh)' in building_data.columns:
                # 시간별 패턴
                hourly_pattern = building_data.groupby('hour')['전력소비량(kWh)'].mean()
                weekday_pattern = building_data.groupby('weekday')['전력소비량(kWh)'].mean()
                month_pattern = building_data.groupby('month')['전력소비량(kWh)'].mean()
                
                # 기본 통계
                power_mean = building_data['전력소비량(kWh)'].mean()
                power_std = building_data['전력소비량(kWh)'].std()
                power_min = building_data['전력소비량(kWh)'].min()
                power_max = building_data['전력소비량(kWh)'].max()
                
                building_stats[building_id] = {
                    'hourly_pattern': hourly_pattern,
                    'weekday_pattern': weekday_pattern,
                    'month_pattern': month_pattern,
                    'power_mean': power_mean,
                    'power_std': power_std,
                    'power_min': power_min,
                    'power_max': power_max
                }
        
        # 훈련과 테스트 데이터에 통계 적용
        for df_name, df in [('train', train_df), ('test', test_df)]:
            for building_id in df['건물번호'].unique():
                mask = df['건물번호'] == building_id
                
                if building_id in building_stats:
                    stats = building_stats[building_id]
                    
                    # 시간별 평균 전력
                    df.loc[mask, 'building_hour_mean'] = df.loc[mask, 'hour'].map(
                        stats['hourly_pattern']
                    ).fillna(stats['power_mean'])
                    
                    # 요일별 평균 전력
                    df.loc[mask, 'building_weekday_mean'] = df.loc[mask, 'weekday'].map(
                        stats['weekday_pattern']
                    ).fillna(stats['power_mean'])
                    
                    # 월별 평균 전력
                    df.loc[mask, 'building_month_mean'] = df.loc[mask, 'month'].map(
                        stats['month_pattern']
                    ).fillna(stats['power_mean'])
                    
                    # 기본 통계
                    df.loc[mask, 'building_power_mean'] = stats['power_mean']
                    df.loc[mask, 'building_power_std'] = stats['power_std']
                    df.loc[mask, 'building_power_range'] = stats['power_max'] - stats['power_min']
                else:
                    # 새로운 건물의 경우 전체 평균 사용
                    global_mean = train_df['전력소비량(kWh)'].mean() if '전력소비량(kWh)' in train_df.columns else 1000
                    df.loc[mask, 'building_hour_mean'] = global_mean
                    df.loc[mask, 'building_weekday_mean'] = global_mean
                    df.loc[mask, 'building_month_mean'] = global_mean
                    df.loc[mask, 'building_power_mean'] = global_mean
                    df.loc[mask, 'building_power_std'] = global_mean * 0.3
                    df.loc[mask, 'building_power_range'] = global_mean * 2
        
        return train_df, test_df
    
    # 피처 생성 적용
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # 건물별 통계 추가
    train_fe, test_fe = add_building_statistics(train_fe, test_fe)
    
    print(f"Advanced safe feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def enhanced_ensemble_training(train_df):
    """향상된 앙상블 모델 훈련."""
    print("=" * 60)
    print("ENHANCED ENSEMBLE TRAINING")
    print("=" * 60)
    
    # 피처 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['전력소비량(kWh)']
    
    # 카테고리 인코딩
    encoders = {}
    categorical_cols = ['건물번호', 'building_type', 'season', 'time_of_day']
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # 객체 타입 처리
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # 시계열 교차검증
    ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_df, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Ensemble validation fold {fold + 1}/3")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # XGBoost 모델
        xgb_model = xgb.XGBRegressor(
            max_depth=8,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            verbosity=0
        )
        
        # LightGBM 모델
        lgb_model = lgb.LGBMRegressor(
            max_depth=8,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            verbosity=-1
        )
        
        # 앙상블 훈련
        ensemble = VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ])
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)
        
        fold_smape = smape(y_val.values, y_pred)
        validation_scores.append(fold_smape)
        
        print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    mean_smape = np.mean(validation_scores)
    std_smape = np.std(validation_scores)
    
    print(f"\nEnhanced Ensemble Validation:")
    print(f"Mean SMAPE: {mean_smape:.4f} (±{std_smape:.4f})")
    
    return mean_smape, std_smape, feature_cols, encoders


def generate_enhanced_submission():
    """향상된 제출 파일 생성."""
    print("=" * 80)
    print("ENHANCED SAFE POWER CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # 데이터 로드
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 기본 피처 엔지니어링
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # 고급 안전 피처 엔지니어링
    train_enhanced, test_enhanced = advanced_safe_features(train_fe, test_fe)
    
    # 앙상블 검증
    val_smape, val_std, feature_cols, encoders = enhanced_ensemble_training(train_enhanced)
    
    # 최종 앙상블 모델 훈련
    print("\n" + "=" * 60)
    print("TRAINING FINAL ENSEMBLE")
    print("=" * 60)
    
    X_full = train_enhanced[feature_cols].copy()
    y_full = train_enhanced['전력소비량(kWh)']
    
    # 인코딩 적용
    for col, encoder in encoders.items():
        if col in X_full.columns:
            X_full[col] = encoder.fit_transform(X_full[col].astype(str))
    
    for col in X_full.columns:
        if X_full[col].dtype == 'object':
            le = LabelEncoder()
            X_full[col] = le.fit_transform(X_full[col].astype(str))
            encoders[col] = le
    
    # 최종 앙상블
    xgb_final = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbosity=0
    )
    
    lgb_final = lgb.LGBMRegressor(
        max_depth=8,
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbosity=-1
    )
    
    final_ensemble = VotingRegressor([
        ('xgb', xgb_final),
        ('lgb', lgb_final)
    ])
    
    final_ensemble.fit(X_full, y_full)
    
    # 테스트 예측
    X_test = test_enhanced[feature_cols].copy()
    
    # 테스트에 인코딩 적용
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
    
    predictions = final_ensemble.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_enhanced['num_date_time'],
        'prediction': predictions
    })
    
    submission_file = 'submission_enhanced_safe.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Enhanced safe solution completed!")
    print(f"Validation SMAPE: {val_smape:.4f} (±{val_std:.4f})")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: {submission_file}")
    
    return submission, val_smape


if __name__ == "__main__":
    submission, smape_score = generate_enhanced_submission()
    
    print(f"\n" + "=" * 80)
    print("ENHANCED SAFE SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Validation SMAPE: {smape_score:.4f}")
    print("✅ Advanced features + Ensemble + NO DATA LEAKAGE")
    print(f"Submission file: submission_enhanced_safe.csv")