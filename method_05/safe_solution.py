"""
Safe Power Consumption Prediction Solution
=========================================

Data leakage 없는 안전한 솔루션 - lag 피처 제거 버전
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def safe_feature_engineering(train_df, test_df):
    """데이터 리케지 없는 안전한 피처 엔지니어링."""
    print("Creating safe features (no lag features)...")
    
    def create_safe_features(df):
        df = df.copy()
        
        # Parse datetime if needed
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # 시간 피처 (안전)
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday  
        df['month'] = df['datetime'].dt.month
        df['week'] = df['datetime'].dt.isocalendar().week
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['quarter'] = df['datetime'].dt.quarter
        
        # 비즈니스 시간 지표 (안전)
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
        
        # 순환 인코딩 (안전)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 날씨 피처 (안전)
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0)
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0)
            df['temp_comfort'] = np.abs(df['temp'] - 22)
            
            # 온도 구간
            df['temp_very_cold'] = (df['temp'] < 10).astype(int)
            df['temp_cold'] = ((df['temp'] >= 10) & (df['temp'] < 18)).astype(int)
            df['temp_comfortable'] = ((df['temp'] >= 18) & (df['temp'] < 26)).astype(int)
            df['temp_hot'] = ((df['temp'] >= 26) & (df['temp'] < 30)).astype(int)
            df['temp_very_hot'] = (df['temp'] >= 30).astype(int)
            
            # 날씨 상호작용
            if 'humidity' in df.columns:
                df['temp_humidity'] = df['temp'] * df['humidity']
                df['humidity_squared'] = df['humidity'] ** 2
            
            if 'wind_speed' in df.columns:
                df['temp_wind'] = df['temp'] * df['wind_speed']
            
            if 'rainfall' in df.columns:
                df['has_rain'] = (df['rainfall'] > 0).astype(int)
                df['rain_log'] = np.log1p(df['rainfall'])
        
        # 건물 피처 (안전)
        if 'total_area' in df.columns:
            df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
            df['log_total_area'] = np.log1p(df['total_area'])
            df['log_cooling_area'] = np.log1p(df['cooling_area'])
            
            # 건물 크기 카테고리
            df['small_building'] = (df['total_area'] < 10000).astype(int)
            df['medium_building'] = ((df['total_area'] >= 10000) & (df['total_area'] < 50000)).astype(int)
            df['large_building'] = (df['total_area'] >= 50000).astype(int)
        
        # PV 피처 (안전)
        if 'pv_capacity' in df.columns:
            df['has_pv'] = (df['pv_capacity'] > 0).astype(int)
            if 'total_area' in df.columns:
                df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
        
        return df
    
    # 피처 생성 적용
    train_fe = create_safe_features(train_df)
    test_fe = create_safe_features(test_df)
    
    print(f"Safe feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def proper_time_series_validation(train_df):
    """올바른 시계열 교차검증."""
    print("=" * 60)
    print("PROPER TIME SERIES VALIDATION")
    print("=" * 60)
    
    # 피처 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['전력소비량(kWh)']
    
    # 카테고리 인코딩
    encoders = {}
    categorical_cols = ['건물번호', 'building_type']
    
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
    ts_cv = TimeSeriesCV(n_splits=5, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_df, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Validation fold {fold + 1}/5")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # 모델 훈련
        model = xgb.XGBRegressor(
            max_depth=6,
            n_estimators=300,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        fold_smape = smape(y_val.values, y_pred)
        validation_scores.append(fold_smape)
        
        print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    mean_smape = np.mean(validation_scores)
    std_smape = np.std(validation_scores)
    
    print(f"\nValidation Results:")
    print(f"Mean SMAPE: {mean_smape:.4f} (±{std_smape:.4f})")
    
    return mean_smape, std_smape, feature_cols, encoders


def generate_safe_submission():
    """안전한 제출 파일 생성."""
    print("=" * 80)
    print("SAFE POWER CONSUMPTION PREDICTION (NO DATA LEAKAGE)")
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
    
    # 안전한 피처 엔지니어링 (lag 없음)
    train_safe, test_safe = safe_feature_engineering(train_fe, test_fe)
    
    # 올바른 검증
    val_smape, val_std, feature_cols, encoders = proper_time_series_validation(train_safe)
    
    # 최종 모델 훈련
    print("\n" + "=" * 60)
    print("TRAINING FINAL SAFE MODEL")
    print("=" * 60)
    
    X_full = train_safe[feature_cols].copy()
    y_full = train_safe['전력소비량(kWh)']
    
    # 인코딩 적용
    for col, encoder in encoders.items():
        if col in X_full.columns:
            X_full[col] = encoder.fit_transform(X_full[col].astype(str))
    
    for col in X_full.columns:
        if X_full[col].dtype == 'object':
            le = LabelEncoder()
            X_full[col] = le.fit_transform(X_full[col].astype(str))
            encoders[col] = le
    
    # 최종 모델
    final_model = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=500,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    
    final_model.fit(X_full, y_full)
    
    # 테스트 예측
    X_test = test_safe[feature_cols].copy()
    
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
    
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_safe['num_date_time'],
        'prediction': predictions
    })
    
    submission_file = 'submission_safe_no_leakage.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Safe solution completed!")
    print(f"Validation SMAPE: {val_smape:.4f} (±{val_std:.4f})")
    print(f"This should match actual test performance!")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: {submission_file}")
    
    return submission, val_smape


if __name__ == "__main__":
    submission, smape_score = generate_safe_submission()
    
    print(f"\n" + "=" * 80)
    print("SAFE SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Validation SMAPE: {smape_score:.4f}")
    print("✅ NO DATA LEAKAGE - Validation should match test performance")
    print(f"Submission file: submission_safe_no_leakage.csv")