"""
Optimized solution based on solution_backup.py
7-8 SMAPE → 5-6 SMAPE 목표
GPU 서버 전용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def load_data(train_path: Path, test_path: Path, building_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge data (원본과 동일)."""
    # Column renaming map to avoid special characters
    rename_map = {
        '기온(°C)': 'temp',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'wind_speed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine_hours',
        '일사(MJ/m2)': 'solar_radiation',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'pv_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity',
        '건물유형': 'building_type',
    }
    
    # Load CSVs
    train = pd.read_csv(train_path, encoding='utf-8-sig')
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    building_info = pd.read_csv(building_path, encoding='utf-8-sig')
    
    # Rename columns
    train.rename(columns=rename_map, inplace=True)
    test.rename(columns=rename_map, inplace=True)
    building_info.rename(columns=rename_map, inplace=True)
    
    # Merge building info
    train = train.merge(building_info, on='건물번호', how='left')
    test = test.merge(building_info, on='건물번호', how='left')
    
    return train, test

def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced feature engineering for better performance."""
    # 기존 피처 엔지니어링 (원본과 동일)
    train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
    test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')
    
    for df in (train, test):
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Missing value imputation (handle '-' strings)
    for col in ['total_area', 'cooling_area', 'pv_capacity', 'ess_capacity', 'pcs_capacity']:
        # Replace '-' with NaN and convert to numeric
        train[col] = pd.to_numeric(train[col].replace('-', np.nan), errors='coerce')
        test[col] = pd.to_numeric(test[col].replace('-', np.nan), errors='coerce')
        
        # Fill with median
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
    
    # Sunshine and solar radiation approximation
    train_august = train[train['month'] == 8]
    avg_sunshine = train_august.groupby('hour')['sunshine_hours'].mean()
    avg_solar = train_august.groupby('hour')['solar_radiation'].mean()
    
    train['sunshine_est'] = train['sunshine_hours']
    train['solar_est'] = train['solar_radiation']
    test['sunshine_est'] = test['hour'].map(avg_sunshine)
    test['solar_est'] = test['hour'].map(avg_solar)
    
    train.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True)
    test.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True, errors='ignore')
    
    # Building-level statistics
    building_mean = train.groupby('건물번호')['전력소비량(kWh)'].mean()
    
    bld_hour_mean = (
        train.groupby(['건물번호', 'hour'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_hour_mean'})
    )
    train = train.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test = test.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test['bld_hour_mean'] = test['bld_hour_mean'].fillna(test['건물번호'].map(building_mean))
    
    bld_wd_mean = (
        train.groupby(['건물번호', 'weekday'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_wd_mean'})
    )
    train = train.merge(bld_wd_mean, on=['건물번호', 'weekday'], how='left')
    test = test.merge(bld_wd_mean, on=['건물번호', 'weekday'], how='left')
    test['bld_wd_mean'] = test['bld_wd_mean'].fillna(test['건물번호'].map(building_mean))
    
    bld_month_mean = (
        train.groupby(['건물번호', 'month'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_month_mean'})
    )
    train = train.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test = test.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test['bld_month_mean'] = test['bld_month_mean'].fillna(test['건물번호'].map(building_mean))
    
    # Enhanced feature engineering for 5-6 SMAPE target
    for df in (train, test):
        # 기존 피처들
        df['area_ratio'] = df['cooling_area'] / df['total_area']
        df['pv_per_area'] = df['pv_capacity'] / df['total_area']
        df['humidity_temp'] = df['humidity'] * df['temp']
        df['rain_wind'] = df['rainfall'] * df['wind_speed']
        
        # 새로운 성능 향상 피처들
        # 1. 순환 시간 피처 (더 부드러운 시간 표현)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # 2. 온도 구간별 피처
        df['temp_category'] = pd.cut(df['temp'], 
                                   bins=[-np.inf, 10, 20, 25, 30, np.inf], 
                                   labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
        
        # 3. 습도 구간별 피처
        df['humidity_category'] = pd.cut(df['humidity'], 
                                       bins=[0, 40, 60, 80, 100], 
                                       labels=['dry', 'normal', 'humid', 'very_humid'])
        
        # 4. 복합 피처들
        df['comfort_index'] = df['temp'] * (1 - df['humidity'] / 100)  # 체감온도
        df['energy_efficiency'] = (df['pv_capacity'] + 1) / (df['total_area'] + 1) * 1000
        df['cooling_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
        
        # 5. 건물별-시간별 복합 피처
        df['bld_hour_interaction'] = df['bld_hour_mean'] * df['hour'] / 24
        df['bld_temp_interaction'] = df['bld_hour_mean'] * df['temp'] / 30
        
        # 6. 계절성 피처
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['peak_hours'] = df['hour'].isin([10, 11, 14, 15, 16]).astype(int)
    
    return train, test

def build_ensemble_model(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    """Build ensemble model for 5-6 SMAPE target."""
    print("Building optimized ensemble model for 5-6 SMAPE target...")
    
    # Feature preparation
    feature_cols = [col for col in train.columns 
                   if col not in ['num_date_time', '건물번호', '일시', 'datetime', '전력소비량(kWh)']]
    
    X = train[feature_cols]
    y = train['전력소비량(kWh)']
    X_test = test[feature_cols]
    
    # Preprocessing
    categorical_features = ['building_type', 'temp_category', 'humidity_category']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 1. Optimized XGBoost (GPU)
    xgb_model = xgb.XGBRegressor(
        max_depth=10,
        n_estimators=1500,  # 더 많은 트리
        learning_rate=0.02,  # 더 낮은 학습률
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        min_child_weight=3,
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        random_state=42,
    )
    
    # 2. Optimized LightGBM (GPU) 
    lgb_model = lgb.LGBMRegressor(
        max_depth=12,
        n_estimators=1800,
        learning_rate=0.015,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=2.5,
        num_leaves=300,
        min_child_samples=20,
        device='gpu',
        gpu_use_dp=True,
        random_state=42,
        verbosity=-1
    )
    
    # 3. Ensemble
    ensemble = VotingRegressor([
        ('xgb', Pipeline([('preprocess', preprocessor), ('model', xgb_model)])),
        ('lgb', Pipeline([('preprocess', preprocessor), ('model', lgb_model)]))
    ])
    
    # Validation split
    cutoff = pd.Timestamp('2024-08-18')
    train_mask = train['datetime'] < cutoff
    val_mask = ~train_mask
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    
    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    
    # Train ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Validation
    val_pred = ensemble.predict(X_val)
    val_smape = smape(y_val.values, val_pred)
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    # Final training on full data
    print("Final training on full dataset...")
    ensemble.fit(X, y)
    
    # Predictions
    test_pred = ensemble.predict(X_test)
    
    # Save results
    validation_path = output_dir / 'optimized_validation.txt'
    with validation_path.open('w') as f:
        f.write(f'Optimized Validation SMAPE: {val_smape:.6f}%\n')
        f.write(f'Target: 5-6 SMAPE\n')
        f.write(f'Features used: {len(feature_cols)}\n')
    
    submission = test[['num_date_time']].copy()
    submission['answer'] = test_pred
    submission.to_csv(output_dir / 'submission_optimized.csv', index=False)
    
    print(f"✅ Optimized model completed!")
    print(f"📊 Validation SMAPE: {val_smape:.4f} (Target: 5-6)")
    print(f"💾 Submission saved: submission_optimized.csv")
    
    return val_smape

def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("OPTIMIZED SOLUTION: 7-8 SMAPE → 5-6 SMAPE")
    print("Based on solution_backup.py + GPU optimization")
    print("=" * 60)
    
    base_dir = Path('../data')
    train_path = base_dir / 'train.csv'
    test_path = base_dir / 'test.csv'
    building_path = base_dir / 'building_info.csv'
    
    # Load and engineer features
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # Build and train optimized model
    result = build_ensemble_model(train_fe, test_fe, Path('.'))
    
    print(f"\n🎯 Mission: Convert 7-8 SMAPE → 5-6 SMAPE")
    print(f"🚀 Result: {result:.4f} SMAPE")
    
    if result < 6.0:
        print("🎉 SUCCESS! Target achieved!")
    elif result < 7.0:
        print("✅ Good progress! Close to target.")
    else:
        print("📈 Need more optimization.")

if __name__ == "__main__":
    main()