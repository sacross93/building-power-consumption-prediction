"""
Advanced solution with feature importance analysis and deeper models
8.0877 SMAPE → 5-6 SMAPE 목표
체계적인 분석과 개선
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def load_data(train_path: Path, test_path: Path, building_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge data."""
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
    
    train = pd.read_csv(train_path, encoding='utf-8-sig')
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    building_info = pd.read_csv(building_path, encoding='utf-8-sig')
    
    train.rename(columns=rename_map, inplace=True)
    test.rename(columns=rename_map, inplace=True)
    building_info.rename(columns=rename_map, inplace=True)
    
    train = train.merge(building_info, on='건물번호', how='left')
    test = test.merge(building_info, on='건물번호', how='left')
    
    return train, test

def advanced_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """고급 피처 엔지니어링 - 더 많은 피처 생성."""
    print("Applying advanced feature engineering...")
    
    # 기본 시간 피처
    train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
    test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')
    
    for df in (train, test):
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # 결측값 처리
    for col in ['total_area', 'cooling_area', 'pv_capacity', 'ess_capacity', 'pcs_capacity']:
        train[col] = pd.to_numeric(train[col].replace('-', np.nan), errors='coerce')
        test[col] = pd.to_numeric(test[col].replace('-', np.nan), errors='coerce')
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
    
    # 일조/일사 추정
    train_august = train[train['month'] == 8]
    avg_sunshine = train_august.groupby('hour')['sunshine_hours'].mean()
    avg_solar = train_august.groupby('hour')['solar_radiation'].mean()
    
    train['sunshine_est'] = train['sunshine_hours']
    train['solar_est'] = train['solar_radiation']
    test['sunshine_est'] = test['hour'].map(avg_sunshine)
    test['solar_est'] = test['hour'].map(avg_solar)
    
    train.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True)
    test.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True, errors='ignore')
    
    # 건물별 통계 (기존)
    building_mean = train.groupby('건물번호')['전력소비량(kWh)'].mean()
    
    # 건물별-시간 통계
    bld_hour_mean = (
        train.groupby(['건물번호', 'hour'])['전력소비량(kWh)'].mean()
        .reset_index().rename(columns={'전력소비량(kWh)': 'bld_hour_mean'})
    )
    train = train.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test = test.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test['bld_hour_mean'] = test['bld_hour_mean'].fillna(test['건물번호'].map(building_mean))
    
    # 건물별-주말 통계
    bld_wd_mean = (
        train.groupby(['건물번호', 'weekday'])['전력소비량(kWh)'].mean()
        .reset_index().rename(columns={'전력소비량(kWh)': 'bld_wd_mean'})
    )
    train = train.merge(bld_wd_mean, on=['건물번호', 'weekday'], how='left')
    test = test.merge(bld_wd_mean, on=['건물번호', 'weekday'], how='left')
    test['bld_wd_mean'] = test['bld_wd_mean'].fillna(test['건물번호'].map(building_mean))
    
    # 건물별-월 통계  
    bld_month_mean = (
        train.groupby(['건물번호', 'month'])['전력소비량(kWh)'].mean()
        .reset_index().rename(columns={'전력소비량(kWh)': 'bld_month_mean'})
    )
    train = train.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test = test.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test['bld_month_mean'] = test['bld_month_mean'].fillna(test['건물번호'].map(building_mean))
    
    # 고급 피처 엔지니어링
    for df in (train, test):
        # 1. 기존 피처들
        df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
        df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
        df['humidity_temp'] = df['humidity'] * df['temp']
        df['rain_wind'] = df['rainfall'] * df['wind_speed']
        
        # 2. 순환 시간 피처 (더 많은 주기)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 3. 온도 관련 고급 피처
        df['temp_squared'] = df['temp'] ** 2
        df['temp_cubed'] = df['temp'] ** 3
        df['temp_category'] = pd.cut(df['temp'], bins=[-np.inf, 5, 15, 25, 35, np.inf], 
                                   labels=['freezing', 'cold', 'mild', 'warm', 'hot'])
        
        # 4. 습도 관련 피처
        df['humidity_squared'] = df['humidity'] ** 2
        df['humidity_category'] = pd.cut(df['humidity'], bins=[0, 30, 50, 70, 100], 
                                       labels=['dry', 'normal', 'humid', 'very_humid'])
        
        # 5. 복합 기상 피처
        df['heat_index'] = df['temp'] + 0.5 * (df['humidity'] / 100 - 1) * (df['temp'] - 14.5)
        df['comfort_index'] = df['temp'] * (1 - df['humidity'] / 100)
        df['weather_stress'] = np.abs(df['temp'] - 22) + np.abs(df['humidity'] - 50) / 100
        
        # 6. 건물 효율성 피처
        df['energy_efficiency'] = (df['pv_capacity'] + 1) / (df['total_area'] + 1) * 1000
        df['cooling_efficiency'] = df['cooling_area'] / (df['total_area'] + 1)
        df['storage_ratio'] = df['ess_capacity'] / (df['pv_capacity'] + 1)
        
        # 7. 시간별 복합 피처
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (~df['is_weekend'])).astype(int)
        df['is_peak_hour'] = df['hour'].isin([10, 11, 14, 15, 16, 17]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 8. 계절성 피처
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int) 
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_transition'] = df['month'].isin([3, 4, 5, 9, 10, 11]).astype(int)
        
        # 9. 상호작용 피처
        df['bld_hour_temp'] = df['bld_hour_mean'] * df['temp'] / 30
        df['bld_hour_humidity'] = df['bld_hour_mean'] * df['humidity'] / 100
        df['temp_area_interaction'] = df['temp'] * df['total_area'] / 10000
        df['hour_area_interaction'] = df['hour'] * df['total_area'] / 10000
        
        # 10. 로그 변환 피처 (큰 값들)
        df['log_total_area'] = np.log1p(df['total_area'])
        df['log_cooling_area'] = np.log1p(df['cooling_area'])
        df['log_pv_capacity'] = np.log1p(df['pv_capacity'])
    
    print(f"Advanced FE completed: {train.shape[1]} features")
    return train, test

def build_deep_ensemble(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> dict:
    """더 깊고 복잡한 앙상블 모델 구축."""
    print("Building deep ensemble model...")
    
    # 결과 저장 디렉토리 생성
    output_dir.mkdir(exist_ok=True)
    
    # 피처 준비
    feature_cols = [col for col in train.columns 
                   if col not in ['num_date_time', '건물번호', '일시', 'datetime', '전력소비량(kWh)']]
    
    X = train[feature_cols]
    y = train['전력소비량(kWh)']
    X_test = test[feature_cols]
    
    print(f"Features used: {len(feature_cols)}")
    
    # 전처리
    categorical_features = ['building_type', 'temp_category', 'humidity_category']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, [col for col in feature_cols if col not in categorical_features])
        ]
    )
    
    # 1. 더 깊은 XGBoost
    xgb_model = xgb.XGBRegressor(
        max_depth=15,  # 더 깊게
        n_estimators=2500,  # 더 많은 트리
        learning_rate=0.01,  # 더 낮은 학습률
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        min_child_weight=5,
        gamma=0.1,
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        random_state=42,
    )
    
    # 2. 더 깊은 LightGBM
    lgb_model = lgb.LGBMRegressor(
        max_depth=20,  # 더 깊게
        n_estimators=3000,  # 더 많은 트리
        learning_rate=0.008,  # 더 낮은 학습률
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        reg_lambda=3.5,
        num_leaves=500,  # 더 많은 잎
        min_child_samples=10,
        min_child_weight=0.01,
        device='gpu',
        gpu_use_dp=True,
        random_state=42,
        verbosity=-1
    )
    
    # 3. 더 깊은 CatBoost
    cat_model = cb.CatBoostRegressor(
        depth=12,  # 더 깊게
        iterations=2000,  # 더 많은 반복
        learning_rate=0.01,  # 더 낮은 학습률
        bootstrap_type='Bernoulli',
        subsample=0.8,
        reg_lambda=3.0,
        min_data_in_leaf=5,
        max_leaves=1000,  # 더 많은 잎
        task_type='GPU',
        gpu_ram_part=0.7,
        random_seed=42,
        verbose=False
    )
    
    # 4. Stacking Ensemble (더 복잡한 메타러너)
    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', Pipeline([('preprocess', preprocessor), ('model', xgb_model)])),
            ('lgb', Pipeline([('preprocess', preprocessor), ('model', lgb_model)])),
            ('cat', Pipeline([('preprocess', preprocessor), ('model', cat_model)]))
        ],
        final_estimator=Ridge(alpha=10.0),  # 정규화된 메타러너
        cv=3
    )
    
    # 검증 분할
    cutoff = pd.Timestamp('2024-08-18')
    train_mask = train['datetime'] < cutoff
    val_mask = ~train_mask
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}")
    
    # 모델 학습
    print("Training deep stacking ensemble...")
    stacking_model.fit(X_train, y_train)
    
    # 검증
    val_pred = stacking_model.predict(X_val)
    val_smape = smape(y_val.values, val_pred)
    
    print(f"Validation SMAPE: {val_smape:.4f}")
    
    # 피처 중요도 분석
    print("Analyzing feature importance...")
    analyze_feature_importance(stacking_model, feature_cols, categorical_features, output_dir)
    
    # 최종 학습
    print("Final training on full dataset...")
    stacking_model.fit(X, y)
    
    # 예측
    test_pred = stacking_model.predict(X_test)
    
    # 결과 저장
    results = {
        'validation_smape': val_smape,
        'model': stacking_model,
        'feature_cols': feature_cols,
        'predictions': test_pred
    }
    
    # 검증 결과 저장
    with open(output_dir / 'validation_results.txt', 'w') as f:
        f.write(f'Deep Ensemble Validation SMAPE: {val_smape:.6f}\n')
        f.write(f'Target: 5-6 SMAPE\n')
        f.write(f'Features used: {len(feature_cols)}\n')
        f.write(f'Model: Stacking(XGBoost + LightGBM + CatBoost)\n')
    
    # 제출 파일 저장
    submission = test[['num_date_time']].copy()
    submission['answer'] = test_pred
    submission.to_csv(output_dir / 'submission_deep_ensemble.csv', index=False)
    
    return results

def analyze_feature_importance(model, feature_cols, categorical_features, output_dir):
    """피처 중요도 분석 및 시각화."""
    try:
        # XGBoost 모델의 피처 중요도 추출
        xgb_model = model.named_estimators_['xgb']['model']
        
        # 전처리된 피처명 생성
        cat_transformer = model.named_estimators_['xgb']['preprocess'].named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(categorical_features)
        else:
            cat_features = [f"cat_{i}" for i in range(len(categorical_features) * 2)]  # 추정
        
        numeric_features = [col for col in feature_cols if col not in categorical_features]
        all_features = list(cat_features) + numeric_features
        
        # 피처 중요도 추출
        importance = xgb_model.feature_importances_
        
        # DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': all_features[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Top 30 피처 시각화
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(30)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 30 Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV 저장
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        print(f"Feature importance analysis saved to {output_dir}")
        print("Top 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")

def main():
    """메인 실행 함수."""
    print("=" * 70)
    print("ADVANCED DEEP ENSEMBLE: 8.08 SMAPE → 5-6 SMAPE")
    print("More features + Deeper models + Feature analysis")
    print("=" * 70)
    
    # 결과 저장 디렉토리
    output_dir = Path('solution_optimized_result')
    output_dir.mkdir(exist_ok=True)
    
    # 데이터 로드
    base_dir = Path('../data')
    train_path = base_dir / 'train.csv'
    test_path = base_dir / 'test.csv'
    building_path = base_dir / 'building_info.csv'
    
    print("Loading and engineering features...")
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = advanced_feature_engineering(train_df, test_df)
    
    print("Building deep ensemble model...")
    results = build_deep_ensemble(train_fe, test_fe, output_dir)
    
    # 결과 출력
    val_smape = results['validation_smape']
    print(f"\n🎯 Deep Ensemble Results:")
    print(f"📊 Validation SMAPE: {val_smape:.4f}")
    print(f"🎯 Target: 5-6 SMAPE")
    print(f"📁 Results saved to: {output_dir}")
    
    if val_smape < 6.0:
        print("🎉 SUCCESS! Target achieved!")
    elif val_smape < 7.0:
        print("✅ Good progress! Very close to target.")
    else:
        print("📈 Need further optimization.")
    
    print(f"\n📋 Files generated:")
    print(f"  - submission_deep_ensemble.csv")
    print(f"  - feature_importance.csv")
    print(f"  - feature_importance.png")
    print(f"  - validation_results.txt")

if __name__ == "__main__":
    main()