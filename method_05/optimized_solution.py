"""
Optimized Power Consumption Prediction Solution
==============================================

피처 선택 + 깊은 학습 + 하이퍼파라미터 최적화
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import VotingRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def smart_feature_engineering(train_df, test_df):
    """스마트한 피처 엔지니어링 - 중요한 피처만 선별."""
    print("Creating smart features...")
    
    def create_features(df):
        df = df.copy()
        
        # 기본 datetime 파싱
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # 핵심 시간 피처
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # 핵심 비즈니스 로직
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (df['weekday'] < 5)).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10) & (df['weekday'] < 5)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (df['weekday'] < 5)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13) & (df['weekday'] < 5)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 핵심 순환 인코딩
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 핵심 날씨 피처
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0)
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0)
            
            # 온도 구간 (핵심만)
            df['temp_cold'] = (df['temp'] < 18).astype(int)
            df['temp_comfortable'] = ((df['temp'] >= 18) & (df['temp'] < 26)).astype(int)
            df['temp_hot'] = (df['temp'] >= 26).astype(int)
            
            if 'humidity' in df.columns:
                df['temp_humidity'] = df['temp'] * df['humidity']
                df['discomfort_index'] = df['temp'] - 0.55 * (1 - df['humidity'] / 100) * (df['temp'] - 14.5)
        
        # 핵심 건물 피처
        if 'total_area' in df.columns:
            df['log_total_area'] = np.log1p(df['total_area'])
            df['area_ratio'] = df['cooling_area'] / (df['total_area'] + 1)
            
            # 건물 크기 (핵심 구간만)
            df['small_building'] = (df['total_area'] < 10000).astype(int)
            df['large_building'] = (df['total_area'] >= 50000).astype(int)
        
        # PV 피처
        if 'pv_capacity' in df.columns:
            df['has_pv'] = (df['pv_capacity'] > 0).astype(int)
            if 'total_area' in df.columns:
                df['pv_per_area'] = df['pv_capacity'] / (df['total_area'] + 1)
        
        return df
    
    # 안전한 건물별 핵심 통계
    def add_building_core_stats(train_df, test_df):
        """핵심 건물별 통계만 추가."""
        
        building_stats = {}
        
        for building_id in train_df['건물번호'].unique():
            building_data = train_df[train_df['건물번호'] == building_id]
            
            if '전력소비량(kWh)' in building_data.columns:
                # 핵심 패턴만
                hourly_pattern = building_data.groupby('hour')['전력소비량(kWh)'].mean()
                weekend_pattern = building_data.groupby('is_weekend')['전력소비량(kWh)'].mean()
                
                power_mean = building_data['전력소비량(kWh)'].mean()
                power_std = building_data['전력소비량(kWh)'].std()
                
                building_stats[building_id] = {
                    'hourly_pattern': hourly_pattern,
                    'weekend_pattern': weekend_pattern,
                    'power_mean': power_mean,
                    'power_std': power_std
                }
        
        # 적용
        for df_name, df in [('train', train_df), ('test', test_df)]:
            for building_id in df['건물번호'].unique():
                mask = df['건물번호'] == building_id
                
                if building_id in building_stats:
                    stats = building_stats[building_id]
                    
                    df.loc[mask, 'building_hour_avg'] = df.loc[mask, 'hour'].map(
                        stats['hourly_pattern']
                    ).fillna(stats['power_mean'])
                    
                    df.loc[mask, 'building_weekend_avg'] = df.loc[mask, 'is_weekend'].map(
                        stats['weekend_pattern']
                    ).fillna(stats['power_mean'])
                    
                    df.loc[mask, 'building_power_mean'] = stats['power_mean']
                    df.loc[mask, 'building_power_std'] = stats['power_std']
                else:
                    global_mean = train_df['전력소비량(kWh)'].mean() if '전력소비량(kWh)' in train_df.columns else 1000
                    df.loc[mask, 'building_hour_avg'] = global_mean
                    df.loc[mask, 'building_weekend_avg'] = global_mean
                    df.loc[mask, 'building_power_mean'] = global_mean
                    df.loc[mask, 'building_power_std'] = global_mean * 0.3
        
        return train_df, test_df
    
    # 피처 생성
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # 건물별 통계 추가
    train_fe, test_fe = add_building_core_stats(train_fe, test_fe)
    
    print(f"Smart feature engineering complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def feature_selection(X, y, k=50):
    """중요한 피처만 선택."""
    print(f"Selecting top {k} features...")
    
    # 수치형 피처만 선택
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    # SelectKBest로 피처 선택
    selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_cols)))
    X_selected = selector.fit_transform(X_numeric, y)
    
    # 선택된 피처명
    selected_features = numeric_cols[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features:")
    print(f"Top 10: {list(selected_features[:10])}")
    
    return X[selected_features], selected_features, selector


def optimize_hyperparameters(X, y, datetime_series, n_trials=20):
    """하이퍼파라미터 최적화."""
    print("Optimizing hyperparameters...")
    
    def objective(trial):
        # XGBoost 파라미터
        params = {
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
            'random_state': 42,
            'verbosity': 0
        }
        
        # 시계열 교차검증
        ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
        
        # 임시 DataFrame for splits
        temp_df = pd.DataFrame({'datetime': datetime_series})
        splits = ts_cv.split(temp_df, 'datetime')
        
        scores = []
        for train_idx, val_idx in splits:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = smape(y_val.values, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best SMAPE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def deep_ensemble_training(train_df):
    """깊은 학습 앙상블 훈련."""
    print("=" * 60)
    print("DEEP ENSEMBLE TRAINING WITH FEATURE SELECTION")
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
    
    # 피처 선택
    X_selected, selected_features, feature_selector = feature_selection(X, y, k=40)
    
    # 하이퍼파라미터 최적화
    best_params = optimize_hyperparameters(X_selected, y, train_df['datetime'], n_trials=15)
    
    # 검증
    ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_df, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Deep ensemble fold {fold + 1}/3")
        
        X_train = X_selected.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_selected.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # 최적화된 XGBoost
        xgb_model = xgb.XGBRegressor(**best_params)
        
        # 최적화된 LightGBM
        lgb_params = best_params.copy()
        lgb_params.pop('verbosity', None)  # LightGBM 호환
        lgb_model = lgb.LGBMRegressor(**lgb_params, verbosity=-1)
        
        # 깊은 XGBoost (더 많은 estimators)
        deep_params = best_params.copy()
        deep_params['n_estimators'] = int(deep_params['n_estimators'] * 1.5)
        deep_params['learning_rate'] = deep_params['learning_rate'] * 0.7
        xgb_deep = xgb.XGBRegressor(**deep_params)
        
        # 3-모델 앙상블
        ensemble = VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('xgb_deep', xgb_deep)
        ], weights=[0.4, 0.3, 0.3])
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)
        
        fold_smape = smape(y_val.values, y_pred)
        validation_scores.append(fold_smape)
        
        print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    mean_smape = np.mean(validation_scores)
    std_smape = np.std(validation_scores)
    
    print(f"\nDeep Ensemble Validation:")
    print(f"Mean SMAPE: {mean_smape:.4f} (±{std_smape:.4f})")
    
    return mean_smape, std_smape, selected_features, encoders, feature_selector, best_params


def generate_optimized_submission():
    """최적화된 제출 파일 생성."""
    print("=" * 80)
    print("OPTIMIZED POWER CONSUMPTION PREDICTION")
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
    
    # 스마트 피처 엔지니어링
    train_smart, test_smart = smart_feature_engineering(train_fe, test_fe)
    
    # 깊은 앙상블 검증
    val_smape, val_std, selected_features, encoders, feature_selector, best_params = deep_ensemble_training(train_smart)
    
    # 최종 모델 훈련
    print("\n" + "=" * 60)
    print("TRAINING FINAL OPTIMIZED ENSEMBLE")
    print("=" * 60)
    
    # 전체 데이터 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_smart.columns if c not in drop_cols]
    
    X_full = train_smart[feature_cols].copy()
    y_full = train_smart['전력소비량(kWh)']
    
    # 인코딩 적용
    for col, encoder in encoders.items():
        if col in X_full.columns:
            X_full[col] = encoder.fit_transform(X_full[col].astype(str))
    
    for col in X_full.columns:
        if X_full[col].dtype == 'object':
            le = LabelEncoder()
            X_full[col] = le.fit_transform(X_full[col].astype(str))
            encoders[col] = le
    
    # 피처 선택 적용
    X_full_selected = feature_selector.transform(X_full.select_dtypes(include=[np.number]))
    X_full_selected = pd.DataFrame(X_full_selected, columns=selected_features)
    
    # 최종 3-모델 앙상블
    xgb_final = xgb.XGBRegressor(**best_params)
    
    lgb_params = best_params.copy()
    lgb_params.pop('verbosity', None)
    lgb_final = lgb.LGBMRegressor(**lgb_params, verbosity=-1)
    
    deep_params = best_params.copy()
    deep_params['n_estimators'] = int(deep_params['n_estimators'] * 1.5)
    deep_params['learning_rate'] = deep_params['learning_rate'] * 0.7
    xgb_deep_final = xgb.XGBRegressor(**deep_params)
    
    final_ensemble = VotingRegressor([
        ('xgb', xgb_final),
        ('lgb', lgb_final),
        ('xgb_deep', xgb_deep_final)
    ], weights=[0.4, 0.3, 0.3])
    
    final_ensemble.fit(X_full_selected, y_full)
    
    # 테스트 예측
    X_test = test_smart[feature_cols].copy()
    
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
    
    # 피처 선택 적용
    X_test_selected = feature_selector.transform(X_test.select_dtypes(include=[np.number]))
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    predictions = final_ensemble.predict(X_test_selected)
    predictions = np.maximum(predictions, 0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_smart['num_date_time'],
        'prediction': predictions
    })
    
    submission_file = 'submission_optimized.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Optimized solution completed!")
    print(f"Validation SMAPE: {val_smape:.4f} (±{val_std:.4f})")
    print(f"Features used: {len(selected_features)}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: {submission_file}")
    
    return submission, val_smape


if __name__ == "__main__":
    submission, smape_score = generate_optimized_submission()
    
    print(f"\n" + "=" * 80)
    print("OPTIMIZED SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Validation SMAPE: {smape_score:.4f}")
    print("✅ Feature Selection + Deep Learning + Hyperparameter Optimization")
    print(f"Submission file: submission_optimized.csv")