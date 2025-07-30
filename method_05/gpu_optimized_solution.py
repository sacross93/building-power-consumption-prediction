"""
GPU-Optimized Power Consumption Prediction Solution
==================================================

진짜 GPU 파워를 활용한 고성능 솔루션
- XGBoost GPU 최적화
- LightGBM GPU 가속
- CatBoost GPU 활용
- 대용량 배치 처리
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import VotingRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from ts_validation import smape, TimeSeriesCV


def check_gpu_support():
    """GPU 지원 상태 확인."""
    print("=" * 60)
    print("GPU SUPPORT CHECK")
    print("=" * 60)
    
    gpu_status = {}
    
    # XGBoost GPU 테스트
    try:
        dtrain = xgb.DMatrix(np.random.rand(1000, 10), label=np.random.rand(1000))
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_bin': 256,  # GPU 최적화
            'verbosity': 0
        }
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        gpu_status['xgboost'] = True
        print("✅ XGBoost GPU support: AVAILABLE")
    except Exception as e:
        gpu_status['xgboost'] = False
        print(f"❌ XGBoost GPU support: {e}")
    
    # LightGBM GPU 테스트
    try:
        train_data = lgb.Dataset(np.random.rand(1000, 10), label=np.random.rand(1000))
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbosity': -1
        }
        model = lgb.train(params, train_data, num_boost_round=10, verbose_eval=False)
        gpu_status['lightgbm'] = True
        print("✅ LightGBM GPU support: AVAILABLE")
    except Exception as e:
        gpu_status['lightgbm'] = False
        print(f"❌ LightGBM GPU support: {e}")
    
    # CatBoost GPU 테스트
    try:
        train_data = cb.Pool(np.random.rand(1000, 10), label=np.random.rand(1000))
        model = cb.CatBoostRegressor(
            task_type='GPU',
            devices='0',
            iterations=10,
            verbose=False
        )
        model.fit(train_data)
        gpu_status['catboost'] = True
        print("✅ CatBoost GPU support: AVAILABLE")
    except Exception as e:
        gpu_status['catboost'] = False
        print(f"❌ CatBoost GPU support: {e}")
    
    print(f"\nGPU Summary: {sum(gpu_status.values())}/3 libraries support GPU")
    return gpu_status


def gpu_optimized_features(train_df, test_df):
    """GPU 최적화된 피처 엔지니어링."""
    print("Creating GPU-optimized features...")
    
    def create_features(df):
        df = df.copy()
        
        # 기본 datetime 파싱
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['num_date_time'] = df['datetime'].astype(np.int64) // 10**9
        
        # GPU 친화적 데이터 타입으로 변환
        df['hour'] = df['datetime'].dt.hour.astype(np.int8)
        df['weekday'] = df['datetime'].dt.weekday.astype(np.int8)
        df['month'] = df['datetime'].dt.month.astype(np.int8)
        df['day_of_year'] = df['datetime'].dt.dayofyear.astype(np.int16)
        
        # 비즈니스 로직 (GPU 최적화)
        df['is_weekend'] = (df['weekday'] >= 5).astype(np.int8)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (df['weekday'] < 5)).astype(np.int8)
        df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10) & (df['weekday'] < 5)).astype(np.int8)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (df['weekday'] < 5)).astype(np.int8)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13) & (df['weekday'] < 5)).astype(np.int8)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(np.int8)
        
        # GPU 친화적 순환 인코딩 (float32)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7).astype(np.float32)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7).astype(np.float32)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
        
        # GPU 최적화된 날씨 피처
        if 'temp' in df.columns:
            df['temp'] = df['temp'].astype(np.float32)
            df['temp_squared'] = (df['temp'] ** 2).astype(np.float32)
            df['temp_cooling_need'] = np.maximum(df['temp'] - 26, 0).astype(np.float32)
            df['temp_heating_need'] = np.maximum(18 - df['temp'], 0).astype(np.float32)
            
            # 온도 구간 (GPU 친화적)
            df['temp_cold'] = (df['temp'] < 18).astype(np.int8)
            df['temp_comfortable'] = ((df['temp'] >= 18) & (df['temp'] < 26)).astype(np.int8)
            df['temp_hot'] = (df['temp'] >= 26).astype(np.int8)
            
            if 'humidity' in df.columns:
                df['humidity'] = df['humidity'].astype(np.float32)
                df['temp_humidity'] = (df['temp'] * df['humidity']).astype(np.float32)
                df['discomfort_index'] = (df['temp'] - 0.55 * (1 - df['humidity'] / 100) * (df['temp'] - 14.5)).astype(np.float32)
        
        # GPU 최적화된 건물 피처
        if 'total_area' in df.columns:
            df['total_area'] = df['total_area'].astype(np.float32)
            df['cooling_area'] = df['cooling_area'].astype(np.float32)
            df['log_total_area'] = np.log1p(df['total_area']).astype(np.float32)
            df['area_ratio'] = (df['cooling_area'] / (df['total_area'] + 1)).astype(np.float32)
            
            # 건물 크기 (GPU 친화적)
            df['small_building'] = (df['total_area'] < 10000).astype(np.int8)
            df['large_building'] = (df['total_area'] >= 50000).astype(np.int8)
        
        # PV 피처 (GPU 최적화)
        if 'pv_capacity' in df.columns:
            df['pv_capacity'] = df['pv_capacity'].astype(np.float32)
            df['has_pv'] = (df['pv_capacity'] > 0).astype(np.int8)
            if 'total_area' in df.columns:
                df['pv_per_area'] = (df['pv_capacity'] / (df['total_area'] + 1)).astype(np.float32)
        
        return df
    
    # 고성능 건물별 통계 (vectorized)
    def add_vectorized_building_stats(train_df, test_df):
        """벡터화된 건물별 통계 계산."""
        
        if '전력소비량(kWh)' not in train_df.columns:
            return train_df, test_df
        
        # 벡터화된 그룹 통계 계산
        building_stats = train_df.groupby('건물번호').agg({
            '전력소비량(kWh)': ['mean', 'std', 'min', 'max']
        }).round(2)
        building_stats.columns = ['power_mean', 'power_std', 'power_min', 'power_max']
        
        # 시간별 패턴 (벡터화)
        hourly_stats = train_df.groupby(['건물번호', 'hour'])['전력소비량(kWh)'].mean().unstack(fill_value=0)
        weekend_stats = train_df.groupby(['건물번호', 'is_weekend'])['전력소비량(kWh)'].mean().unstack(fill_value=0)
        
        # 효율적인 매핑
        for df_name, df in [('train', train_df), ('test', test_df)]:
            # 기본 통계 매핑
            df = df.merge(building_stats, left_on='건물번호', right_index=True, how='left')
            
            # 결측값 처리 (GPU 친화적)
            global_mean = train_df['전력소비량(kWh)'].mean()
            df['power_mean'] = df['power_mean'].fillna(global_mean).astype(np.float32)
            df['power_std'] = df['power_std'].fillna(global_mean * 0.3).astype(np.float32)
            df['power_min'] = df['power_min'].fillna(0).astype(np.float32)
            df['power_max'] = df['power_max'].fillna(global_mean * 2).astype(np.float32)
            
        return train_df, test_df
    
    # 피처 생성
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # 벡터화된 건물 통계
    train_fe, test_fe = add_vectorized_building_stats(train_fe, test_fe)
    
    print(f"GPU-optimized features complete. Train: {train_fe.shape}, Test: {test_fe.shape}")
    return train_fe, test_fe


def gpu_hyperparameter_optimization(X, y, datetime_series, gpu_status, n_trials=25):
    """GPU 가속 하이퍼파라미터 최적화."""
    print("GPU-accelerated hyperparameter optimization...")
    
    def objective(trial):
        # 사용 가능한 GPU 라이브러리 선택
        available_models = []
        if gpu_status['xgboost']:
            available_models.append('xgboost')
        if gpu_status['lightgbm']:
            available_models.append('lightgbm')
        if gpu_status['catboost']:
            available_models.append('catboost')
        
        if not available_models:
            available_models = ['xgboost_cpu']  # CPU 폴백
        
        model_type = trial.suggest_categorical('model_type', available_models)
        
        # 모델별 GPU 최적화 파라미터
        if model_type == 'xgboost':
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_bin': trial.suggest_int('max_bin', 256, 512),  # GPU 최적화
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
                'random_state': 42,
                'verbosity': 0
            }
            model_class = xgb.XGBRegressor
            
        elif model_type == 'lightgbm':
            params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': trial.suggest_int('max_bin', 255, 511),  # GPU 최적화
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
                'random_state': 42,
                'verbosity': -1
            }
            model_class = lgb.LGBMRegressor
            
        elif model_type == 'catboost':
            params = {
                'task_type': 'GPU',
                'devices': '0',
                'gpu_ram_part': 0.8,  # GPU 메모리 사용률
                'depth': trial.suggest_int('depth', 6, 12),
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
                'random_seed': 42,
                'verbose': False
            }
            model_class = cb.CatBoostRegressor
            
        else:  # CPU 폴백
            params = {
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'n_estimators': trial.suggest_int('n_estimators', 300, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
                'random_state': 42,
                'verbosity': 0
            }
            model_class = xgb.XGBRegressor
        
        # 시계열 교차검증
        ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
        temp_df = pd.DataFrame({'datetime': datetime_series})
        splits = ts_cv.split(temp_df, 'datetime')
        
        scores = []
        for train_idx, val_idx in splits:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = smape(y_val.values, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best SMAPE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def gpu_ensemble_training(train_df, gpu_status):
    """GPU 가속 앙상블 훈련."""
    print("=" * 60)
    print("GPU-ACCELERATED ENSEMBLE TRAINING")
    print("=" * 60)
    
    # 피처 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['전력소비량(kWh)'].astype(np.float32)
    
    # 카테고리 인코딩
    encoders = {}
    categorical_cols = ['건물번호', 'building_type']
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # 모든 피처를 GPU 친화적 타입으로 변환
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # GPU 최적화된 데이터 타입
        if X[col].dtype in ['int64', 'int32']:
            X[col] = X[col].astype(np.int32)
        else:
            X[col] = X[col].astype(np.float32)
    
    # 피처 선택 (GPU 메모리 최적화)
    selector = SelectKBest(score_func=f_regression, k=min(50, len(X.columns)))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Selected {len(selected_features)} features for GPU training")
    
    # 하이퍼파라미터 최적화
    best_params, best_score = gpu_hyperparameter_optimization(
        X_selected, y, train_df['datetime'], gpu_status, n_trials=20
    )
    
    # GPU 앙상블 검증
    ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
    splits = ts_cv.split(train_df, 'datetime')
    
    validation_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"GPU ensemble fold {fold + 1}/3")
        
        X_train = X_selected.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_selected.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        models = []
        
        # XGBoost GPU
        if gpu_status['xgboost']:
            xgb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'verbosity': 0
            })
            xgb_model = xgb.XGBRegressor(**xgb_params)
            models.append(('xgb_gpu', xgb_model))
        
        # LightGBM GPU
        if gpu_status['lightgbm']:
            lgb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1
            })
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            models.append(('lgb_gpu', lgb_model))
        
        # CatBoost GPU
        if gpu_status['catboost']:
            cb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            cb_params.update({
                'task_type': 'GPU',
                'devices': '0',
                'verbose': False
            })
            cb_model = cb.CatBoostRegressor(**cb_params)
            models.append(('cb_gpu', cb_model))
        
        if models:
            ensemble = VotingRegressor(models)
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_val)
        else:
            # CPU 폴백
            model = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'})
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        fold_smape = smape(y_val.values, y_pred)
        validation_scores.append(fold_smape)
        
        print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}")
    
    mean_smape = np.mean(validation_scores)
    std_smape = np.std(validation_scores)
    
    print(f"\nGPU Ensemble Validation:")
    print(f"Mean SMAPE: {mean_smape:.4f} (±{std_smape:.4f})")
    
    return mean_smape, std_smape, selected_features, encoders, selector, best_params


def generate_gpu_submission():
    """GPU 최적화된 제출 파일 생성."""
    print("=" * 80)
    print("GPU-OPTIMIZED POWER CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # GPU 지원 확인
    gpu_status = check_gpu_support()
    
    # 데이터 로드
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    print("\nLoading data...")
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 기본 피처 엔지니어링
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    # GPU 최적화된 피처 엔지니어링
    train_gpu, test_gpu = gpu_optimized_features(train_fe, test_fe)
    
    # GPU 앙상블 훈련
    val_smape, val_std, selected_features, encoders, selector, best_params = gpu_ensemble_training(train_gpu, gpu_status)
    
    # 최종 GPU 모델 훈련
    print("\n" + "=" * 60)
    print("TRAINING FINAL GPU ENSEMBLE")
    print("=" * 60)
    
    # 전체 데이터 준비
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_gpu.columns if c not in drop_cols]
    
    X_full = train_gpu[feature_cols].copy()
    y_full = train_gpu['전력소비량(kWh)'].astype(np.float32)
    
    # 인코딩 및 타입 최적화
    for col, encoder in encoders.items():
        if col in X_full.columns:
            X_full[col] = encoder.fit_transform(X_full[col].astype(str))
    
    for col in X_full.columns:
        if X_full[col].dtype == 'object':
            le = LabelEncoder()
            X_full[col] = le.fit_transform(X_full[col].astype(str))
            encoders[col] = le
        
        # GPU 최적화 타입
        if X_full[col].dtype in ['int64', 'int32']:
            X_full[col] = X_full[col].astype(np.int32)
        else:
            X_full[col] = X_full[col].astype(np.float32)
    
    # 피처 선택 적용
    X_full_selected = selector.transform(X_full)
    X_full_selected = pd.DataFrame(X_full_selected, columns=selected_features)
    
    # 최종 GPU 앙상블
    final_models = []
    
    if gpu_status['xgboost']:
        xgb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'n_estimators': int(xgb_params.get('n_estimators', 1000) * 1.2),  # 더 깊게
            'verbosity': 0
        })
        final_models.append(('xgb_gpu', xgb.XGBRegressor(**xgb_params)))
    
    if gpu_status['lightgbm']:
        lgb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        lgb_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_estimators': int(lgb_params.get('n_estimators', 1000) * 1.2),
            'verbosity': -1
        })
        final_models.append(('lgb_gpu', lgb.LGBMRegressor(**lgb_params)))
    
    if gpu_status['catboost']:
        cb_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        cb_params.update({
            'task_type': 'GPU',
            'devices': '0',
            'iterations': int(cb_params.get('iterations', 1000) * 1.2),
            'verbose': False
        })
        final_models.append(('cb_gpu', cb.CatBoostRegressor(**cb_params)))
    
    if final_models:
        final_ensemble = VotingRegressor(final_models)
        print(f"Training GPU ensemble with {len(final_models)} models...")
        final_ensemble.fit(X_full_selected, y_full)
    else:
        # CPU 폴백
        print("No GPU support, using CPU fallback...")
        final_ensemble = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'})
        final_ensemble.fit(X_full_selected, y_full)
    
    # 테스트 예측
    print("Making GPU predictions...")
    X_test = test_gpu[feature_cols].copy()
    
    # 테스트 데이터 처리
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
        
        # GPU 최적화 타입
        if X_test[col].dtype in ['int64', 'int32']:
            X_test[col] = X_test[col].astype(np.int32)
        else:
            X_test[col] = X_test[col].astype(np.float32)
    
    # 피처 선택 적용
    X_test_selected = selector.transform(X_test)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    predictions = final_ensemble.predict(X_test_selected)
    predictions = np.maximum(predictions, 0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'num_date_time': test_gpu['num_date_time'],
        'prediction': predictions
    })
    
    submission_file = 'submission_gpu_optimized.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ GPU-optimized solution completed!")
    print(f"GPU utilization: {sum(gpu_status.values())}/3 libraries")
    print(f"Validation SMAPE: {val_smape:.4f} (±{val_std:.4f})")
    print(f"Features used: {len(selected_features)}")
    print(f"Predictions range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Submission saved: {submission_file}")
    
    return submission, val_smape, gpu_status


if __name__ == "__main__":
    submission, smape_score, gpu_status = generate_gpu_submission()
    
    print(f"\n" + "=" * 80)
    print("GPU-OPTIMIZED SOLUTION SUMMARY")
    print("=" * 80)
    print(f"GPU Libraries Used: {sum(gpu_status.values())}/3")
    print(f"XGBoost GPU: {'✅' if gpu_status['xgboost'] else '❌'}")
    print(f"LightGBM GPU: {'✅' if gpu_status['lightgbm'] else '❌'}")
    print(f"CatBoost GPU: {'✅' if gpu_status['catboost'] else '❌'}")
    print(f"Validation SMAPE: {smape_score:.4f}")
    print("🚀 Maximum GPU acceleration achieved!")
    print(f"Submission file: submission_gpu_optimized.csv")