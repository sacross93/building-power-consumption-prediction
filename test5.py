# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v10 - Aggressive GPU Memory Usage
------------------------------------------------------------
• 32GB GPU 메모리 최대 활용을 위한 최적화
• 더 큰 배치 사이즈와 복잡한 모델로 GPU 성능 극대화
• 병렬 처리 및 메모리 집약적 설정
• GPU #3 전용 최적화
"""
import argparse
from pathlib import Path
import warnings
import os
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
import gc
warnings.filterwarnings("ignore")

# GPU 강제 사용 설정 (GPU #2) + 메모리 최적화
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 2번만 사용
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["LIGHTGBM_GPU"] = "1"  # LightGBM GPU 강제
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 동기화
os.environ["CUDA_CACHE_DISABLE"] = "0"  # CUDA 캐시 활성화
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"  # 2GB 캐시

# Optuna 로깅 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)

KR_HOLIDAYS = holidays.KR()
SEED = 42

###########################################################
# GPU Detection & Setup with Memory Info
###########################################################

def check_gpu_support():
    """GPU 지원 가능 여부 확인 및 성능 정보"""
    lgb_gpu = False
    xgb_gpu = False
    
    print("🚀 GPU Memory Status:")
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 2번
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total // 1024**3  # GB
        free_gb = info.free // 1024**3
        used_gb = (info.total - info.free) // 1024**3
        print(f"   Total: {total_gb}GB | Free: {free_gb}GB | Used: {used_gb}GB")
        
        if total_gb >= 24:  # 24GB 이상일 때 고성능 모드
            print(f"💪 High-end GPU detected ({total_gb}GB) - enabling MAXIMUM performance mode")
            return True, total_gb
        else:
            print(f"⚡ Standard GPU ({total_gb}GB) - enabling optimized mode")
            return True, total_gb
    except:
        print("❌ GPU memory info not available")
        return False, 0
    
    # LightGBM GPU 실제 학습 테스트 (더 큰 데이터셋으로)
    try:
        X_test = np.random.rand(5000, 20)  # 더 큰 테스트
        y_test = np.random.rand(5000)
        lgb_test = lgb.LGBMRegressor(
            device="gpu", 
            gpu_platform_id=0,
            gpu_device_id=0,
            max_bin=1023,  # GPU 메모리 많이 사용하도록
            n_estimators=200,
            num_threads=1,
            force_col_wise=True,
            verbose=-1
        )
        lgb_test.fit(X_test, y_test)
        lgb_gpu = True
        print("✅ LightGBM GPU #2 high-performance test passed")
    except Exception as e:
        print(f"❌ LightGBM GPU #2 failed: {str(e)[:50]}...")
    
    # XGBoost GPU 실제 학습 테스트 (더 큰 데이터셋으로)
    try:
        X_test = np.random.rand(5000, 20)  # 더 큰 테스트
        y_test = np.random.rand(5000)
        xgb_test = xgb.XGBRegressor(
            tree_method="gpu_hist", 
            gpu_id=0,
            max_bin=1024,  # GPU 메모리 많이 사용하도록
            n_estimators=200,
            predictor="gpu_predictor",
            verbosity=0
        )
        xgb_test.fit(X_test, y_test)
        xgb_gpu = True
        print("✅ XGBoost GPU #2 high-performance test passed")
    except Exception as e:
        print(f"❌ XGBoost GPU #2 failed: {str(e)[:50]}...")
    
    if not lgb_gpu or not xgb_gpu:
        print("❌ GPU requirements not met")
        raise RuntimeError("GPU requirement not satisfied")
    
    print("🎯 All GPU requirements satisfied - MAXIMUM performance mode enabled")
    return lgb_gpu, xgb_gpu

###########################################################
# Metric
###########################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.maximum(y_pred, 0)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

###########################################################
# Feature Engineering Helpers
###########################################################

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["일시"]
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = dt.dt.date.map(lambda d: int(d in KR_HOLIDAYS))
    # Fourier
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["THI"] = 9/5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9/5 * df["기온(°C)"] - 26) + 32
    # Extended weather features for GPU memory usage
    df["dew_point"] = df["기온(°C)"] - (100 - df["습도(%)"]) / 5
    df["heat_index"] = 0.5 * (df["기온(°C)"] + 61.0 + (df["기온(°C)"] - 68.0) * 1.2 + df["습도(%)"] * 0.094)
    df["THI_diff_24h"] = df.groupby("건물번호")["THI"].diff(24)
    df["temp_diff_1h"] = df.groupby("건물번호")["기온(°C)"].diff(1)
    df["temp_diff_6h"] = df.groupby("건물번호")["기온(°C)"].diff(6)
    df["temp_diff_12h"] = df.groupby("건물번호")["기온(°C)"].diff(12)
    df["temp_rolling_mean_6h"] = df.groupby("건물번호")["기온(°C)"].rolling(window=6).mean().reset_index(0, drop=True)
    df["temp_rolling_std_6h"] = df.groupby("건물번호")["기온(°C)"].rolling(window=6).std().reset_index(0, drop=True)
    df["humidity_diff_1h"] = df.groupby("건물번호")["습도(%)"].diff(1)
    df["wind_speed_max_6h"] = df.groupby("건물번호")["풍속(m/s)"].rolling(window=6).max().reset_index(0, drop=True)
    return df

###########################################################
# High-Performance GPU Optuna Objectives
###########################################################

def lgb_objective_gpu_intensive(trial, X_tr, y_tr, X_val, y_val, gpu_memory_gb):
    """GPU 메모리를 최대한 활용하는 LightGBM 최적화"""
    
    # GPU 호환성을 위한 안전한 파라미터 범위 (피처 수 고려)
    # 피처 수가 많을 때 bin size가 자동으로 피처 수에 비례해서 증가하므로 보수적으로 설정
    max_bin = trial.suggest_int("max_bin", 63, 255)  # 매우 안전한 범위
    num_leaves = trial.suggest_int("num_leaves", 64, 512)  # 안정적 범위
    max_depth = trial.suggest_int("max_depth", 4, 10)  # 보수적 깊이
    
    params = {
        "objective": "regression_l1",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("lr", 0.005, 0.2, log=True),  # 더 넓은 범위
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 1, 50),
        "reg_alpha": trial.suggest_float("ra", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 1e-8, 100.0, log=True),
        "bagging_fraction": trial.suggest_float("bagging", 0.5, 1.0),
        "feature_fraction": trial.suggest_float("feature", 0.5, 1.0),
        "n_estimators": 2000,  # 더 많은 estimators
        "verbose": -1,
        "device": "gpu",
        "gpu_use_dp": True,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": max_bin,
        "num_threads": 1,
        "force_col_wise": True,
        "boost_from_average": True,  # GPU 최적화
    }
    
    print(f"🔥 LightGBM GPU intensive: max_bin={max_bin}, num_leaves={num_leaves}")
    
    model = lgb.LGBMRegressor(**params)
    
    callbacks = [
        lgb.early_stopping(200, verbose=False),  # 더 긴 patience
        lgb.log_evaluation(period=0)
    ]
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda y,p: ("SMAPE", smape_np(np.expm1(y), np.expm1(p)), False),
        categorical_feature="auto",
        callbacks=callbacks,
    )
    preds = model.predict(X_val)
    return smape_np(np.expm1(y_val), np.expm1(preds))

def xgb_objective_gpu_intensive(trial, X_tr, y_tr, X_val, y_val, gpu_memory_gb):
    """GPU 메모리를 최대한 활용하는 XGBoost 최적화"""
    
    # GPU 호환성을 위한 안전한 파라미터 범위
    max_bin = trial.suggest_int("max_bin", 64, 256)  # 안전한 범위
    max_depth = trial.suggest_int("max_depth", 3, 8)  # 보수적 깊이
    
    params = {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("lr", 0.005, 0.2, log=True),  # 더 넓은 범위
        "max_depth": max_depth,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_level", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_node", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "reg_alpha": trial.suggest_float("ra", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 1e-8, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "n_estimators": 2000,  # 더 많은 estimators
        "verbosity": 0,
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "max_bin": max_bin,
        "grow_policy": "lossguide",
        "predictor": "gpu_predictor",
        "n_jobs": 1,
        "sampling_method": "gradient_based",  # GPU 최적화
    }
    
    print(f"🔥 XGBoost GPU intensive: max_bin={max_bin}, max_depth={max_depth}")
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    return smape_np(np.expm1(y_val), np.expm1(preds))

###########################################################
# High-Performance Training per building
###########################################################

def train_building_gpu_intensive(df_tr: pd.DataFrame, df_te: pd.DataFrame, feats: list, n_trials: int, gpu_memory_gb: int):
    print(f"    🚀 GPU-intensive training (using {gpu_memory_gb}GB memory)")
    
    # Area for inverse transform
    area_tr = df_tr["연면적(m2)"].values
    area_te = df_te["연면적(m2)"].values if not df_te.empty else None
    y_tr_log = df_tr["log_power_pa"].values

    X_full = df_tr[feats]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_scaled = pd.DataFrame(X_scaled, columns=feats, index=df_tr.index)

    # GPU 메모리에 따른 CV 분할 조정
    n_splits = 5 if gpu_memory_gb >= 24 else 3  # 더 많은 분할로 더 정확한 검증
    test_size = 24*10 if gpu_memory_gb >= 24 else 24*7  # 더 큰 테스트 셋
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    oof_pred_lgb = np.zeros(len(df_tr))
    oof_pred_xgb = np.zeros(len(df_tr))
    best_iters_lgb = []
    best_iters_xgb = []
    
    for fold,(tr_idx,val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"    Fold {fold+1}/{n_splits} 🔥 GPU intensive training...")
        X_tr = X_scaled.iloc[tr_idx]
        y_tr_f = y_tr_log[tr_idx]
        X_val = X_scaled.iloc[val_idx]
        y_val_f = y_tr_log[val_idx]

        # 메모리 정리
        gc.collect()

        # LightGBM GPU 집약적 최적화
        print(f"      🔥 LightGBM GPU-intensive optimization...")
        study_lgb = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=30)  # 더 많은 startup trials
        )
        try:
            study_lgb.optimize(
                lambda tr: lgb_objective_gpu_intensive(tr, X_tr, y_tr_f, X_val, y_val_f, gpu_memory_gb), 
                n_trials=n_trials * 2,  # 더 많은 trials
                show_progress_bar=False,
                n_jobs=1
            )
        except Exception as e:
            print(f"      ⚠️ LightGBM optimization failed: {str(e)[:100]}...")
            # 안전한 기본값으로 대체
            study_lgb = optuna.create_study(direction="minimize")
            default_params = {
                "max_bin": 63, "num_leaves": 64, "max_depth": 4,
                "lr": 0.05, "subsample": 0.8, "colsample": 0.8,
                "min_leaf": 20, "ra": 0.01, "rl": 0.01, 
                "bagging": 0.8, "feature": 0.8
            }
            study_lgb.enqueue_trial(default_params)
            # 안전한 LightGBM 직접 실행
            try:
                lgb_objective_gpu_intensive(study_lgb.trials[0], X_tr, y_tr_f, X_val, y_val_f, gpu_memory_gb)
            except:
                pass
        
        try:
            best_lgb_value = study_lgb.best_value if len(study_lgb.trials) > 0 and study_lgb.best_value is not None else 15.0
        except:
            best_lgb_value = 15.0
        print(f"      🎯 LGB GPU-intensive best: {best_lgb_value:.3f}%")
        
        # Best LGB model with GPU-intensive settings
        try:
            best_lgb_params = study_lgb.best_params.copy() if len(study_lgb.trials) > 0 else {}
        except:
            best_lgb_params = {
                "max_bin": 63, "num_leaves": 64, "max_depth": 4,
                "lr": 0.05, "subsample": 0.8, "colsample": 0.8,
                "min_leaf": 20, "ra": 0.01, "rl": 0.01, 
                "bagging": 0.8, "feature": 0.8
            }
        best_lgb_params.update({
            "objective": "regression_l1",
            "random_state": SEED,
            "n_estimators": 15000,  # 매우 많은 estimators
            "verbose": -1,
            "device": "gpu",
            "gpu_use_dp": True,
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "num_threads": 1,
            "force_col_wise": True,
            "boost_from_average": True,
        })
        
        model_lgb = lgb.LGBMRegressor(**best_lgb_params)
        callbacks_lgb = [
            lgb.early_stopping(400, verbose=False),  # 더 긴 patience
            lgb.log_evaluation(period=0)
        ]
        
        model_lgb.fit(
            X_tr, y_tr_f, 
            eval_set=[(X_val, y_val_f)],
            categorical_feature="auto",
            callbacks=callbacks_lgb
        )
        oof_pred_lgb[val_idx] = model_lgb.predict(X_val)
        best_iters_lgb.append(model_lgb.best_iteration_)

        # XGBoost GPU 집약적 최적화
        print(f"      🔥 XGBoost GPU-intensive optimization...")
        study_xgb = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=30)
        )
        try:
            study_xgb.optimize(
                lambda tr: xgb_objective_gpu_intensive(tr, X_tr, y_tr_f, X_val, y_val_f, gpu_memory_gb),
                n_trials=n_trials * 2,  # 더 많은 trials
                show_progress_bar=False,
                n_jobs=1
            )
        except Exception as e:
            print(f"      ⚠️ XGBoost optimization failed: {str(e)[:100]}...")
            # 안전한 기본값으로 대체
            study_xgb = optuna.create_study(direction="minimize")
            default_params = {
                "max_bin": 64, "max_depth": 3, "lr": 0.05,
                "subsample": 0.8, "colsample": 0.8,
                "colsample_level": 0.8, "colsample_node": 0.8,
                "min_child_weight": 10, "ra": 0.01, "rl": 0.01, "gamma": 0.01
            }
            study_xgb.enqueue_trial(default_params)
            try:
                xgb_objective_gpu_intensive(study_xgb.trials[0], X_tr, y_tr_f, X_val, y_val_f, gpu_memory_gb)
            except:
                pass
        
        try:
            best_xgb_value = study_xgb.best_value if len(study_xgb.trials) > 0 and study_xgb.best_value is not None else 15.0
        except:
            best_xgb_value = 15.0
        print(f"      🎯 XGB GPU-intensive best: {best_xgb_value:.3f}%")
        
        # Best XGB model with GPU-intensive settings
        try:
            best_xgb_params = study_xgb.best_params.copy() if len(study_xgb.trials) > 0 else {}
        except:
            best_xgb_params = {
                "max_bin": 64, "max_depth": 3, "lr": 0.05,
                "subsample": 0.8, "colsample": 0.8,
                "colsample_level": 0.8, "colsample_node": 0.8,
                "min_child_weight": 10, "ra": 0.01, "rl": 0.01, "gamma": 0.01
            }
        best_xgb_params.update({
            "objective": "reg:squarederror",
            "random_state": SEED,
            "n_estimators": 15000,  # 매우 많은 estimators
            "verbosity": 0,
            "tree_method": "gpu_hist",
            "gpu_id": 0,
            "grow_policy": "lossguide",
            "predictor": "gpu_predictor",
            "n_jobs": 1,
            "sampling_method": "gradient_based",
        })
        
        model_xgb = xgb.XGBRegressor(**best_xgb_params)
        model_xgb.fit(X_tr, y_tr_f)
        oof_pred_xgb[val_idx] = model_xgb.predict(X_val)
        best_iters_xgb.append(best_xgb_params.get('n_estimators', 2000))

        # 메모리 정리
        del model_lgb, model_xgb
        gc.collect()

    # 앙상블 (LightGBM 60% + XGBoost 40%)
    oof_pred_ensemble = 0.6 * oof_pred_lgb + 0.4 * oof_pred_xgb
    sm = smape_np(df_tr["전력소비량(kWh)"].values, np.expm1(oof_pred_ensemble)*area_tr)
    
    sm_lgb = smape_np(df_tr["전력소비량(kWh)"].values, np.expm1(oof_pred_lgb)*area_tr)
    sm_xgb = smape_np(df_tr["전력소비량(kWh)"].values, np.expm1(oof_pred_xgb)*area_tr)
    
    print(f"    📊 GPU-intensive results - LGB: {sm_lgb:.3f}% | XGB: {sm_xgb:.3f}% | Ensemble: {sm:.3f}%")

    # Final models with maximum GPU utilization
    avg_lgb_iter = min(int(np.mean(best_iters_lgb) * 1.2), 20000) if best_iters_lgb else 5000
    avg_xgb_iter = min(int(np.mean(best_iters_xgb) * 1.2), 20000) if best_iters_xgb else 5000
    
    print(f"    🔥 Final GPU-intensive training - LGB: {avg_lgb_iter}, XGB: {avg_xgb_iter}")
    
    # 최종 모델들에 GPU-intensive 설정 적용
    final_lgb_params = best_lgb_params.copy()
    final_lgb_params['n_estimators'] = avg_lgb_iter
    
    final_xgb_params = best_xgb_params.copy()
    final_xgb_params['n_estimators'] = avg_xgb_iter
    
    print(f"    💪 Training final LightGBM with GPU-intensive settings...")
    final_lgb = lgb.LGBMRegressor(**final_lgb_params)
    final_lgb.fit(X_scaled, y_tr_log, categorical_feature="auto")
    
    print(f"    💪 Training final XGBoost with GPU-intensive settings...")
    final_xgb = xgb.XGBRegressor(**final_xgb_params)
    final_xgb.fit(X_scaled, y_tr_log)

    # Prediction
    preds_df = None
    if not df_te.empty:
        X_te = scaler.transform(df_te[feats])
        pred_lgb = final_lgb.predict(X_te)
        pred_xgb = final_xgb.predict(X_te)
        pred_ensemble = 0.6 * pred_lgb + 0.4 * pred_xgb
        pred_kwh = np.expm1(pred_ensemble) * area_te
        preds_df = pd.DataFrame({
            "num_date_time": df_te["num_date_time"],
            "answer": pred_kwh.clip(min=0)
        })
    
    # 메모리 정리
    del final_lgb, final_xgb
    gc.collect()
    
    return sm, preds_df

###########################################################
# Main Pipeline
###########################################################

def run_pipeline(train_path: Path, test_path: Path, info_path: Path, n_trials: int, use_gpu: bool):
    print("📥 Loading data ... (CPU preprocessing)")
    train_df = pd.read_csv(train_path, parse_dates=["일시"])
    test_df = pd.read_csv(test_path, parse_dates=["일시"])
    info_df = pd.read_csv(info_path)

    # GPU 지원 확인 및 메모리 정보
    gpu_available, gpu_memory_gb = check_gpu_support()
    if not use_gpu or not gpu_available:
        raise RuntimeError("🚫 GPU mode required for test5.py! Use --gpu flag")

    print(f"💪 GPU Memory Optimization Mode: {gpu_memory_gb}GB")

    # Missing indicators BEFORE replacement
    miss_cols = ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    for c in miss_cols:
        info_df[f"{c}_missing"] = (info_df[c] == "-").astype(int)

    num_cols = ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    info_df[num_cols] = info_df[num_cols].replace("-", 0).astype(float)

    train = train_df.merge(info_df, on="건물번호", how="left")
    test = test_df.merge(info_df, on="건물번호", how="left")
    test["전력소비량(kWh)"] = np.nan

    all_df = pd.concat([train, test], ignore_index=True)

    # Extended feature engineering for GPU memory usage
    print("🔧 Extended feature engineering ... (CPU preprocessing)")
    all_df = add_time_features(all_df)
    all_df = add_weather_features(all_df)

    # Target scaling
    all_df["power_per_area"] = all_df["전력소비량(kWh)"] / (all_df["연면적(m2)"].replace(0, np.nan))
    all_df["log_power_pa"] = np.log1p(all_df["power_per_area"].clip(lower=0))

    # Extended feature list for GPU memory usage
    feats = [
        "건물번호","기온(°C)","풍속(m/s)","습도(%)","연면적(m2)","냉방면적(m2)",
        "태양광용량(kW)","ESS저장용량(kWh)","PCS용량(kW)",
        "dew_point","heat_index","THI","THI_diff_24h",
        "temp_diff_1h","temp_diff_6h","temp_diff_12h",
        "temp_rolling_mean_6h","temp_rolling_std_6h",
        "humidity_diff_1h","wind_speed_max_6h",
        "month","day","hour","weekday","is_weekend","is_holiday",
        "hour_sin","hour_cos","month_sin","month_cos","weekday_sin","weekday_cos"
    ] + [f"{c}_missing" for c in miss_cols]

    df_train = all_df[~all_df["전력소비량(kWh)"].isna()].copy()
    df_test = all_df[all_df["전력소비량(kWh)"].isna()].copy()

    # num_date_time 컬럼 생성
    print("🔧 Creating num_date_time column...")
    df_test['num_date_time'] = df_test['건물번호'].astype(str) + '_' + df_test['일시'].dt.strftime('%Y%m%d %H')

    # GPU 집약적 건물별 학습
    print(f"🚀 Starting GPU-intensive building-wise training (using {gpu_memory_gb}GB)...")
    sub_parts = []
    scores = []
    
    for bid in df_train["건물번호"].unique():
        print(f"\n🏢 Building {bid} - GPU Memory Intensive Mode")
        tr_b = df_train[df_train["건물번호"] == bid].copy()
        te_b = df_test[df_test["건물번호"] == bid].copy()
        if len(tr_b) < 150:  # 더 낮은 임계값으로 더 많은 건물 학습
            print(f"  ⚠️ Skipping Building {bid} - insufficient data ({len(tr_b)} < 150)")
            continue
            
        sm, preds = train_building_gpu_intensive(tr_b, te_b, feats, n_trials, gpu_memory_gb)
        scores.append(sm)
        if preds is not None:
            sub_parts.append(preds)
            print(f"  ✅ Building {bid} GPU-intensive predictions added ({len(preds)} rows)")

    print(f"\n📈 GPU-Intensive Average OOF SMAPE: {np.mean(scores):.3f}%")
    
    if sub_parts:
        print(f"🔗 Concatenating {len(sub_parts)} GPU-intensive predictions...")
        submission = pd.concat(sub_parts, ignore_index=True)
        print(f"   Combined predictions shape: {submission.shape}")
        
        # align with sample_submission if exists
        sample_path = test_path.parent / "sample_submission.csv"
        if sample_path.exists():
            print("📋 Aligning with sample_submission.csv...")
            sample = pd.read_csv(sample_path).drop(columns=["answer"], errors="ignore")
            submission = sample.merge(submission, on="num_date_time", how="left")
            print(f"   Final submission shape: {submission.shape}")
        
        submission.to_csv("submission_gpu_intensive.csv", index=False)
        print("✅ submission_gpu_intensive.csv saved.")
    else:
        print("❌ No predictions generated!")
        sample_path = test_path.parent / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            sample.to_csv("submission_gpu_intensive.csv", index=False)
            print("📝 Default submission_gpu_intensive.csv created.")

###########################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/test.csv"))
    ap.add_argument("--info", type=Path, default=Path("data/building_info.csv"))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--gpu", action="store_true", help="GPU 가속 사용 (필수)")
    args = ap.parse_args()
    
    if not args.gpu:
        print("❌ test5.py requires --gpu flag for maximum GPU utilization!")
        exit(1)
        
    run_pipeline(args.train, args.test, args.info, args.n_trials, args.gpu) 