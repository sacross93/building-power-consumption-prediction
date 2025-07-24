# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v9 - Optuna + GPU Support
-----------------------------------------------------
• 면적 정규화(log1p(kWh / 연면적))로 목표변수 스케일 개선
• 결측 인디케이터(태양광/ESS/PCS) 추가
• 새로운 날씨 파생 피처(Dew Point, Heat Index, THI_diff_24h)
• 건물별 개별 Optuna 최적화 + LightGBM/XGBoost 앙상블
• GPU 가속 지원 (--gpu 플래그)
• submission.csv 매 실행마다 덮어쓰기
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
warnings.filterwarnings("ignore")

KR_HOLIDAYS = holidays.KR()
SEED = 42

###########################################################
# GPU Detection & Setup
###########################################################

def check_gpu_support():
    """GPU 지원 가능 여부 확인"""
    try:
        # LightGBM GPU 확인
        lgb_test = lgb.LGBMRegressor(device="gpu", n_estimators=1)
        lgb_gpu = True
        print("✅ LightGBM GPU support detected")
    except Exception:
        lgb_gpu = False
        print("❌ LightGBM GPU not available, using CPU")
    
    try:
        # XGBoost GPU 확인
        xgb_test = xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=1)
        xgb_gpu = True
        print("✅ XGBoost GPU support detected")
    except Exception:
        xgb_gpu = False
        print("❌ XGBoost GPU not available, using CPU")
    
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
    # New
    df["dew_point"] = df["기온(°C)"] - (100 - df["습도(%)"]) / 5
    df["heat_index"] = 0.5 * (df["기온(°C)"] + 61.0 + (df["기온(°C)"] - 68.0) * 1.2 + df["습도(%)"] * 0.094)
    df["THI_diff_24h"] = df.groupby("건물번호")["THI"].diff(24)
    # 온도 변화량
    df["temp_diff_1h"] = df.groupby("건물번호")["기온(°C)"].diff(1)
    df["temp_diff_6h"] = df.groupby("건물번호")["기온(°C)"].diff(6)
    return df

###########################################################
# Optuna Objectives
###########################################################

def lgb_objective(trial, X_tr, y_tr, X_val, y_val, cat_cols, use_gpu=False):
    params = {
        "objective": "regression_l1",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("lr", 0.003, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 5, 100),
        "reg_alpha": trial.suggest_float("ra", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 1e-8, 10.0, log=True),
        "n_estimators": 2000,
        "verbose": -1,
    }
    
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = True
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda y,p: ("SMAPE", smape_np(np.expm1(y), np.expm1(p)), False),
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    preds = model.predict(X_val)
    return smape_np(np.expm1(y_val), np.expm1(preds))

def xgb_objective(trial, X_tr, y_tr, X_val, y_val, use_gpu=False):
    params = {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("lr", 0.003, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("ra", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 1e-8, 10.0, log=True),
        "n_estimators": 2000,
        "verbosity": 0,
    }
    
    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["gpu_id"] = 0
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return smape_np(np.expm1(y_val), np.expm1(preds))

###########################################################
# Training per building
###########################################################

def train_building(df_tr: pd.DataFrame, df_te: pd.DataFrame, feats: list, n_trials: int, use_gpu: bool, lgb_gpu: bool, xgb_gpu: bool):
    # Area for inverse transform
    area_tr = df_tr["연면적(m2)"].values
    area_te = df_te["연면적(m2)"].values if not df_te.empty else None
    y_tr_log = df_tr["log_power_pa"].values

    X_full = df_tr[feats]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_scaled = pd.DataFrame(X_scaled, columns=feats, index=df_tr.index)

    tscv = TimeSeriesSplit(n_splits=3, test_size=24*7)

    oof_pred_lgb = np.zeros(len(df_tr))
    oof_pred_xgb = np.zeros(len(df_tr))
    
    for fold,(tr_idx,val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"    Fold {fold+1}/3")
        X_tr = X_scaled.iloc[tr_idx]
        y_tr_f = y_tr_log[tr_idx]
        X_val = X_scaled.iloc[val_idx]
        y_val_f = y_tr_log[val_idx]

        # LightGBM 최적화
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(
            lambda tr: lgb_objective(tr, X_tr, y_tr_f, X_val, y_val_f, ["건물번호"], use_gpu and lgb_gpu), 
            n_trials=n_trials//2
        )
        best_lgb_params = study_lgb.best_params
        best_lgb_params.update({
            "objective": "regression_l1",
            "random_state": SEED,
            "learning_rate": best_lgb_params.pop("lr"),
            "num_leaves": best_lgb_params.pop("num_leaves"),
            "subsample": best_lgb_params.pop("subsample"),
            "colsample_bytree": best_lgb_params.pop("colsample"),
            "min_data_in_leaf": best_lgb_params.pop("min_leaf"),
            "reg_alpha": best_lgb_params.pop("ra"),
            "reg_lambda": best_lgb_params.pop("rl"),
            "n_estimators": 3000,
            "verbose": -1,
        })
        
        if use_gpu and lgb_gpu:
            best_lgb_params["device"] = "gpu"
            best_lgb_params["gpu_use_dp"] = True
        
        model_lgb = lgb.LGBMRegressor(**best_lgb_params)
        model_lgb.fit(X_tr, y_tr_f, categorical_feature=["건물번호"], verbose=-1)
        oof_pred_lgb[val_idx] = model_lgb.predict(X_val)

        # XGBoost 최적화
        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(
            lambda tr: xgb_objective(tr, X_tr, y_tr_f, X_val, y_val_f, use_gpu and xgb_gpu),
            n_trials=n_trials//2
        )
        best_xgb_params = study_xgb.best_params
        best_xgb_params.update({
            "objective": "reg:squarederror",
            "random_state": SEED,
            "learning_rate": best_xgb_params.pop("lr"),
            "max_depth": best_xgb_params.pop("max_depth"),
            "subsample": best_xgb_params.pop("subsample"),
            "colsample_bytree": best_xgb_params.pop("colsample"),
            "min_child_weight": best_xgb_params.pop("min_child_weight"),
            "reg_alpha": best_xgb_params.pop("ra"),
            "reg_lambda": best_xgb_params.pop("rl"),
            "n_estimators": 3000,
            "verbosity": 0,
        })
        
        if use_gpu and xgb_gpu:
            best_xgb_params["tree_method"] = "gpu_hist"
            best_xgb_params["gpu_id"] = 0
        
        model_xgb = xgb.XGBRegressor(**best_xgb_params)
        model_xgb.fit(X_tr, y_tr_f, verbose=False)
        oof_pred_xgb[val_idx] = model_xgb.predict(X_val)

    # 앙상블 (LightGBM 60% + XGBoost 40%)
    oof_pred_ensemble = 0.6 * oof_pred_lgb + 0.4 * oof_pred_xgb
    sm = smape_np(df_tr["전력소비량(kWh)"].values, np.expm1(oof_pred_ensemble)*area_tr)
    print(f"    OOF SMAPE: {sm:.3f}%")

    # train final models on all data
    final_lgb = lgb.LGBMRegressor(**best_lgb_params)
    final_lgb.fit(X_scaled, y_tr_log, categorical_feature=["건물번호"], verbose=-1)
    
    final_xgb = xgb.XGBRegressor(**best_xgb_params)
    final_xgb.fit(X_scaled, y_tr_log, verbose=False)

    # prediction
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
    return sm, preds_df

###########################################################
# Main Pipeline
###########################################################

def run_pipeline(train_path: Path, test_path: Path, info_path: Path, n_trials: int, use_gpu: bool):
    print("📥 Loading data ...")
    train_df = pd.read_csv(train_path, parse_dates=["일시"])
    test_df = pd.read_csv(test_path, parse_dates=["일시"])
    info_df = pd.read_csv(info_path)

    # GPU 지원 확인
    lgb_gpu, xgb_gpu = False, False
    if use_gpu:
        print("\n🚀 Checking GPU support...")
        lgb_gpu, xgb_gpu = check_gpu_support()

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

    # Feature engineering
    all_df = add_time_features(all_df)
    all_df = add_weather_features(all_df)

    # Target scaling
    all_df["power_per_area"] = all_df["전력소비량(kWh)"] / (all_df["연면적(m2)"].replace(0, np.nan))
    all_df["log_power_pa"] = np.log1p(all_df["power_per_area"].clip(lower=0))

    # Feature list
    feats = [
        "건물번호","기온(°C)","풍속(m/s)","습도(%)","연면적(m2)","냉방면적(m2)",
        "태양광용량(kW)","ESS저장용량(kWh)","PCS용량(kW)",
        "dew_point","heat_index","THI","THI_diff_24h","temp_diff_1h","temp_diff_6h",
        "month","day","hour","weekday","is_weekend","is_holiday",
        "hour_sin","hour_cos","month_sin","month_cos","weekday_sin","weekday_cos"
    ] + [f"{c}_missing" for c in miss_cols]

    df_train = all_df[~all_df["전력소비량(kWh)"].isna()].copy()
    df_test = all_df[all_df["전력소비량(kWh)"].isna()].copy()

    # train per building
    sub_parts = []
    scores = []
    for bid in df_train["건물번호"].unique():
        print(f"\n🏢 Building {bid}")
        tr_b = df_train[df_train["건물번호"] == bid].copy()
        te_b = df_test[df_test["건물번호"] == bid].copy()
        if len(tr_b) < 200:
            continue
        sm, preds = train_building(tr_b, te_b, feats, n_trials, use_gpu, lgb_gpu, xgb_gpu)
        scores.append(sm)
        if preds is not None:
            sub_parts.append(preds)

    print(f"\n📈 Average OOF SMAPE: {np.mean(scores):.3f}%")
    submission = pd.concat(sub_parts, ignore_index=True)
    # align with sample_submission if exists
    sample_path = test_path.parent / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path).drop(columns=["answer"], errors="ignore")
        submission = sample.merge(submission, on="num_date_time", how="left")
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved.")

###########################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/test.csv"))
    ap.add_argument("--info", type=Path, default=Path("data/building_info.csv"))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--gpu", action="store_true", help="GPU 가속 사용")
    args = ap.parse_args()
    run_pipeline(args.train, args.test, args.info, args.n_trials, args.gpu) 