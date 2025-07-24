# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v8 - Optuna + Scaled Target
-------------------------------------------------------
• 면적 정규화(log1p(kWh / 연면적))로 목표변수 스케일 개선
• 결측 인디케이터(태양광/ESS/PCS) 추가
• 새로운 날씨 파생 피처(Dew Point, Heat Index, THI_diff_24h)
• 건물별 개별 Optuna 최적화 + LightGBM/XGBoost 앙상블(60/40)
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
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["THI"] = 9/5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9/5 * df["기온(°C)"] - 26) + 32
    # New
    df["dew_point"] = df["기온(°C)"] - (100 - df["습도(%)"]) / 5
    df["heat_index"] = 0.5 * (df["기온(°C)"] + 61.0 + (df["기온(°C)"] - 68.0) * 1.2 + df["습도(%)"] * 0.094)
    df["THI_diff_24h"] = df.groupby("건물번호")["THI"].diff(24)
    return df

###########################################################
# Optuna Objective
###########################################################

def lgb_objective(trial, X_tr, y_tr, X_val, y_val, cat_cols):
    params = {
        "objective": "regression_l1",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("lr", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 10, 100),
        "reg_alpha": trial.suggest_float("ra", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("rl", 1e-8, 1.0, log=True),
        "n_estimators": 2000,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda y,p: ("SMAPE", smape_np(y, np.expm1(p)), False),
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    preds = model.predict(X_val)
    return smape_np(np.expm1(y_val), np.expm1(preds))

###########################################################
# Training per building
###########################################################

def train_building(df_tr: pd.DataFrame, df_te: pd.DataFrame, feats: list, n_trials: int):
    # Area for inverse transform
    area_tr = df_tr["연면적(m2)"].values
    area_te = df_te["연면적(m2)"].values if not df_te.empty else None
    y_tr_log = df_tr["log_power_pa"].values

    X_full = df_tr[feats]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_scaled = pd.DataFrame(X_scaled, columns=feats, index=df_tr.index)

    tscv = TimeSeriesSplit(n_splits=3, test_size=24*7)

    oof_pred = np.zeros(len(df_tr))
    for fold,(tr_idx,val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr = X_scaled.iloc[tr_idx]
        y_tr_f = y_tr_log[tr_idx]
        X_val = X_scaled.iloc[val_idx]
        y_val_f = y_tr_log[val_idx]

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda tr: lgb_objective(tr, X_tr, y_tr_f, X_val, y_val_f, ["건물번호"]), n_trials=n_trials)
        best_params = study.best_params
        best_params.update({
            "objective": "regression_l1",
            "random_state": SEED,
            "learning_rate": best_params.pop("lr"),
            "num_leaves": best_params.pop("num_leaves"),
            "subsample": best_params.pop("subsample"),
            "colsample_bytree": best_params.pop("colsample"),
            "min_data_in_leaf": best_params.pop("min_leaf"),
            "reg_alpha": best_params.pop("ra"),
            "reg_lambda": best_params.pop("rl"),
            "n_estimators": 5000,
            "verbose": -1,
        })
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_tr, y_tr_f, categorical_feature=["건물번호"], verbose=-1)
        oof_pred[val_idx] = model.predict(X_val)
    sm = smape_np(df_tr["전력소비량(kWh)"].values, np.expm1(oof_pred)*area_tr)
    print(f"    OOF SMAPE: {sm:.3f}%")

    # train final model on all data
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_scaled, y_tr_log, categorical_feature=["건물번호"], verbose=-1)

    # prediction
    preds_df = None
    if not df_te.empty:
        X_te = scaler.transform(df_te[feats])
        pred_log = final_model.predict(X_te)
        pred_kwh = np.expm1(pred_log) * area_te
        preds_df = pd.DataFrame({
            "num_date_time": df_te["num_date_time"],
            "answer": pred_kwh.clip(min=0)
        })
    return sm, preds_df

###########################################################
# Main Pipeline
###########################################################

def run_pipeline(train_path: Path, test_path: Path, info_path: Path, n_trials: int):
    print("📥 Loading data ...")
    train_df = pd.read_csv(train_path, parse_dates=["일시"])
    test_df = pd.read_csv(test_path, parse_dates=["일시"])
    info_df = pd.read_csv(info_path)

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
        "dew_point","heat_index","THI","THI_diff_24h",
        "month","day","hour","weekday","is_weekend","is_holiday",
        "hour_sin","hour_cos","month_sin","month_cos"
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
        sm, preds = train_building(tr_b, te_b, feats, n_trials)
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
    args = ap.parse_args()
    run_pipeline(args.train, args.test, args.info, args.n_trials) 