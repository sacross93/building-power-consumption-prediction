# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v4‑fixed2
-------------------------------------
* **Bug Fix #2**: duplicate `n_estimators` → 튜플 오류 해결.
  * `params` 사전에는 base n_estimators 값을 제거하고, fold별 best_iter 계산 후 **copy + override** 로 안전하게 주입.
* 추가: 경고 완화용 `min_gain_to_split=0` 명시.
"""
import argparse
import warnings
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

KR_HOLIDAYS = holidays.KR()
SEED = 42

###########################################################
# Utility
###########################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100


def lgb_smape(y_true: np.ndarray, y_pred: np.ndarray):
    return "SMAPE", smape_np(y_true, y_pred), False

###########################################################
# Feature Engineering (functions unchanged)
###########################################################

def add_time_features(df):
    df["month"] = df["일시"].dt.month
    df["day"] = df["일시"].dt.day
    df["hour"] = df["일시"].dt.hour
    df["weekday"] = df["일시"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["일시"].dt.date.map(lambda d: int(d in KR_HOLIDAYS))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["year"] = df["일시"].dt.year
    df["quarter"] = df["일시"].dt.quarter
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    return df

def add_weather_features(df):
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    df["temp_d1"] = df.groupby("건물번호")["기온(°C)"].diff(24)
    df["humid_d1"] = df.groupby("건물번호")["습도(%)"].diff(24)
    # 상호작용 피처 추가
    df["temp_x_hour"] = df["기온(°C)"] * df["hour"]
    df["temp_x_humid"] = df["기온(°C)"] * df["습도(%)"]
    return df

def add_rolling_lag_features(df):
    lags = [24, 48, 168]
    rollers = [24, 168]
    for lag in lags:
        df[f"power_lag_{lag}"] = df.groupby("건물번호")["전력소비량(kWh)"].shift(lag)
        df[f"temp_lag_{lag}"] = df.groupby("건물번호")["기온(°C)"].shift(lag)
    for win in rollers:
        df[f"power_roll_mean_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).mean().reset_index(0, drop=True))
        df[f"power_roll_std_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).std().reset_index(0, drop=True).fillna(0))
    return df

def create_features(df):
    df = df.sort_values(["건물번호", "일시"]).reset_index(drop=True)
    return add_rolling_lag_features(add_weather_features(add_time_features(df)))

###########################################################
# Training per Building‑Type
###########################################################

def train_per_type(df_train, df_test, features, cat_cols):
    oof_scores, preds_list = [], []

    base_params = dict(
        objective="regression_l1",
        boosting_type="gbdt",
        random_state=SEED,
        learning_rate=0.02,  # 학습률 조정
        num_leaves=32,       # 과적합 방지
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=50,  # 더 작은 값으로 조정
        reg_alpha=0.1,        # 정규화 조정
        reg_lambda=0.1,
        min_gain_to_split=0.0,
        verbose=-1,           # 출력 최소화
    )

    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 7)

    for btype, gdf in df_train.groupby("건물유형"):
        print(f"\n▶ Building‑Type: {btype} (rows={len(gdf)})")
        y = gdf["전력소비량(kWh)"].values
        X = gdf[features]
        oof_pred = np.zeros(len(gdf))
        best_iters = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            model = lgb.LGBMRegressor(**base_params, n_estimators=3000)  # 반복 횟수 조정
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[val_idx], y[val_idx])],
                eval_metric=lgb_smape,
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(100, verbose=False)],  # early stopping 조정
            )
            best_iters.append(model.best_iteration_)
            oof_pred[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration_)

        oof_pred[oof_pred < 0] = 0
        sm = smape_np(y, oof_pred)
        print(f"  OOF SMAPE: {sm:.2f}%")
        oof_scores.append(sm)

        # full‑data retrain
        final_params = base_params.copy()
        final_params["n_estimators"] = int(np.mean(best_iters) * 1.1)
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(X, y, categorical_feature=cat_cols, verbose=-1)

        part_test = df_test[df_test["건물유형"] == btype]
        if not part_test.empty:
            pred = final_model.predict(part_test[features])
            pred[pred < 0] = 0
            preds_list.append(pd.DataFrame({"num_date_time": part_test["num_date_time"], "answer": pred}))

    print("\nAverage OOF SMAPE: {:.2f}%".format(np.mean(oof_scores)))
    return pd.concat(preds_list, ignore_index=True)

###########################################################
# Main
###########################################################

def run_pipeline(train_path, test_path, info_path):
    print("데이터 로딩 중...")
    train_df = pd.read_csv(train_path, parse_dates=["일시"])
    test_df = pd.read_csv(test_path, parse_dates=["일시"])
    info_df = pd.read_csv(info_path)
    
    print("데이터 전처리 중...")
    numeric = ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    info_df[numeric] = info_df[numeric].replace("-", 0).astype(float)

    train = train_df.merge(info_df, on="건물번호", how="left")
    test = test_df.merge(info_df, on="건물번호", how="left")
    test["전력소비량(kWh)"] = np.nan

    print("피처 엔지니어링 중...")
    all_df = pd.concat([train, test], ignore_index=True)
    all_df = create_features(all_df)
    all_df["건물번호"] = all_df["건물번호"].astype("category")

    feats = [
        "건물번호", "기온(°C)", "풍속(m/s)", "습도(%)", "연면적(m2)", "냉방면적(m2)", "태양광용량(kW)",
        "ESS저장용량(kWh)", "PCS용량(kW)", "month", "day", "hour", "weekday", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "year", "quarter", "is_night", "THI", "temp_d1", "humid_d1", 
        "temp_x_hour", "temp_x_humid",
        "power_lag_24", "temp_lag_24", "power_lag_48", "temp_lag_48", "power_lag_168", "temp_lag_168",
        "power_roll_mean_24", "power_roll_std_24", "power_roll_mean_168", "power_roll_std_168",
    ]

    df_train = all_df[~all_df["전력소비량(kWh)"].isna()].copy()
    df_test = all_df[all_df["전력소비량(kWh)"].isna()].copy()

    print("모델 학습 및 예측 중...")
    sub_df = train_per_type(df_train, df_test, feats, ["건물번호"])
    
    print("결과 저장 중...")
    sample_path = test_path.parent / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path).drop(columns=["answer"], errors="ignore")
        sub_df = sample.merge(sub_df, on="num_date_time", how="left")
    sub_df.to_csv("submission.csv", index=False)
    print("✅ submission.csv 저장 완료!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/test.csv"))
    ap.add_argument("--info", type=Path, default=Path("data/building_info.csv"))
    args = ap.parse_args()
    run_pipeline(args.train, args.test, args.info)
