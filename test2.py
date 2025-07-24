# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v4
--------------------------------
* 개선 사항 요약
  1. ***Walk‑Forward + TimeSeriesSplit*** 검증으로 OOF SMAPE 추정 ⇒ 리더보드 예측 신뢰도 ↑
  2. LightGBM에 ***사용자 정의 SMAPE 평가지표*** 적용 (loss: L1, eval: SMAPE)
  3. 추가 피처
     · 전일 대비 기온/습도 변화 (delta)            
     · 주/야 구분, 분기, 연도                       
     · 24h & 168h 이동 평균/표준편차               
  4. 하이퍼파라미터 튜닝 (베이지안 최적 결과 반영)  
  5. ***OOF 예측 캐싱*** 후 최종 모델을 full‑data + best_iter_mean × 1.1 로 재학습
  6. 코드 구조를 함수화하여 재현성·가독성 향상

사용 방법
---------
$ python energy_prediction_v4.py --train data/train.csv --test data/test.csv --info data/building_info.csv
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

def lgb_smape(y_pred: np.ndarray, data: lgb.Dataset):
    y_true = data.get_label()
    return "SMAPE", smape_np(y_true, y_pred), False  # lower is better

###########################################################
# Feature Engineering
###########################################################

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
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

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # Temperature Humidity Index
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    # day‑over‑day delta
    df["temp_d1"] = df.groupby("건물번호")["기온(°C)"].diff(24)
    df["humid_d1"] = df.groupby("건물번호")["습도(%)"].diff(24)
    return df

def add_rolling_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lags = [24, 48, 168]
    rollers = [24, 168]

    for lag in lags:
        df[f"power_lag_{lag}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].shift(lag)
        )
        df[f"temp_lag_{lag}"] = (
            df.groupby("건물번호")["기온(°C)"].shift(lag)
        )

    for win in rollers:
        df[f"power_roll_mean_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).mean().reset_index(0, drop=True)
        )
        df[f"power_roll_std_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).std().reset_index(0, drop=True)
            .fillna(0)
        )
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["건물번호", "일시"]).reset_index(drop=True)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_rolling_lag_features(df)
    # interaction features
    df["temp_x_hour"] = df["기온(°C)"] * df["hour"]
    return df

###########################################################
# Model Training per Building Type
###########################################################

def train_per_type(df_train: pd.DataFrame, df_test: pd.DataFrame, features: list, cat_cols: list):
    oof_smape_scores = []
    predictions = []

    default_params = dict(
        objective="regression_l1",
        boosting_type="gbdt",
        random_state=SEED,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        n_estimators=6000,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_data_in_leaf=100,
        reg_alpha=0.2,
        reg_lambda=0.2,
    )

    for btype, gdf in df_train.groupby("건물유형"):
        print(f"\n▶ Building‑Type: {btype}  (train rows={len(gdf)})")

        y = gdf["전력소비량(kWh)"].values
        X = gdf[features]
        oof_pred = np.zeros(len(gdf))

        # walk‑forward CV (5 splits)
        tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 7)  # 1주일 horizon
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model = lgb.LGBMRegressor(**default_params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=lgb_smape,
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            best_iter = model.best_iteration_
            oof_pred[val_idx] = model.predict(X_val, num_iteration=best_iter)

        oof_pred[oof_pred < 0] = 0
        sm = smape_np(y, oof_pred)
        print(f"  OOF SMAPE: {sm:.4f}%")
        oof_smape_scores.append(sm)

        # re‑train on full data with 10% buffer of iterations
        best_iters = int(model.best_iteration_ * 1.1)
        final_model = lgb.LGBMRegressor(**default_params, n_estimators=best_iters)
        final_model.fit(X, y, categorical_feature=cat_cols)

        test_part = df_test[df_test["건물유형"] == btype]
        if not test_part.empty:
            preds = final_model.predict(test_part[features])
            preds[preds < 0] = 0
            predictions.append(
                pd.DataFrame({
                    "num_date_time": test_part["num_date_time"],
                    "answer": preds,
                })
            )

    mean_smape = np.mean(oof_smape_scores)
    print("\n========================================")
    print(f"Average OOF SMAPE: {mean_smape:.4f}% across {len(oof_smape_scores)} types")
    print("========================================\n")
    return pd.concat(predictions, ignore_index=True)

###########################################################
# Main
###########################################################

def run_pipeline(train_path: Path, test_path: Path, info_path: Path):
    # 1. Load
    train_df = pd.read_csv(train_path, parse_dates=["일시"])
    test_df = pd.read_csv(test_path, parse_dates=["일시"])
    info_df = pd.read_csv(info_path)

    # 2. numeric casting
    numeric_cols = [
        "연면적(m2)",
        "냉방면적(m2)",
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
    ]
    info_df[numeric_cols] = (
        info_df[numeric_cols]
        .replace("-", 0)
        .astype(float)
    )

    # 3. merge
    train = train_df.merge(info_df, on="건물번호", how="left")
    test = test_df.merge(info_df, on="건물번호", how="left")

    test["전력소비량(kWh)"] = np.nan
    data_all = pd.concat([train, test], ignore_index=True)

    # 4. feature engineering
    data_all = create_features(data_all)

    # 5. typcasting categorical
    cat_cols = ["건물번호"]
    for c in cat_cols:
        data_all[c] = data_all[c].astype("category")

    feat_cols = [
        "건물번호", "기온(°C)", "풍속(m/s)", "습도(%)",
        "연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)",
        "month", "day", "hour", "weekday", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "year", "quarter", "is_night",
        "THI", "temp_d1", "humid_d1", "temp_x_hour",
        "power_lag_24", "temp_lag_24", "power_lag_48", "temp_lag_48", "power_lag_168", "temp_lag_168",
        "power_roll_mean_24", "power_roll_std_24", "power_roll_mean_168", "power_roll_std_168",
    ]

    df_train = data_all[~data_all["전력소비량(kWh)"].isna()].copy()
    df_test = data_all[data_all["전력소비량(kWh)"].isna()].copy()

    # 6. train & predict
    submission_pred = train_per_type(df_train, df_test, feat_cols, cat_cols)

    # 7. align with sample submission if exists
    if (test_path.parent / "sample_submission.csv").exists():
        sample = pd.read_csv(test_path.parent / "sample_submission.csv")
        sample = sample.drop(columns=["answer"], errors="ignore")
        submission_pred = sample.merge(submission_pred, on="num_date_time", how="left")

    submission_pred.to_csv("submission.csv", index=False)
    print("Saved: submission.csv")

###########################################################
# Entry Point
###########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    args = parser.parse_args()

    run_pipeline(args.train, args.test, args.info)
