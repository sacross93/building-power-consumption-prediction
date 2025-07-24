# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v6 - SMAPE 6% 이하 목표
----------------------------------------------------
* 핵심 개선사항:
  1. 고급 시계열 피처 (푸리에 변환, 계절성 분해)
  2. 건물별 개별 모델링
  3. 다중 앙상블 (LightGBM + XGBoost + CatBoost)
  4. 베이지안 하이퍼파라미터 최적화
  5. 시계열 특화 검증
  6. 후처리 및 앙상블
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from scipy import stats
from scipy.signal import periodogram
import warnings
warnings.filterwarnings("ignore")

KR_HOLIDAYS = holidays.KR()
SEED = 42

###########################################################
# Utility Functions
###########################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """정확한 SMAPE 계산"""
    eps = 1e-8
    y_pred = np.maximum(y_pred, 0)  # 음수 예측값 처리
    
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(numerator / denominator) * 100

def lgb_smape(y_pred: np.ndarray, data: lgb.Dataset):
    y_true = data.get_label()
    return "SMAPE", smape_np(y_true, y_pred), False

def xgb_smape(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    return "SMAPE", smape_np(y_true, y_pred)

###########################################################
# Advanced Feature Engineering
###########################################################

def add_advanced_time_features(df):
    """고급 시간 피처"""
    df["month"] = df["일시"].dt.month
    df["day"] = df["일시"].dt.day
    df["hour"] = df["일시"].dt.hour
    df["weekday"] = df["일시"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["일시"].dt.date.map(lambda d: int(d in KR_HOLIDAYS))
    
    # 푸리에 변환 피처 (주기성)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    
    # 추가 시간 피처
    df["year"] = df["일시"].dt.year
    df["quarter"] = df["일시"].dt.quarter
    df["day_of_year"] = df["일시"].dt.dayofyear
    df["week_of_year"] = df["일시"].dt.isocalendar().week
    
    # 시간대 구분
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    
    # 계절성
    df["season"] = df["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                                   6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    
    # 주기성 피처
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["week_of_year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    
    return df

def add_advanced_weather_features(df):
    """고급 날씨 피처"""
    # 기본 날씨 피처
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    
    # 변화율 피처
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f"temp_d{lag}"] = df.groupby("건물번호")["기온(°C)"].diff(lag)
        df[f"humid_d{lag}"] = df.groupby("건물번호")["습도(%)"].diff(lag)
        df[f"wind_d{lag}"] = df.groupby("건물번호")["풍속(m/s)"].diff(lag)
    
    # 상호작용 피처
    df["temp_x_hour"] = df["기온(°C)"] * df["hour"]
    df["temp_x_humid"] = df["기온(°C)"] * df["습도(%)"]
    df["temp_x_wind"] = df["기온(°C)"] * df["풍속(m/s)"]
    df["humid_x_wind"] = df["습도(%)"] * df["풍속(m/s)"]
    df["temp_x_day"] = df["기온(°C)"] * df["day"]
    df["temp_x_month"] = df["기온(°C)"] * df["month"]
    
    # 날씨 구간 피처
    df["temp_cold"] = (df["기온(°C)"] < 10).astype(int)
    df["temp_mild"] = ((df["기온(°C)"] >= 10) & (df["기온(°C)"] < 25)).astype(int)
    df["temp_hot"] = (df["기온(°C)"] >= 25).astype(int)
    df["temp_extreme_cold"] = (df["기온(°C)"] < 0).astype(int)
    df["temp_extreme_hot"] = (df["기온(°C)"] >= 30).astype(int)
    
    # 습도 구간
    df["humid_low"] = (df["습도(%)"] < 40).astype(int)
    df["humid_medium"] = ((df["습도(%)"] >= 40) & (df["습도(%)"] < 70)).astype(int)
    df["humid_high"] = (df["습도(%)"] >= 70).astype(int)
    
    return df

def add_advanced_lag_features(df):
    """고급 시차 및 이동통계 피처"""
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504]  # 더 많은 시차
    rollers = [6, 12, 24, 48, 72, 168, 336]  # 더 많은 이동통계
    
    # 시차 피처
    for lag in lags:
        df[f"power_lag_{lag}"] = df.groupby("건물번호")["전력소비량(kWh)"].shift(lag)
        df[f"temp_lag_{lag}"] = df.groupby("건물번호")["기온(°C)"].shift(lag)
        df[f"humid_lag_{lag}"] = df.groupby("건물번호")["습도(%)"].shift(lag)
        df[f"wind_lag_{lag}"] = df.groupby("건물번호")["풍속(m/s)"].shift(lag)
    
    # 이동통계 피처
    for win in rollers:
        # 전력 소비량 이동통계
        df[f"power_roll_mean_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).mean().reset_index(0, drop=True)
        )
        df[f"power_roll_std_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        )
        df[f"power_roll_min_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).min().reset_index(0, drop=True)
        )
        df[f"power_roll_max_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).max().reset_index(0, drop=True)
        )
        df[f"power_roll_median_{win}"] = (
            df.groupby("건물번호")["전력소비량(kWh)"].rolling(win, min_periods=1).median().reset_index(0, drop=True)
        )
        
        # 기온 이동통계
        df[f"temp_roll_mean_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).mean().reset_index(0, drop=True)
        )
        df[f"temp_roll_std_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        )
        df[f"temp_roll_min_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).min().reset_index(0, drop=True)
        )
        df[f"temp_roll_max_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).max().reset_index(0, drop=True)
        )
    
    return df

def add_building_specific_features(df):
    """건물별 특성 피처"""
    # 면적 관련 피처
    df["area_per_floor"] = df["연면적(m2)"] / 10
    df["cooling_ratio"] = df["냉방면적(m2)"] / df["연면적(m2)"]
    df["area_log"] = np.log1p(df["연면적(m2)"])
    df["cooling_area_log"] = np.log1p(df["냉방면적(m2)"])
    
    # 태양광 관련 피처
    df["solar_capacity"] = df["태양광용량(kW)"].fillna(0)
    df["solar_per_area"] = df["solar_capacity"] / df["연면적(m2)"]
    df["solar_log"] = np.log1p(df["solar_capacity"])
    
    # ESS 관련 피처
    df["ess_capacity"] = df["ESS저장용량(kWh)"].fillna(0)
    df["pcs_capacity"] = df["PCS용량(kW)"].fillna(0)
    df["ess_pcs_ratio"] = df["ess_capacity"] / (df["pcs_capacity"] + 1e-8)
    df["ess_log"] = np.log1p(df["ess_capacity"])
    df["pcs_log"] = np.log1p(df["pcs_capacity"])
    
    # 건물 유형별 인코딩
    building_type_map = {
        'IDC(전화국)': 0, '백화점': 1, '병원': 2, '상용': 3, '아파트': 4,
        '연구소': 5, '학교': 6, '호텔': 7, '공공': 8, '건물기타': 9
    }
    df["building_type_encoded"] = df["건물유형"].map(building_type_map)
    
    # 건물별 특성 조합
    df["has_solar"] = (df["solar_capacity"] > 0).astype(int)
    df["has_ess"] = (df["ess_capacity"] > 0).astype(int)
    df["has_pcs"] = (df["pcs_capacity"] > 0).astype(int)
    
    return df

def create_features(df):
    """통합 피처 생성"""
    df = df.sort_values(["건물번호", "일시"]).reset_index(drop=True)
    df = add_advanced_time_features(df)
    df = add_advanced_weather_features(df)
    df = add_advanced_lag_features(df)
    df = add_building_specific_features(df)
    return df

###########################################################
# Model Training
###########################################################

def train_lightgbm(X_train, y_train, X_val, y_val, cat_cols):
    """LightGBM 모델 학습"""
    params = {
        'objective': 'regression_l1',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'learning_rate': 0.005,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_data_in_leaf': 50,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'n_estimators': 10000,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_smape,
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost 모델 학습"""
    params = {
        'objective': 'reg:absoluteerror',
        'random_state': SEED,
        'learning_rate': 0.005,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'n_estimators': 10000,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=xgb_smape,
        early_stopping_rounds=200,
        verbose=False
    )
    
    return model

def train_catboost(X_train, y_train, X_val, y_val, cat_cols):
    """CatBoost 모델 학습"""
    params = {
        'loss_function': 'MAE',
        'random_seed': SEED,
        'learning_rate': 0.005,
        'depth': 8,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'min_data_in_leaf': 50,
        'reg_lambda': 0.01,
        'iterations': 10000,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=200,
        verbose=False
    )
    
    return model

def train_per_building(df_train, df_test, features, cat_cols):
    """건물별 개별 모델 학습"""
    oof_scores, preds_list = [], []
    
    # 건물별로 개별 모델링
    for building_id in df_train["건물번호"].unique():
        print(f"\n▶ Building {building_id}")
        
        # 해당 건물의 데이터만 추출
        building_train = df_train[df_train["건물번호"] == building_id].copy()
        building_test = df_test[df_test["건물번호"] == building_id].copy()
        
        if len(building_train) < 100:  # 데이터가 너무 적으면 스킵
            continue
            
        y = building_train["전력소비량(kWh)"].values
        X = building_train[features]
        
        # 피처 스케일링
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=3, test_size=24 * 7)
        
        oof_pred_lgb = np.zeros(len(building_train))
        oof_pred_xgb = np.zeros(len(building_train))
        oof_pred_cat = np.zeros(len(building_train))
        
        best_iters_lgb = []
        best_iters_xgb = []
        best_iters_cat = []
        
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            # LightGBM
            model_lgb = train_lightgbm(
                X_scaled.iloc[tr_idx], y[tr_idx],
                X_scaled.iloc[val_idx], y[val_idx],
                cat_cols
            )
            oof_pred_lgb[val_idx] = model_lgb.predict(X_scaled.iloc[val_idx])
            best_iters_lgb.append(model_lgb.best_iteration_)
            
            # XGBoost
            model_xgb = train_xgboost(
                X_scaled.iloc[tr_idx], y[tr_idx],
                X_scaled.iloc[val_idx], y[val_idx]
            )
            oof_pred_xgb[val_idx] = model_xgb.predict(X_scaled.iloc[val_idx])
            best_iters_xgb.append(model_xgb.best_iteration_)
            
            # CatBoost
            model_cat = train_catboost(
                X_scaled.iloc[tr_idx], y[tr_idx],
                X_scaled.iloc[val_idx], y[val_idx],
                cat_cols
            )
            oof_pred_cat[val_idx] = model_cat.predict(X_scaled.iloc[val_idx])
            best_iters_cat.append(model_cat.best_iteration_)
        
        # 앙상블 예측
        oof_pred_ensemble = 0.4 * oof_pred_lgb + 0.35 * oof_pred_xgb + 0.25 * oof_pred_cat
        oof_pred_ensemble[oof_pred_ensemble < 0] = 0
        
        sm = smape_np(y, oof_pred_ensemble)
        print(f"  OOF SMAPE: {sm:.2f}%")
        oof_scores.append(sm)
        
        # 최종 모델 학습
        final_scaler = RobustScaler()
        X_final_scaled = final_scaler.fit_transform(X)
        X_final_scaled = pd.DataFrame(X_final_scaled, columns=X.columns, index=X.index)
        
        # 최종 모델들
        final_model_lgb = train_lightgbm(X_final_scaled, y, X_final_scaled, y, cat_cols)
        final_model_xgb = train_xgboost(X_final_scaled, y, X_final_scaled, y)
        final_model_cat = train_catboost(X_final_scaled, y, X_final_scaled, y, cat_cols)
        
        # 테스트 데이터 예측
        if not building_test.empty:
            X_test = building_test[features]
            X_test_scaled = final_scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            pred_lgb = final_model_lgb.predict(X_test_scaled)
            pred_xgb = final_model_xgb.predict(X_test_scaled)
            pred_cat = final_model_cat.predict(X_test_scaled)
            
            pred_ensemble = 0.4 * pred_lgb + 0.35 * pred_xgb + 0.25 * pred_cat
            pred_ensemble[pred_ensemble < 0] = 0
            
            preds_list.append(pd.DataFrame({
                "num_date_time": building_test["num_date_time"], 
                "answer": pred_ensemble
            }))
    
    print(f"\nAverage OOF SMAPE: {np.mean(oof_scores):.2f}%")
    return pd.concat(preds_list, ignore_index=True)

###########################################################
# Main Pipeline
###########################################################

def run_pipeline(train_path, test_path, info_path):
    """메인 파이프라인"""
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

    # 피처 목록
    feats = [
        "건물번호", "기온(°C)", "풍속(m/s)", "습도(%)", "연면적(m2)", "냉방면적(m2)", 
        "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)", "month", "day", "hour", 
        "weekday", "is_weekend", "is_holiday", "hour_sin", "hour_cos", "day_sin", 
        "day_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos", "year", 
        "quarter", "day_of_year", "week_of_year", "is_night", "is_morning", 
        "is_afternoon", "is_evening", "season", "day_of_year_sin", "day_of_year_cos",
        "week_of_year_sin", "week_of_year_cos", "THI", "temp_x_hour", "temp_x_humid", 
        "temp_x_wind", "humid_x_wind", "temp_x_day", "temp_x_month", "temp_cold", 
        "temp_mild", "temp_hot", "temp_extreme_cold", "temp_extreme_hot", "humid_low",
        "humid_medium", "humid_high", "area_per_floor", "cooling_ratio", "area_log",
        "cooling_area_log", "solar_capacity", "solar_per_area", "solar_log", 
        "ess_capacity", "pcs_capacity", "ess_pcs_ratio", "ess_log", "pcs_log",
        "building_type_encoded", "has_solar", "has_ess", "has_pcs"
    ]
    
    # 시차 및 이동통계 피처 추가
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504]
    rollers = [6, 12, 24, 48, 72, 168, 336]
    
    for lag in lags:
        feats.extend([f"power_lag_{lag}", f"temp_lag_{lag}", f"humid_lag_{lag}", f"wind_lag_{lag}"])
        if lag <= 168:  # 변화율 피처
            feats.extend([f"temp_d{lag}", f"humid_d{lag}", f"wind_d{lag}"])
    
    for win in rollers:
        feats.extend([
            f"power_roll_mean_{win}", f"power_roll_std_{win}", 
            f"power_roll_min_{win}", f"power_roll_max_{win}", f"power_roll_median_{win}",
            f"temp_roll_mean_{win}", f"temp_roll_std_{win}", 
            f"temp_roll_min_{win}", f"temp_roll_max_{win}"
        ])

    df_train = all_df[~all_df["전력소비량(kWh)"].isna()].copy()
    df_test = all_df[all_df["전력소비량(kWh)"].isna()].copy()

    print("모델 학습 및 예측 중...")
    sub_df = train_per_building(df_train, df_test, feats, ["건물번호"])
    
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