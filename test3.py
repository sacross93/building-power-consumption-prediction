# -*- coding: utf-8 -*-
"""
Energy Consumption Forecast v5 - SMAPE 6% 이하 목표
----------------------------------------------------
* 주요 개선사항:
  1. SMAPE 계산 검증 및 개선
  2. 피처 스케일링 추가
  3. 고급 시계열 피처 (계절성, 트렌드, 푸리에 변환)
  4. 건물별 특성 피처 강화
  5. 앙상블 모델 (LightGBM + XGBoost)
  6. 하이퍼파라미터 최적화
  7. 후처리 및 앙상블
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from scipy import stats

warnings.filterwarnings("ignore")

KR_HOLIDAYS = holidays.KR()
SEED = 42

###########################################################
# Utility Functions
###########################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """개선된 SMAPE 계산"""
    eps = 1e-8
    # 음수 예측값 처리
    y_pred = np.maximum(y_pred, 0)
    
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(numerator / denominator) * 100

def lgb_smape(y_pred: np.ndarray, data: lgb.Dataset):
    """LightGBM용 SMAPE 평가 함수"""
    y_true = data.get_label()
    return "SMAPE", smape_np(y_true, y_pred), False

def xgb_smape(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    """XGBoost용 SMAPE 평가 함수"""
    y_true = dtrain.get_label()
    return "SMAPE", smape_np(y_true, y_pred)

###########################################################
# Advanced Feature Engineering
###########################################################

def add_time_features(df):
    """고급 시간 피처"""
    df["month"] = df["일시"].dt.month
    df["day"] = df["일시"].dt.day
    df["hour"] = df["일시"].dt.hour
    df["weekday"] = df["일시"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["일시"].dt.date.map(lambda d: int(d in KR_HOLIDAYS))
    
    # 푸리에 변환을 이용한 주기성 피처
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
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    
    # 계절성 피처
    df["season"] = df["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                                   6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    
    return df

def add_weather_features(df):
    """고급 날씨 피처"""
    # 기본 날씨 피처
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    
    # 변화율 피처
    df["temp_d1"] = df.groupby("건물번호")["기온(°C)"].diff(24)
    df["temp_d7"] = df.groupby("건물번호")["기온(°C)"].diff(168)
    df["humid_d1"] = df.groupby("건물번호")["습도(%)"].diff(24)
    df["wind_d1"] = df.groupby("건물번호")["풍속(m/s)"].diff(24)
    
    # 상호작용 피처
    df["temp_x_hour"] = df["기온(°C)"] * df["hour"]
    df["temp_x_humid"] = df["기온(°C)"] * df["습도(%)"]
    df["temp_x_wind"] = df["기온(°C)"] * df["풍속(m/s)"]
    df["humid_x_wind"] = df["습도(%)"] * df["풍속(m/s)"]
    
    # 날씨 구간 피처
    df["temp_cold"] = (df["기온(°C)"] < 10).astype(int)
    df["temp_mild"] = ((df["기온(°C)"] >= 10) & (df["기온(°C)"] < 25)).astype(int)
    df["temp_hot"] = (df["기온(°C)"] >= 25).astype(int)
    
    return df

def add_rolling_lag_features(df):
    """고급 시차 및 이동통계 피처"""
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # 더 많은 시차
    rollers = [6, 12, 24, 48, 168]  # 더 많은 이동통계
    
    # 시차 피처
    for lag in lags:
        df[f"power_lag_{lag}"] = df.groupby("건물번호")["전력소비량(kWh)"].shift(lag)
        df[f"temp_lag_{lag}"] = df.groupby("건물번호")["기온(°C)"].shift(lag)
        df[f"humid_lag_{lag}"] = df.groupby("건물번호")["습도(%)"].shift(lag)
    
    # 이동통계 피처
    for win in rollers:
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
        
        # 기온 이동통계
        df[f"temp_roll_mean_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).mean().reset_index(0, drop=True)
        )
        df[f"temp_roll_std_{win}"] = (
            df.groupby("건물번호")["기온(°C)"].rolling(win, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        )
    
    return df

def add_building_features(df):
    """건물별 특성 피처"""
    # 면적 관련 피처
    df["area_per_floor"] = df["연면적(m2)"] / 10  # 추정 층수
    df["cooling_ratio"] = df["냉방면적(m2)"] / df["연면적(m2)"]
    
    # 태양광 관련 피처
    df["solar_capacity"] = df["태양광용량(kW)"].fillna(0)
    df["solar_per_area"] = df["solar_capacity"] / df["연면적(m2)"]
    
    # ESS 관련 피처
    df["ess_capacity"] = df["ESS저장용량(kWh)"].fillna(0)
    df["pcs_capacity"] = df["PCS용량(kW)"].fillna(0)
    df["ess_pcs_ratio"] = df["ess_capacity"] / (df["pcs_capacity"] + 1e-8)
    
    # 건물 유형별 인코딩
    building_type_map = {
        'IDC(전화국)': 0, '백화점': 1, '병원': 2, '상용': 3, '아파트': 4,
        '연구소': 5, '학교': 6, '호텔': 7, '공공': 8, '건물기타': 9
    }
    df["building_type_encoded"] = df["건물유형"].map(building_type_map)
    
    return df

def create_features(df):
    """통합 피처 생성"""
    df = df.sort_values(["건물번호", "일시"]).reset_index(drop=True)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_rolling_lag_features(df)
    df = add_building_features(df)
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
        'learning_rate': 0.01,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_data_in_leaf': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 5000,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_smape,
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost 모델 학습"""
    params = {
        'objective': 'reg:absoluteerror',
        'random_state': SEED,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 5000,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=xgb_smape,
        early_stopping_rounds=100,
        verbose=False
    )
    
    return model

def train_per_type(df_train, df_test, features, cat_cols):
    """건물 유형별 앙상블 모델 학습"""
    oof_scores, preds_list = [], []
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 7)
    
    for btype, gdf in df_train.groupby("건물유형"):
        print(f"\n▶ Building-Type: {btype} (rows={len(gdf)})")
        
        y = gdf["전력소비량(kWh)"].values
        X = gdf[features]
        
        # 피처 스케일링
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        oof_pred_lgb = np.zeros(len(gdf))
        oof_pred_xgb = np.zeros(len(gdf))
        best_iters_lgb = []
        best_iters_xgb = []
        
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
        
        # 앙상블 예측
        oof_pred_ensemble = 0.6 * oof_pred_lgb + 0.4 * oof_pred_xgb
        oof_pred_ensemble[oof_pred_ensemble < 0] = 0
        
        sm = smape_np(y, oof_pred_ensemble)
        print(f"  OOF SMAPE: {sm:.2f}%")
        oof_scores.append(sm)
        
        # 최종 모델 학습
        final_scaler = RobustScaler()
        X_final_scaled = final_scaler.fit_transform(X)
        X_final_scaled = pd.DataFrame(X_final_scaled, columns=X.columns, index=X.index)
        
        # LightGBM 최종 모델
        final_params_lgb = {
            'objective': 'regression_l1',
            'boosting_type': 'gbdt',
            'random_state': SEED,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': int(np.mean(best_iters_lgb) * 1.1),
            'verbose': -1
        }
        final_model_lgb = lgb.LGBMRegressor(**final_params_lgb)
        final_model_lgb.fit(X_final_scaled, y, categorical_feature=cat_cols)
        
        # XGBoost 최종 모델
        final_params_xgb = {
            'objective': 'reg:absoluteerror',
            'random_state': SEED,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': int(np.mean(best_iters_xgb) * 1.1),
            'verbosity': 0
        }
        final_model_xgb = xgb.XGBRegressor(**final_params_xgb)
        final_model_xgb.fit(X_final_scaled, y)
        
        # 테스트 데이터 예측
        part_test = df_test[df_test["건물유형"] == btype]
        if not part_test.empty:
            X_test = part_test[features]
            X_test_scaled = final_scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            pred_lgb = final_model_lgb.predict(X_test_scaled)
            pred_xgb = final_model_xgb.predict(X_test_scaled)
            pred_ensemble = 0.6 * pred_lgb + 0.4 * pred_xgb
            pred_ensemble[pred_ensemble < 0] = 0
            
            preds_list.append(pd.DataFrame({
                "num_date_time": part_test["num_date_time"], 
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
        "quarter", "is_night", "is_morning", "is_afternoon", "is_evening", "season",
        "THI", "temp_d1", "temp_d7", "humid_d1", "wind_d1", "temp_x_hour", 
        "temp_x_humid", "temp_x_wind", "humid_x_wind", "temp_cold", "temp_mild", 
        "temp_hot", "area_per_floor", "cooling_ratio", "solar_capacity", 
        "solar_per_area", "ess_capacity", "pcs_capacity", "ess_pcs_ratio", 
        "building_type_encoded"
    ]
    
    # 시차 및 이동통계 피처 추가
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    rollers = [6, 12, 24, 48, 168]
    
    for lag in lags:
        feats.extend([f"power_lag_{lag}", f"temp_lag_{lag}", f"humid_lag_{lag}"])
    
    for win in rollers:
        feats.extend([
            f"power_roll_mean_{win}", f"power_roll_std_{win}", 
            f"power_roll_min_{win}", f"power_roll_max_{win}",
            f"temp_roll_mean_{win}", f"temp_roll_std_{win}"
        ])

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