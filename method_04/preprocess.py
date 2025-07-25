import argparse
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

############################################################
# Helper: Hampel filter for outlier smoothing
############################################################

def hampel_filter(series: pd.Series, window_size: int = 24, n_sigmas: float = 3.0) -> pd.Series:
    """간단한 Hampel 필터 구현 (window_size 시간 창 기준)"""
    rolling_median = series.rolling(window=window_size, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = diff.rolling(window=window_size, center=True).median()
    threshold = n_sigmas * 1.4826 * mad  # 1.4826: MAD -> std 근사
    outlier_idx = diff > threshold
    series_clean = series.copy()
    series_clean[outlier_idx] = rolling_median[outlier_idx]
    return series_clean

############################################################
# Feature Engineering (시간 & 날씨) – test5.py 에서 사용한 함수 일부 재활용
############################################################

KR_HOLIDAYS = None  # holidays 패키지 사용은 지연 import(속도)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["일시"]
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    global KR_HOLIDAYS
    if KR_HOLIDAYS is None:
        import holidays
        KR_HOLIDAYS = holidays.KR()
    df["is_holiday"] = dt.dt.date.map(lambda d: int(d in KR_HOLIDAYS))
    # Fourier 변환
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    # 추세용
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["time_idx"] = (dt - dt.min()).dt.days
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # 가정: 기온(°C), 습도(%), 풍속(m/s) 컬럼이 존재
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    df["dew_point"] = df["기온(°C)"] - (100 - df["습도(%)"]) / 5
    df["heat_index"] = 0.5 * (df["기온(°C)"] + 61.0 + (df["기온(°C)"] - 68.0) * 1.2 + df["습도(%)"] * 0.094)
    # Degree Days
    df["HDD"] = (18 - df["기온(°C)"]).clip(lower=0)
    df["CDD"] = (df["기온(°C)"] - 22).clip(lower=0)
    # 온도 구간 (Bin)
    bins = [-50, 10, 15, 20, 25, 30, 60]
    labels = ["temp_bin_1", "temp_bin_2", "temp_bin_3", "temp_bin_4", "temp_bin_5", "temp_bin_6"]
    df["temp_bin"] = pd.cut(df["기온(°C)"], bins=bins, labels=labels)
    return df

############################################################
# Lag & Rolling 생성기
############################################################

def create_lag_rolling(df: pd.DataFrame, group_cols: list, target_cols: list) -> pd.DataFrame:
    """주어진 타겟 컬럼에 대해 랙과 롤링 피처를 생성한다."""
    for col in target_cols:
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f"{col}_lag_{lag}"] = df.groupby(group_cols)[col].shift(lag)
        # rolling 6/24 window mean & std (단기, 일간)
        for win in [6, 24]:
            df[f"{col}_roll_mean_{win}"] = (
                df.groupby(group_cols)[col].rolling(window=win).mean().reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{win}"] = (
                df.groupby(group_cols)[col].rolling(window=win).std().reset_index(level=0, drop=True)
            )
    return df

############################################################
# 메인 전처리 함수
############################################################

def preprocess(train_path: Path, test_path: Path, info_path: Path, output_dir: Path):
    print("📥 데이터 로드 중 ...")
    train_df = pd.read_csv(train_path, parse_dates=["일시"], low_memory=False)
    test_df = pd.read_csv(test_path, parse_dates=["일시"], low_memory=False)
    info_df = pd.read_csv(info_path)

    # dtype 설정
    train_df["건물번호"] = train_df["건물번호"].astype("category")
    test_df["건물번호"] = test_df["건물번호"].astype("category")
    info_df["건물번호"] = info_df["건물번호"].astype("category")
    info_df["건물유형"] = info_df["건물유형"].astype("category")

    ########################################################
    # 1) 결측치 기호 "-" 처리 및 플래그 생성
    ########################################################
    miss_cols = ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    for c in miss_cols:
        info_df[f"{c}_missing"] = (info_df[c] == "-").astype(int)
    num_cols = ["연면적(m2)", "냉방면적(m2)"] + miss_cols
    info_df[num_cols] = info_df[num_cols].replace("-", np.nan).astype(float)

    # 건물유형별 중앙값으로 보간
    for col in num_cols:
        info_df[col] = info_df.groupby("건물유형")[col].transform(lambda x: x.fillna(x.median()))

    ########################################################
    # 2) train / test 합치기 & 전력소비 클리핑
    ########################################################
    test_df["전력소비량(kWh)"] = np.nan
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.merge(info_df, on="건물번호", how="left")

    # 음수 제거, 극단치 winsorize
    all_df["전력소비량(kWh)"] = all_df["전력소비량(kWh)"].clip(lower=0)
    # winsorize per building (99.5 pct)
    def _winsorize(group):
        if group.isna().all():
            return group
        cap = group.quantile(0.995)
        return group.clip(upper=cap)
    all_df["전력소비량(kWh)"] = all_df.groupby("건물번호")["전력소비량(kWh)"].transform(_winsorize)

    ########################################################
    # 3) Hampel 필터 (log-space) – train 영역만 적용
    ########################################################
    mask_train = ~all_df["전력소비량(kWh)"].isna()
    temp_series = np.log1p(all_df.loc[mask_train, "전력소비량(kWh)"])
    all_df.loc[mask_train, "전력소비량(kWh)"] = np.expm1(hampel_filter(temp_series))

    ########################################################
    # 4) 타겟 변환 & 면적/용량 로그 변환
    ########################################################
    all_df["log_power"] = np.log1p(all_df["전력소비량(kWh)"])
    for col in ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]:
        all_df[f"log_{col}"] = np.log1p(all_df[col])
    all_df["cooling_ratio"] = all_df["냉방면적(m2)"] / (all_df["연면적(m2)"] + 1e-6)

    ########################################################
    # 5) 시간 & 날씨 파생
    ########################################################
    all_df = add_time_features(all_df)
    all_df = add_weather_features(all_df)

    ########################################################
    # 6) 시차 / 롤링 피처 (주요 변수)
    ########################################################
    all_df = create_lag_rolling(
        all_df,
        group_cols=["건물번호"],
        target_cols=["전력소비량(kWh)", "기온(°C)", "습도(%)"]
    )

    ########################################################
    # 7) 범주형 temp_bin, 건물유형, 건물번호 유지
    ########################################################
    all_df["temp_bin"] = all_df["temp_bin"].astype("category")

    ########################################################
    # 8) 캐싱 – train / test 분리 저장
    ########################################################
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train = all_df[mask_train].copy()
    df_test = all_df[~mask_train].copy()
    train_out = output_dir / "train_preprocessed.parquet"
    test_out = output_dir / "test_preprocessed.parquet"
    df_train.to_parquet(train_out, index=False)
    df_test.to_parquet(test_out, index=False)
    print(f"✅ 전처리 완료 → {train_out.name}, {test_out.name} 저장")

############################################################
# CLI
############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="에너지 소비 예측용 전처리 스크립트 (method_04)")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--info", type=Path, default=Path("data/building_info.csv"))
    parser.add_argument("--out", type=Path, default=Path("method_04/cache"))
    args = parser.parse_args()

    preprocess(args.train, args.test, args.info, args.out) 