#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def convert_capacity_to_zero(info_df: pd.DataFrame) -> pd.DataFrame:
    capacity_cols = [
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
    ]
    for c in capacity_cols:
        if c in info_df.columns:
            info_df[c] = (
                info_df[c]
                .replace('-', 0)
                .replace({np.nan: 0})
            )
            # 안전 캐스팅
            info_df[c] = pd.to_numeric(info_df[c], errors="coerce").fillna(0.0)
    return info_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["일시"]
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    # 공휴일
    try:
        import holidays
        KR = holidays.KR()
        df["is_holiday"] = dt.dt.date.map(lambda d: int(d in KR))
    except Exception:
        df["is_holiday"] = 0
    # Fourier
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    # 순서/추세
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # 물리 하한
    if "풍속(m/s)" in df.columns:
        df["풍속(m/s)"] = df["풍속(m/s)"].clip(lower=0)
    # 파생 (온도/습도/풍속만 사용)
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    df["dew_point"] = df["기온(°C)"] - (100 - df["습도(%)"]) / 5
    df["heat_index"] = 0.5 * (df["기온(°C)"] + 61.0 + (df["기온(°C)"] - 68.0) * 1.2 + df["습도(%)"] * 0.094)
    df["HDD"] = (18 - df["기온(°C)"]).clip(lower=0)
    df["CDD"] = (df["기온(°C)"] - 22).clip(lower=0)
    df["WCT"] = (
        13.12 + 0.6125 * df["기온(°C)"] - 11.37 * (df["풍속(m/s)"] ** 0.16) + 0.3965 * (df["풍속(m/s)"] ** 0.16) * df["기온(°C)"]
    )
    return df


def create_lag_rolling_no_leak(df: pd.DataFrame, group_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
    # 정렬 보장
    df = df.sort_values(group_cols + ["일시"]).copy()
    for col in target_cols:
        # lag
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f"{col}_lag_{lag}"] = df.groupby(group_cols)[col].shift(lag)
        # rolling from shifted series
        shifted = df.groupby(group_cols)[col].shift(1)
        for win in [6, 24]:
            roll = (
                shifted.groupby(df[group_cols].apply(tuple, axis=1))
                .rolling(window=win)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_mean_{win}"] = roll
        shifted = df.groupby(group_cols)[col].shift(1)
        for win in [6, 24]:
            roll = (
                shifted.groupby(df[group_cols].apply(tuple, axis=1))
                .rolling(window=win)
                .std()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{win}"] = roll
    return df


def winsorize_per_building(df: pd.DataFrame, col: str, upper_q: float = 0.995) -> pd.Series:
    def _cap(group: pd.Series) -> pd.Series:
        if group.isna().all():
            return group
        cap = group.quantile(upper_q)
        return group.clip(upper=cap)
    return df.groupby("건물번호")[col].transform(_cap)


def preprocess(train_path: Path, test_path: Path, info_path: Path, output_dir: Path, drop_target_lags: bool = True) -> None:
    print("📥 데이터 로드 ...")
    train_df = pd.read_csv(train_path, parse_dates=["일시"], low_memory=False)
    test_df = pd.read_csv(test_path, parse_dates=["일시"], low_memory=False)
    info_df = pd.read_csv(info_path, low_memory=False)

    # dtype
    train_df["건물번호"] = train_df["건물번호"].astype("category")
    test_df["건물번호"] = test_df["건물번호"].astype("category")
    if "건물번호" in info_df.columns:
        info_df["건물번호"] = info_df["건물번호"].astype("category")
    if "건물유형" in info_df.columns:
        info_df["건물유형"] = info_df["건물유형"].astype("category")

    # 설비 용량 처리: '-' / NaN -> 0
    info_df = convert_capacity_to_zero(info_df)

    # test에 타깃 없음 지정
    test_df["전력소비량(kWh)"] = np.nan
    # 합치기
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.merge(info_df, on="건물번호", how="left")

    # 타깃 정리: 음수 제거 + 건물별 상위 극단값 캡핑(train만)
    all_df["전력소비량(kWh)"] = all_df["전력소비량(kWh)"].clip(lower=0)
    mask_train = ~all_df["전력소비량(kWh)"].isna()
    all_df.loc[mask_train, "전력소비량(kWh)"] = winsorize_per_building(all_df.loc[mask_train], "전력소비량(kWh)")

    # 로그 타깃 및 면적/용량 파생
    all_df["log_power"] = np.log1p(all_df["전력소비량(kWh)"])
    for col in ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]:
        if col in all_df.columns:
            all_df[f"log_{col}"] = np.log1p(all_df[col].fillna(0))
    if "연면적(m2)" in all_df.columns and "냉방면적(m2)" in all_df.columns:
        all_df["cooling_ratio"] = all_df["냉방면적(m2)"] / (all_df["연면적(m2)"] + 1e-6)

    # 시간 파생 + 날씨 파생
    all_df = add_time_features(all_df)
    all_df = add_weather_features(all_df)

    # 일조/일사 대체 피처: 테스트에는 없으므로 8월 시간대 평균으로 근사치 생성 후 원 컬럼 제거
    try:
        if "일조(hr)" in all_df.columns and "일사(MJ/m2)" in all_df.columns:
            aug_mask = mask_train & (all_df["month"] == 8)
            # 시간대 평균 계산 (훈련 8월 기준)
            sunshine_hour_mean = (
                all_df.loc[aug_mask].groupby("hour")["일조(hr)"].mean()
                if aug_mask.any() else None
            )
            solar_hour_mean = (
                all_df.loc[aug_mask].groupby("hour")["일사(MJ/m2)"].mean()
                if aug_mask.any() else None
            )
            # 에스티메이터 컬럼 생성
            all_df["sunshine_est"] = all_df["일조(hr)"]
            all_df["solar_est"] = all_df["일사(MJ/m2)"]
            # 테스트 구간 보강: 시간대 평균으로 채움, 남으면 전체 평균
            if sunshine_hour_mean is not None:
                all_df.loc[~mask_train, "sunshine_est"] = (
                    all_df.loc[~mask_train, "hour"].map(sunshine_hour_mean)
                )
            if solar_hour_mean is not None:
                all_df.loc[~mask_train, "solar_est"] = (
                    all_df.loc[~mask_train, "hour"].map(solar_hour_mean)
                )
            # 전체 평균으로 최종 보강
            all_df["sunshine_est"] = all_df["sunshine_est"].fillna(all_df["sunshine_est"].mean())
            all_df["solar_est"] = all_df["solar_est"].fillna(all_df["solar_est"].mean())
            # 원본 컬럼 제거 (테스트에 없어 불일치 방지)
            all_df.drop(columns=["일조(hr)", "일사(MJ/m2)"], inplace=True)
    except Exception:
        # 문제 발생 시 조용히 패스하여 파이프라인 지속
        pass

    # 결측 보강(보수적): 날씨 NaN은 건물×시간대 중앙값 → 전체 중앙값 순서로 대체
    for c in ["기온(°C)", "습도(%)", "풍속(m/s)", "강수량(mm)"]:
        if c in all_df.columns:
            med = all_df.groupby(["건물번호", "hour"])[c].transform("median")
            all_df[c] = all_df[c].fillna(med)
            all_df[c] = all_df[c].fillna(all_df[c].median())

    # lag & rolling (누수 방지)
    all_df = create_lag_rolling_no_leak(
        all_df,
        group_cols=["건물번호"],
        target_cols=[col for col in ["전력소비량(kWh)", "기온(°C)", "습도(%)"] if col in all_df.columns],
    )

    # 옵션: 타깃 기반 랙/롤링 피처 제거 (테스트 구간 정보 공백으로 인한 일반화 저하 방지)
    if drop_target_lags:
        drop_cols = [c for c in all_df.columns if c.startswith("전력소비량(kWh)_lag_") or c.startswith("전력소비량(kWh)_roll_")]
        if drop_cols:
            all_df.drop(columns=drop_cols, inplace=True)

    # 통계 피처 (train으로 추정)
    print("통계 피처 계산 중 ...")
    train_stats = all_df.loc[mask_train].copy()
    # 건물×시간×요일
    bhw = (
        train_stats.groupby(["건물번호", "hour", "weekday"])["전력소비량(kWh)"]
        .agg(["mean", "std"]).reset_index()
    )
    bhw.columns = ["건물번호", "hour", "weekday", "building_hour_weekday_mean", "building_hour_weekday_std"]
    # 건물×시간
    bh = (
        train_stats.groupby(["건물번호", "hour"])["전력소비량(kWh)"]
        .agg(["mean", "std"]).reset_index()
    )
    bh.columns = ["건물번호", "hour", "building_hour_mean", "building_hour_std"]
    # 건물×월
    bm = (
        train_stats.groupby(["건물번호", "month"])["전력소비량(kWh)"]
        .agg(["mean", "std"]).reset_index()
    )
    bm.columns = ["건물번호", "month", "building_month_mean", "building_month_std"]

    # 머지
    all_df = all_df.merge(bhw, on=["건물번호", "hour", "weekday"], how="left")
    all_df = all_df.merge(bh, on=["건물번호", "hour"], how="left")
    all_df = all_df.merge(bm, on=["건물번호", "month"], how="left")

    # NaN 채움(보수적)
    overall_mean = train_stats["전력소비량(kWh)"].mean()
    overall_std = train_stats["전력소비량(kWh)"].std()
    for c in [
        "building_hour_weekday_mean",
        "building_hour_weekday_std",
        "building_hour_mean",
        "building_hour_std",
        "building_month_mean",
        "building_month_std",
    ]:
        if c in all_df.columns:
            if c.endswith("mean"):
                all_df[c] = all_df[c].fillna(overall_mean)
            else:
                all_df[c] = all_df[c].fillna(overall_std)

    # 범주형 유지
    all_df["건물번호"] = all_df["건물번호"].astype("category")

    # 캐시 저장
    ensure_dir(output_dir)
    # 병합/재할당 이후 인덱스가 바뀌어 기존 mask와 정렬이 어긋날 수 있어 재계산
    mask_train_final = ~all_df["전력소비량(kWh)"].isna()
    df_train = all_df.loc[mask_train_final].copy()
    df_test = all_df.loc[~mask_train_final].copy()
    # 안전: 학습 타깃 결손 제거
    if "log_power" in df_train.columns:
        df_train = df_train[df_train["log_power"].notna()].copy()
    train_out = output_dir / "train_preprocessed.parquet"
    test_out = output_dir / "test_preprocessed.parquet"
    df_train.to_parquet(train_out, index=False)
    df_test.to_parquet(test_out, index=False)
    print(f"✅ 전처리 완료 → {train_out} | {test_out}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess for method_06")
    parser.add_argument("--train", type=Path, default=Path("/home/wlsdud022/ds_test/data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("/home/wlsdud022/ds_test/data/test.csv"))
    parser.add_argument("--info", type=Path, default=Path("/home/wlsdud022/ds_test/data/building_info.csv"))
    parser.add_argument("--out", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/cache"))
    parser.add_argument("--keep-target-lags", action="store_true", help="전력소비량 타깃 기반 lag/rolling 피처를 유지합니다")
    args = parser.parse_args()

    preprocess(args.train, args.test, args.info, args.out, drop_target_lags=not args.keep_target_lags)


if __name__ == "__main__":
    main()

