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
    # 대형마트(할인마트) 일요일 휴무 플래그(2,4번째 일요일 휴무 가정)
    try:
        # nth Sunday in month: ((day-1)//7)+1 when weekday==6
        nth_in_month = ((df["day"] - 1) // 7) + 1
        is_sunday = (df["weekday"] == 6)
        is_mart = (df.get("건물유형").astype(str) == "할인마트") if "건물유형" in df.columns else False
        df["mart_sunday_holiday"] = (is_mart & is_sunday & nth_in_month.isin([2, 4])).astype(int)
    except Exception:
        df["mart_sunday_holiday"] = 0
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # 물리 하한
    if "풍속(m/s)" in df.columns:
        df["풍속(m/s)"] = df["풍속(m/s)"].clip(lower=0)
    # 파생 (온도/습도/풍속만 사용)
    df["THI"] = 9 / 5 * df["기온(°C)"] - 0.55 * (1 - df["습도(%)"] / 100) * (9 / 5 * df["기온(°C)"] - 26) + 32
    df["HDD"] = (18 - df["기온(°C)"]).clip(lower=0)
    df["CDD"] = (df["기온(°C)"] - 22).clip(lower=0)
    # 과민/중복 가능 파생은 제거: dew_point, heat_index, WCT 미생성
    return df


def add_cdh_features(df: pd.DataFrame) -> pd.DataFrame:
    # CDH: Cooling Degree Hours relative to 26°C
    if "기온(°C)" not in df.columns:
        return df
    df = df.sort_values(["건물번호", "일시"]).copy()
    heat = (df["기온(°C)"] - 26).clip(lower=0)
    # 그룹별 rolling sum (윈도우 3/12/24)
    for win in [3, 12, 24]:
        df[f"CDH_{win}"] = (
            heat.groupby(df["건물번호"]).rolling(window=win, min_periods=1).sum().reset_index(level=0, drop=True)
        )
    return df


def add_holiday_adj_features(df: pd.DataFrame) -> pd.DataFrame:
    if "is_holiday" not in df.columns:
        return df
    dt = df["일시"].dt.date
    # 이전/다음날 공휴일 여부
    prev_day = (df["일시"] - pd.Timedelta(days=1)).dt.date
    next_day = (df["일시"] + pd.Timedelta(days=1)).dt.date
    # KR holidays
    try:
        import holidays
        KR = holidays.KR()
        is_prev_holiday = prev_day.map(lambda d: int(d in KR))
        is_next_holiday = next_day.map(lambda d: int(d in KR))
    except Exception:
        is_prev_holiday = 0
        is_next_holiday = 0
    df["pre_holiday"] = is_prev_holiday
    df["post_holiday"] = is_next_holiday
    # 3일 연휴(금/월 포함) 근사: 금요일/월요일이 공휴일 이웃
    weekday = df["weekday"]
    df["long_weekend"] = (((weekday == 4) & (df["post_holiday"] == 1)) | ((weekday == 0) & (df["pre_holiday"] == 1))).astype(int)
    # 방학(7-8월)
    df["is_vacation"] = df["month"].isin([7, 8]).astype(int)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # 시간×온도(THI) 교호항
    if "THI" in df.columns:
        df["hour_THI_sin"] = df["hour_sin"] * df["THI"]
        df["hour_THI_cos"] = df["hour_cos"] * df["THI"]
    return df


def add_normalized_equipment(df: pd.DataFrame) -> pd.DataFrame:
    if "연면적(m2)" not in df.columns:
        return df
    denom = df["연면적(m2)"] + 1e-6
    for col, newc in [
        ("태양광용량(kW)", "pv_per_area"),
        ("ESS저장용량(kWh)", "ess_per_area"),
        ("PCS용량(kW)", "pcs_per_area"),
    ]:
        if col in df.columns:
            df[newc] = df[col] / denom
    return df


def add_recent_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["건물번호", "일시"]).copy()
    for col in ["기온(°C)", "습도(%)"]:
        if col not in df.columns:
            continue
        for win in [6, 24]:
            df[f"{col}_roll_min_{win}"] = (
                df.groupby("건물번호")[col].rolling(win, min_periods=1).min().reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_max_{win}"] = (
                df.groupby("건물번호")[col].rolling(win, min_periods=1).max().reset_index(level=0, drop=True)
            )
        # slope: 현재 - lag_k / k
        for k in [6, 24]:
            lagk = df.groupby("건물번호")[col].shift(k)
            df[f"{col}_slope_{k}"] = (df[col] - lagk) / float(k)
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
    # 설비 보유 플래그(0/1)
    for c, flag in [("태양광용량(kW)", "solar_power_utility"), ("ESS저장용량(kWh)", "ess_utility"), ("PCS용량(kW)", "pcs_utility")]:
        if c in info_df.columns:
            info_df[flag] = (info_df[c] > 0).astype(int)

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
    if getattr(preprocess, "_use_cdh", False):
        all_df = add_cdh_features(all_df)
    if getattr(preprocess, "_use_holiday_adj", False):
        all_df = add_holiday_adj_features(all_df)
    if getattr(preprocess, "_use_interactions", False):
        all_df = add_interactions(all_df)
    if getattr(preprocess, "_use_normalized_equipment", False):
        all_df = add_normalized_equipment(all_df)
    if getattr(preprocess, "_use_recent_weather", False):
        all_df = add_recent_weather(all_df)

    # 로그 변환 (사용자 지정 리스트)
    log_list: list[str] = getattr(preprocess, "_log_transform_list", [])
    for col in log_list:
        if col in all_df.columns:
            newc = f"log_{col}"
            if newc not in all_df.columns:
                # 음수 방지: 음수 존재 시 0으로 시프트하지 않고 생성 생략
                if (all_df[col].dropna() < 0).any():
                    continue
                all_df[newc] = np.log1p(all_df[col])

    # 윈저라이즈(상한 캡핑) – 사용자 지정 리스트에 대해 건물 단위 적용
    win_q: float = getattr(preprocess, "_winsorize_q", 0.0)
    win_cols: list[str] = getattr(preprocess, "_winsorize_cols", [])
    if win_q and win_cols:
        for col in win_cols:
            if col in all_df.columns:
                all_df[col] = winsorize_per_building(all_df, col, upper_q=win_q)

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

    # 옵션: 강수 제거(노이즈 완화 실험용)
    if getattr(preprocess, "_drop_rainfall_flag", False) and "강수량(mm)" in all_df.columns:
        all_df.drop(columns=["강수량(mm)"], inplace=True)

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

    # 원시 면적 컬럼 제거 옵션 (로그/ratio/정규화 사용)
    if getattr(preprocess, "_drop_raw_area", False):
        for c in ["연면적(m2)", "냉방면적(m2)"]:
            if c in all_df.columns:
                all_df.drop(columns=[c], inplace=True)

    # 캘린더 최소화 옵션: day/day_of_year/week_of_year 제거(푸리에/weekday 유지)
    if getattr(preprocess, "_calendar_minimal", False):
        for c in ["day", "day_of_year", "week_of_year"]:
            if c in all_df.columns:
                all_df.drop(columns=[c], inplace=True)

    # 캐시 저장
    ensure_dir(output_dir)
    # 병합/재할당 이후 인덱스가 바뀌어 기존 mask와 정렬이 어긋날 수 있어 재계산
    mask_train_final = ~all_df["전력소비량(kWh)"].isna()
    df_train = all_df.loc[mask_train_final].copy()
    df_test = all_df.loc[~mask_train_final].copy()
    # 안전: 학습 타깃 결손 제거
    if "log_power" in df_train.columns:
        df_train = df_train[df_train["log_power"].notna()].copy()
    # 일별 온도 집계(최/평/최저 및 일교차) 생성
    try:
        for df in (df_train, df_test):
            grp = df.groupby(["건물번호", "month", "day"])  # 동일 월/일 기준
            df["day_max_temperature"] = grp["기온(°C)"].transform("max")
            df["day_mean_temperature"] = grp["기온(°C)"].transform("mean")
            df["day_min_temperature"] = grp["기온(°C)"].transform("min")
            df["day_temperature_range"] = df["day_max_temperature"] - df["day_min_temperature"]
    except Exception:
        pass
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
    parser.add_argument("--drop-rainfall", action="store_true", help="강수량(mm) 피처를 제거합니다")
    parser.add_argument("--use-cdh", action="store_true")
    parser.add_argument("--use-holiday-adj", action="store_true")
    parser.add_argument("--use-interactions", action="store_true")
    parser.add_argument("--use-normalized-equipment", action="store_true")
    parser.add_argument("--use-recent-weather", action="store_true")
    parser.add_argument("--log-transform", type=str, default="", help="comma-separated feature names to log1p (create log_<col>)")
    parser.add_argument("--winsorize-q", type=float, default=0.0, help="upper quantile for winsorize (e.g., 0.995)")
    parser.add_argument("--winsorize-cols", type=str, default="", help="comma-separated feature names to winsorize")
    parser.add_argument("--drop-raw-area", action="store_true", help="drop raw area columns (연면적/냉방면적)")
    parser.add_argument("--calendar-minimal", action="store_true", help="drop day/day_of_year/week_of_year")
    args = parser.parse_args()

    # 내부 플래그 세팅(간단 전달)
    preprocess._drop_rainfall_flag = bool(args.drop_rainfall)
    preprocess._use_cdh = bool(args.use_cdh)
    preprocess._use_holiday_adj = bool(args.use_holiday_adj)
    preprocess._use_interactions = bool(args.use_interactions)
    preprocess._use_normalized_equipment = bool(args.use_normalized_equipment)
    preprocess._use_recent_weather = bool(args.use_recent_weather)
    preprocess._log_transform_list = [s.strip() for s in args.log_transform.split(',') if s.strip()]
    preprocess._winsorize_q = float(args.winsorize_q) if args.winsorize_q else 0.0
    preprocess._winsorize_cols = [s.strip() for s in args.winsorize_cols.split(',') if s.strip()]
    preprocess._drop_raw_area = bool(args.drop_raw_area)
    preprocess._calendar_minimal = bool(args.calendar_minimal)
    preprocess(args.train, args.test, args.info, args.out, drop_target_lags=not args.keep_target_lags)


if __name__ == "__main__":
    main()

