import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def coerce_known_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_like = [
        "연면적(m2)",
        "냉방면적(m2)",
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
        "전력소비량(kWh)",
        "기온(°C)",
        "습도(%)",
        "일조(hr)",
        "일사(MJ/m2)",
        "풍속(m/s)",
        "강수량(mm)",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("-", "0", regex=False), errors="coerce"
            )
    return df


def ensure_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    # 기대 컬럼: '일시' 또는 'num_date_time' 유사. 없으면 스킵
    if "일시" in df.columns:
        # 형식: YYYY-MM-DD HH:MM 또는 유사 → pandas 변환 시도
        try:
            ts = pd.to_datetime(df["일시"], errors="coerce")
        except Exception:
            ts = pd.to_datetime(df["일시"].astype(str), errors="coerce")
    elif "num_date_time" in df.columns:
        # 예: "100_20240823 08"와 같이 숫자 접두가 있을 수 있으므로 뒤의 날짜/시간만 추출 시도
        ts = pd.to_datetime(
            df["num_date_time"].astype(str).str.extract(r"(\d{8}\s\d{2})")[0],
            format="%Y%m%d %H",
            errors="coerce",
        )
    else:
        return df

    df = df.copy()
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (df["weekday"].isin([5, 6])).astype(int)

    # 주기성 인코딩
    for name, period in [("hour", 24), ("month", 12), ("weekday", 7)]:
        if name in df.columns:
            angle = 2 * np.pi * df[name] / period
            df[f"{name}_sin"] = np.sin(angle)
            df[f"{name}_cos"] = np.cos(angle)
    return df


def compute_thi(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    # Temperature-Humidity Index (간단 공식)
    # THI = T - (0.55 - 0.0055*RH) * (T - 14.5)
    T = temperature_c.astype(float)
    RH = humidity_pct.astype(float)
    return T - (0.55 - 0.0055 * RH) * (T - 14.5)


def compute_absolute_humidity(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    # 절대습도 AH (g/m^3) 근사
    # es: 포화수증기압(hPa), e = RH/100 * es, AH = 2.1674 * e / (T + 273.15)
    T = temperature_c.astype(float)
    RH = humidity_pct.astype(float)
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = (RH / 100.0) * es
    ah = 2.1674 * e / (T + 273.15)  # g/m^3
    return ah


def degree_day(series: pd.Series, base: float, mode: str) -> pd.Series:
    if mode == "CDD":
        return np.maximum(series - base, 0.0)
    if mode == "HDD":
        return np.maximum(base - series, 0.0)
    raise ValueError("mode must be 'CDD' or 'HDD'")


def rolling_sum_shifted(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).sum().shift(1)


def two_part_zeroinfl_features(df: pd.DataFrame, col: str, upper_q: float = 0.995) -> pd.DataFrame:
    """제로-인플레이션 컬럼에 대해 플래그 + 양수 로그 + 윈저라이즈 파생을 생성.
    생성 컬럼:
      - is_{col}_positive (0/1)
      - {col}_log_pos (0 또는 log1p(x))
    양수 구간은 상위 분위수로 클리핑(윈저라이즈) 후 로그 적용.
    """
    if col not in df.columns:
        return df

    s = pd.to_numeric(df[col], errors="coerce")
    is_pos = (s > 0).astype(int)
    s_pos = s.where(s > 0, np.nan)

    # 윈저라이즈 상한 (양수 값만)
    try:
        upper = float(s_pos.quantile(upper_q))
    except Exception:
        upper = np.nan
    if np.isfinite(upper):
        s_pos = s_pos.clip(upper=upper)

    s_log_pos = np.log1p(s_pos)

    out = df.copy()
    out[f"is_{col}_positive"] = is_pos
    out[f"{col}_log_pos"] = s_log_pos.fillna(0.0)
    return out


def add_equipment_area_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 설비 보유 플래그
    for c in ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]:
        if c in out.columns:
            out[f"{c}_utility"] = (pd.to_numeric(out[c], errors="coerce") > 0).astype(int)

    # 면적 정규화
    if "연면적(m2)" in out.columns:
        area = pd.to_numeric(out["연면적(m2)"], errors="coerce")
        for c in ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]:
            if c in out.columns:
                out[f"{c.split('(')[0]}_per_area"] = pd.to_numeric(out[c], errors="coerce") / (area + 1e-6)

    # 냉방 비율
    if "연면적(m2)" in out.columns and "냉방면적(m2)" in out.columns:
        out["cooling_ratio"] = pd.to_numeric(out["냉방면적(m2)"], errors="coerce") / (
            pd.to_numeric(out["연면적(m2)"], errors="coerce") + 1e-6
        )
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ("기온(°C)" in out.columns) and ("습도(%)" in out.columns):
        out["THI"] = compute_thi(out["기온(°C)"], out["습도(%)"])
        out["AH_gm3"] = compute_absolute_humidity(out["기온(°C)"], out["습도(%)"])

        # CDD/HDD (다중 임계)
        for base in [18, 22, 24, 26, 28, 30]:
            out[f"CDD{base}"] = degree_day(out["기온(°C)"], base=float(base), mode="CDD")
        out["HDD18"] = degree_day(out["기온(°C)"], base=18.0, mode="HDD")

        # CDH (누수 방지 위해 shift(1))
        out["CDH26_3"] = rolling_sum_shifted(out["CDD26"], window=3)
        out["CDH26_12"] = rolling_sum_shifted(out["CDD26"], window=12)
        out["CDH26_24"] = rolling_sum_shifted(out["CDD26"], window=24)

    # 물리 하한 보정
    if "풍속(m/s)" in out.columns:
        out["풍속(m/s)"] = out["풍속(m/s)"].clip(lower=0)
    return out


def add_solar_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 2-파트 변환 (일사/일조/강수/설비)
    zeroinfl_cols = [
        "일사(MJ/m2)",
        "일조(hr)",
        "강수량(mm)",
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
    ]
    for c in zeroinfl_cols:
        if c in out.columns:
            out = two_part_zeroinfl_features(out, c, upper_q=0.995)

    # 시간 주기와 일사 상호작용
    if "일사(MJ/m2)" in out.columns and "hour_sin" in out.columns:
        out["solar_hour_sin"] = out["일사(MJ/m2)"] * out["hour_sin"]
        out["solar_hour_cos"] = out["일사(MJ/m2)"] * out["hour_cos"]

    # 설비 보유 × 일사
    if "태양광용량(kW)_utility" in out.columns and "일사(MJ/m2)" in out.columns:
        out["pv_utility_solar"] = out["태양광용량(kW)_utility"] * out["일사(MJ/m2)"]
    return out


def add_building_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """건물×hour×weekday 평균을 베이스라인으로 만들고 편차/비율을 추가.
    경고: 학습/검증 분리 전 전체로 추정하면 누수 우려가 있으나,
    본 스크립트는 탐색/학습 전처리 재현을 위한 일괄 통계로 제한한다.
    """
    if "전력소비량(kWh)" not in df.columns:
        return df
    out = df.copy()
    keys = []
    for k in ["건물번호", "hour", "weekday"]:
        if k in out.columns:
            keys.append(k)
    if len(keys) < 2:
        return out
    grp = out.groupby(keys)["전력소비량(kWh)"].mean().rename("building_hour_weekday_mean").reset_index()
    out = out.merge(grp, on=keys, how="left")
    base = out["building_hour_weekday_mean"]
    power = out["전력소비량(kWh)"]
    out["load_dev"] = power - base
    out["load_ratio"] = power / (base + 1e-6)
    return out


def run(input_csv: Path, out_dir: Path, save_csv: bool = False, output_name: str | None = None) -> None:
    print(f"[INFO] 입력: {input_csv}")
    print(f"[INFO] 출력 디렉토리: {out_dir}")
    make_dirs(out_dir)

    df = read_csv_utf8(input_csv)
    df = coerce_known_numeric_columns(df)
    df = ensure_time_parts(df)
    df = add_weather_features(df)
    df = add_equipment_area_features(df)
    df = add_solar_interactions(df)
    df = add_building_baseline(df)

    # 저장
    # 출력 파일명 자동 결정: 타깃 존재 여부로 train/test 구분
    has_target = "전력소비량(kWh)" in df.columns and not pd.isna(df["전력소비량(kWh)"].head(10)).all()
    base_name = output_name or ("train_engineered" if has_target else "test_engineered")
    parquet_path = out_dir / f"{base_name}.parquet"
    csv_path = out_dir / f"{base_name}.csv"
    parquet_ok = True
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_ok = False
    if save_csv or not parquet_ok:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 생성 요약
    created_cols = sorted(set(df.columns) - set(read_csv_utf8(input_csv).columns))
    summary_path = out_dir / "feature_engineer_summary.txt"
    saved_files = []
    if parquet_ok:
        saved_files.append(str(parquet_path))
    if save_csv or not parquet_ok:
        saved_files.append(str(csv_path))
    summary_lines = [
        "생성된 파생 컬럼 목록",
        "=",
        *[f"- {c}" for c in created_cols],
        "",
        "저장 파일",
        "=",
        *[f"- {p}" for p in saved_files],
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[DONE] 저장: {', '.join(saved_files)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="전력사용량 예측용 파생변수 생성 스크립트")
    p.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "train_building_merged.csv"),
        help="입력 CSV 경로",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache"),
        help="출력 디렉토리 (기본: method_07/cache)",
    )
    p.add_argument("--save-csv", action="store_true", help="CSV도 함께 저장")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input), Path(args.outdir), save_csv=args.save_csv)


