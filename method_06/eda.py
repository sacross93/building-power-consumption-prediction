#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.clip(y_pred, 0, None)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def load_data(train_path: Path, test_path: Path, info_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, parse_dates=["일시"], low_memory=False)
    test_df = pd.read_csv(test_path, parse_dates=["일시"], low_memory=False)
    info_df = pd.read_csv(info_path, low_memory=False)
    return train_df, test_df, info_df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_basic(train_df: pd.DataFrame, test_df: pd.DataFrame, info_df: pd.DataFrame) -> dict:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    info_cols = list(info_df.columns)
    missing_in_test = sorted(list(set(train_cols) - set(test_cols)))
    extra_in_test = sorted(list(set(test_cols) - set(train_cols)))

    summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "info_rows": len(info_df),
        "train_cols": train_cols,
        "test_cols": test_cols,
        "info_cols": info_cols,
        "missing_in_test": missing_in_test,
        "extra_in_test": extra_in_test,
        "train_period": (train_df["일시"].min(), train_df["일시"].max()),
        "test_period": (test_df["일시"].min(), test_df["일시"].max()),
        "train_buildings": train_df["건물번호"].nunique(),
        "test_buildings": test_df["건물번호"].nunique(),
        "shared_buildings": len(set(train_df["건물번호"].unique()).intersection(set(test_df["건물번호"].unique()))),
    }
    return summary


def check_num_date_time(df: pd.DataFrame) -> float:
    if "num_date_time" not in df.columns:
        return 0.0
    try:
        expected = df["건물번호"].astype(str) + "_" + df["일시"].dt.strftime("%Y%m%d %H")
        ok_ratio = float((expected == df["num_date_time"].astype(str)).mean())
        return ok_ratio
    except Exception:
        return 0.0


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().rename("missing_count").to_frame()
    miss["missing_ratio"] = miss["missing_count"] / len(df)
    return miss.sort_values("missing_count", ascending=False)


def numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def target_distribution(train_df: pd.DataFrame) -> pd.DataFrame:
    target = "전력소비량(kWh)"
    desc = train_df[target].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_frame("value")
    desc.index.name = "stat"
    return desc


def correlation_with_target(train_df: pd.DataFrame) -> pd.DataFrame:
    target = "전력소비량(kWh)"
    exclude = [target]
    num_cols = numeric_columns(train_df, exclude=exclude)
    if len(num_cols) == 0:
        return pd.DataFrame()
    corr = train_df[num_cols + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
    return corr.rename("corr_with_target").to_frame()


def per_building_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    target = "전력소비량(kWh)"
    stats = train_df.groupby("건물번호")[target].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    return stats


def time_coverage(df: pd.DataFrame) -> pd.DataFrame:
    cov = df.groupby("건물번호").agg(start=("일시", "min"), end=("일시", "max"), n=("일시", "count")).reset_index()
    return cov


def building_info_quality(info_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    report = {}
    for c in cols:
        if c in info_df.columns:
            # count '-' strings and NaNs
            val = info_df[c].astype(str)
            report[c] = {
                "dash_ratio": float((val == "-").mean()),
                "nan_ratio": float((val.isna()).mean()),
            }
    return pd.DataFrame(report).T


def baseline_last24_smape(train_df: pd.DataFrame) -> float:
    """Simple baseline: predict using last-24h lag per building.
    Evaluate on the last 20% time range of train.
    """
    df = train_df.copy().sort_values(["건물번호", "일시"]).reset_index(drop=True)
    target = "전력소비량(kWh)"
    # split by time proportion
    time_sorted = df["일시"].sort_values()
    cutoff = time_sorted.iloc[int(0.8 * len(time_sorted))]
    df["y"] = df[target]
    df["y_lag24"] = df.groupby("건물번호")["y"].shift(24)
    valid_mask = df["일시"] >= cutoff
    sub = df.loc[valid_mask & df["y_lag24"].notna(), ["y", "y_lag24"]]
    if len(sub) == 0:
        return float("nan")
    return smape_np(sub["y"].to_numpy(), sub["y_lag24"].to_numpy())


def run_eda(train_path: Path, test_path: Path, info_path: Path, out_dir: Path) -> None:
    ensure_output_dir(out_dir)
    print("📥 Loading data ...")
    train_df, test_df, info_df = load_data(train_path, test_path, info_path)

    print("📊 Summarizing ...")
    summary = summarize_basic(train_df, test_df, info_df)
    with (out_dir / "data_summary.txt").open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("🔎 Validating num_date_time format ...")
    train_key_ok = check_num_date_time(train_df)
    test_key_ok = check_num_date_time(test_df)
    with (out_dir / "key_checks.txt").open("w", encoding="utf-8") as f:
        f.write(f"train_num_date_time_ok: {train_key_ok:.4f}\n")
        f.write(f"test_num_date_time_ok: {test_key_ok:.4f}\n")

    print("🧩 Missing value reports ...")
    miss_train = missing_report(train_df)
    miss_test = missing_report(test_df)
    miss_info = missing_report(info_df)
    miss_train.to_csv(out_dir / "missing_train.csv")
    miss_test.to_csv(out_dir / "missing_test.csv")
    miss_info.to_csv(out_dir / "missing_info.csv")

    print("📈 Target distribution ...")
    tgt_desc = target_distribution(train_df)
    tgt_desc.to_csv(out_dir / "target_distribution.csv")

    print("📈 Correlations ...")
    corr = correlation_with_target(train_df)
    corr.to_csv(out_dir / "feature_correlations.csv")

    print("🏢 Per-building stats & coverage ...")
    bstats = per_building_stats(train_df)
    bstats.to_csv(out_dir / "per_building_stats.csv", index=False)
    tcov_train = time_coverage(train_df)
    tcov_test = time_coverage(test_df)
    tcov_train.to_csv(out_dir / "time_coverage_train.csv", index=False)
    tcov_test.to_csv(out_dir / "time_coverage_test.csv", index=False)

    print("🏗️ Building info quality ...")
    biq = building_info_quality(info_df)
    biq.to_csv(out_dir / "building_info_quality.csv")

    print("🧪 Baseline (last-24h lag) SMAPE on holdout ...")
    try:
        baseline_smape = baseline_last24_smape(train_df)
    except Exception:
        baseline_smape = float("nan")
    with (out_dir / "baseline_smape.txt").open("w", encoding="utf-8") as f:
        f.write(f"baseline_last24_smape: {baseline_smape:.4f}\n")

    print("✅ EDA report saved to:", out_dir)


def main():
    parser = argparse.ArgumentParser(description="EDA for 2025 Power Usage Forecast (method_06)")
    parser.add_argument("--train", type=Path, default=Path("/home/wlsdud022/ds_test/data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("/home/wlsdud022/ds_test/data/test.csv"))
    parser.add_argument("--info", type=Path, default=Path("/home/wlsdud022/ds_test/data/building_info.csv"))
    parser.add_argument("--out", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/eda_report"))
    args = parser.parse_args()

    run_eda(args.train, args.test, args.info, args.out)


if __name__ == "__main__":
    main()

