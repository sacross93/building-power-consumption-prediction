import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def configure_matplotlib_font() -> None:
    """한글 경고를 줄이기 위해 사용 가능한 폰트를 우선적으로 설정.
    시스템에 폰트가 없을 수 있으므로 best-effort로만 처리한다.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        # 폰트 후보: NanumGothic, Malgun Gothic(Windows), AppleGothic(macOS), DejaVu Sans(기본)
        from matplotlib import font_manager, rcParams

        font_candidates: List[str] = [
            "NanumGothic",
            "Malgun Gothic",
            "AppleGothic",
            "DejaVu Sans",
        ]

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in font_candidates:
            if name in available:
                rcParams["font.family"] = name
                break

        # 마이너스 기호 깨짐 방지
        rcParams["axes.unicode_minus"] = False
    except Exception:
        # 폰트 설정 실패해도 그래프 저장은 가능하므로 무시
        pass


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("%", "pct")
    )


def coerce_known_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_like = [
        "연면적(m2)",
        "냉방면적(m2)",
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("-", "0", regex=False), errors="coerce"
            )
    return df


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_dir = out_dir / "numeric"
    make_dirs(numeric_dir)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        # Histogram + KDE
        plt.figure(figsize=(7, 4))
        sns.histplot(series, bins=50, kde=True)
        plt.title(f"Distribution - {col}")
        plt.tight_layout()
        plt.savefig(numeric_dir / f"dist_{safe_filename(col)}.png")
        plt.close()

        # Boxplot
        plt.figure(figsize=(7, 2.8))
        sns.boxplot(x=series)
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(numeric_dir / f"box_{safe_filename(col)}.png")
        plt.close()

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(min(0.6 * len(num_cols) + 3, 18), min(0.6 * len(num_cols) + 3, 14)))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation (numeric)")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_numeric.png")
        plt.close()


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    categorical_dir = out_dir / "categorical"
    make_dirs(categorical_dir)

    num_cols = set(df.select_dtypes(include=[np.number]).columns.tolist())
    cat_cols = [c for c in df.columns if c not in num_cols and c != "일시"]

    for col in cat_cols:
        vc = df[col].astype(str).value_counts().head(30)
        if vc.empty:
            continue
        plt.figure(figsize=(max(7, 0.3 * len(vc) + 4), 4.2))
        sns.barplot(x=vc.index, y=vc.values)
        plt.title(f"Count - {col} (top 30)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(categorical_dir / f"count_{safe_filename(col)}.png")
        plt.close()


def plot_missing_bar(df: pd.DataFrame, out_dir: Path) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    miss = df.isna().mean().sort_values(ascending=False)
    if miss.max() == 0:
        return
    plt.figure(figsize=(max(7, 0.25 * len(miss) + 4), 5))
    sns.barplot(x=miss.index, y=miss.values)
    plt.title("Missing Ratio by Column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_ratio.png")
    plt.close()


def run(input_csv: Path, out_dir: Path) -> None:
    print(f"[INFO] 입력 경로: {input_csv}")
    print(f"[INFO] 출력 디렉토리: {out_dir}")
    make_dirs(out_dir)

    configure_matplotlib_font()

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df = coerce_known_numeric_columns(df)

    # 그래프 생성
    plot_numeric_distributions(df, out_dir)
    plot_categorical_counts(df, out_dir)
    plot_missing_bar(df, out_dir)

    print(f"[DONE] 그래프 저장 완료 → {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="간단 EDA 분포 그래프 생성기")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "train_building_merged.csv"),
        help="입력 CSV 경로 (기본: method_07/train_building_merged.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eda/train_building_merged"),
        help="출력 디렉토리 (기본: method_07/eda/train_building_merged)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input), Path(args.outdir))








