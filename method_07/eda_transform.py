import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def configure_matplotlib_font() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager, rcParams

        # Best-effort 한글 폰트 설정
        candidates: List[str] = [
            "NanumGothic",
            "Malgun Gothic",
            "AppleGothic",
            "DejaVu Sans",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                rcParams["font.family"] = name
                break
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("%", "pct")
    )


def coerce_known_numeric(df: pd.DataFrame) -> pd.DataFrame:
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


def choose_transform(series: pd.Series) -> Tuple[str, pd.Series, float, float]:
    """시리즈의 왜도를 줄이는 변환을 선택하여 반환.
    반환: (선택된 변환명, 변환값 시리즈, 원본 왜도, 변환 왜도)
    지원 변환: log1p(비음수), yeo-johnson
    """
    from sklearn.preprocessing import PowerTransformer
    from scipy.stats import skew

    s = series.dropna()
    if s.empty:
        return ("none", series, float("nan"), float("nan"))

    orig_skew = float(skew(s))

    # Yeo-Johnson
    try:
        yj = PowerTransformer(method="yeo-johnson", standardize=False)
        yj_vals = yj.fit_transform(s.values.reshape(-1, 1)).ravel()
        yj_skew = float(skew(yj_vals))
    except Exception:
        yj_vals, yj_skew = None, float("inf")

    # log1p (only for non-negative)
    if s.min() >= 0:
        try:
            log_vals = np.log1p(s.values)
            log_skew = float(skew(log_vals))
        except Exception:
            log_vals, log_skew = None, float("inf")
    else:
        log_vals, log_skew = None, float("inf")

    # 선택: 왜도 절대값이 더 작은 쪽
    candidates: Dict[str, Tuple[np.ndarray, float]] = {
        "yeo_johnson": (yj_vals, abs(yj_skew)),
        "log1p": (log_vals, abs(log_skew)),
    }
    best_name = min(
        [k for k, (v, sk) in candidates.items() if v is not None],
        key=lambda k: candidates[k][1],
        default="none",
    )

    if best_name == "yeo_johnson" and yj_vals is not None:
        trans_vals = yj_vals
        trans_skew = yj_skew
    elif best_name == "log1p" and log_vals is not None:
        trans_vals = log_vals
        trans_skew = log_skew
    else:
        return ("none", series, orig_skew, orig_skew)

    # NaN 복원
    out = pd.Series(index=series.index, dtype=float)
    out.loc[s.index] = trans_vals
    return (best_name, out, orig_skew, float(trans_skew))


def plot_original_and_transformed(
    col: str,
    s_orig: pd.Series,
    s_trans: pd.Series,
    out_dir: Path,
    transform_name: str,
) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats

    make_dirs(out_dir)

    # Histogram + KDE: original
    plt.figure(figsize=(7, 4))
    sns.histplot(s_orig.dropna(), bins=50, kde=True)
    plt.title(f"Distribution (orig) - {col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"dist_orig_{safe_name(col)}.png")
    plt.close()

    # Histogram + KDE: transformed
    plt.figure(figsize=(7, 4))
    sns.histplot(s_trans.dropna(), bins=50, kde=True)
    plt.title(f"Distribution ({transform_name}) - {col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"dist_trans_{safe_name(col)}.png")
    plt.close()

    # QQ-plot original
    plt.figure(figsize=(5, 5))
    stats.probplot(s_orig.dropna(), dist="norm", plot=plt)
    plt.title(f"QQ (orig) - {col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"qq_orig_{safe_name(col)}.png")
    plt.close()

    # QQ-plot transformed
    plt.figure(figsize=(5, 5))
    stats.probplot(s_trans.dropna(), dist="norm", plot=plt)
    plt.title(f"QQ ({transform_name}) - {col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"qq_trans_{safe_name(col)}.png")
    plt.close()


def run(input_csv: Path, out_dir: Path, skew_threshold: float = 1.0) -> None:
    print(f"[INFO] 입력: {input_csv}")
    print(f"[INFO] 출력: {out_dir}")
    configure_matplotlib_font()
    make_dirs(out_dir)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df = coerce_known_numeric(df)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 식별자 성격 컬럼 제외
    exclude_cols = {"건물번호"}
    num_cols = [c for c in num_cols if c not in exclude_cols]

    # 요약 보고서
    report_lines: List[str] = []
    report_lines.append("정규성 변환 보고서")
    report_lines.append("=")

    from scipy.stats import skew
    for col in num_cols:
        s = df[col]
        s_nonan = s.dropna()
        if s_nonan.empty:
            continue
        orig_sk = float(skew(s_nonan))

        if abs(orig_sk) < skew_threshold:
            # 변환 불필요 → 원본만 그래프 저장
            out_sub = out_dir / safe_name(col)
            plot_original_and_transformed(col, s, s, out_sub, "orig")
            report_lines.append(
                f"- {col}: 변환 생략 (|skew|={orig_sk:.3f} < {skew_threshold})"
            )
            continue

        # 변환 선택 및 적용
        name, s_trans, orig_skew, trans_skew = choose_transform(s)
        out_sub = out_dir / safe_name(col)
        plot_original_and_transformed(col, s, s_trans, out_sub, name)
        report_lines.append(
            f"- {col}: {name} 적용 (skew: {orig_skew:.3f} → {trans_skew:.3f})"
        )

    # 보고서 저장
    (out_dir / "transformation_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"[DONE] 결과 저장: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="왜도 완화 변환 및 시각화 스크립트")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "train_building_merged.csv"),
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eda/train_building_merged_transformed"),
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=1.0,
        help="변환을 적용할 왜도 임계값(|skew|>=임계값에서 변환)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input), Path(args.outdir), skew_threshold=args.skew_threshold)








