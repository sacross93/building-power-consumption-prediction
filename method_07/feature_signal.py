import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib_font() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager, rcParams

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


def choose_features(df: pd.DataFrame, target: str, drop_cols: Iterable[str]) -> List[str]:
    drop_set = set(drop_cols) | {target}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in drop_set]
    # 제거: 상수 컬럼
    features = [c for c in features if df[c].nunique(dropna=True) > 1]
    return features


def corr_top(df: pd.DataFrame, features: List[str], target: str, topk: int = 30) -> pd.DataFrame:
    sub = df[features + [target]].copy()
    corr = sub.corr(numeric_only=True)[target].drop(labels=[target]).sort_values(key=lambda s: s.abs(), ascending=False)
    out = pd.DataFrame({"feature": corr.index, "pearson_corr": corr.values})
    return out.head(topk)


def bin_plot(df: pd.DataFrame, x: str, y: str, out_png: Path, bins: int = 20) -> None:
    import matplotlib.pyplot as plt
    configure_matplotlib_font()

    s = df[[x, y]].dropna()
    if s.empty:
        return

    # 분위수 구간 평균
    q = np.linspace(0, 1, bins + 1)
    edges = s[x].quantile(q).values
    # 경계 중복 제거
    edges = np.unique(edges)
    if len(edges) < 3:
        return
    cats = pd.cut(s[x], bins=edges, include_lowest=True, duplicates="drop")
    g = s.groupby(cats)[y].mean().reset_index()
    centers = [(iv.left + iv.right) / 2 for iv in g[x]]

    plt.figure(figsize=(6.5, 4))
    plt.plot(centers, g[y].values, marker="o", lw=1.5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Binned mean of {y} vs {x}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def quick_xgb_importance(df: pd.DataFrame, features: List[str], target: str, out_png: Path, sample_rows: int = 150000, random_state: int = 2025) -> Tuple[object, List[str]]:
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor

    configure_matplotlib_font()

    X = df[features]
    y = df[target]

    if len(X) > sample_rows:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(len(X), size=sample_rows, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]

    # 단순 학습/검증 분할(EDA 용도)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_valid = X.iloc[:split], X.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.05,
        n_estimators=200,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=random_state,
        n_jobs=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:30]
    top_feats = [features[i] for i in order]
    top_vals = importances[order]

    plt.figure(figsize=(7.5, 8))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals[::-1])
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel("XGBoost feature importance")
    plt.title("Top feature importances (quick XGB)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    return model, top_feats


def run(input_parquet: Path, out_dir: Path, target_col: str = "전력소비량(kWh)") -> None:
    print(f"[INFO] 입력: {input_parquet}")
    print(f"[INFO] 출력 디렉토리: {out_dir}")
    make_dirs(out_dir)

    df = pd.read_parquet(input_parquet)

    # 타깃과 로그 타깃
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 없음: {target_col}")
    df["log_power"] = np.log1p(pd.to_numeric(df[target_col], errors="coerce")).clip(lower=0)

    # 피처 선택
    drop_cols = [
        "일시",
        "num_date_time",
        target_col,
        "건물번호",
    ]
    feats = choose_features(df, target="log_power", drop_cols=drop_cols)

    # 상관 상위 표
    corr_df = corr_top(df, feats, target="log_power", topk=30)
    corr_df.to_csv(out_dir / "corr_top30.csv", index=False, encoding="utf-8-sig")

    # 빠른 XGB 중요도
    model, top_feats = quick_xgb_importance(
        df, feats, target="log_power", out_png=out_dir / "xgb_feature_importance.png"
    )

    # 신호 곡선(대표 피처 및 중요 상위 일부)
    key_feats = [
        "THI",
        "AH_gm3",
        "CDD26",
        "CDH26_24",
        "일사(MJ/m2)",
        "일사(MJ/m2)_log_pos",
        "is_일사(MJ/m2)_positive",
        "일조(hr)",
        "일조(hr)_log_pos",
        "is_일조(hr)_positive",
        "강수량(mm)",
        "강수량(mm)_log_pos",
        "is_강수량(mm)_positive",
        "태양광용량(kW)_utility",
        "pv_utility_solar",
        "cooling_ratio",
        "연면적(m2)",
        "냉방면적(m2)",
    ]
    # 중요 상위와 합집합 후 존재하는 것만
    plot_feats = []
    for c in key_feats + top_feats:
        if c in df.columns and c not in plot_feats:
            plot_feats.append(c)

    # binned mean plot (log_power, 그리고 원스케일 power도 보조)
    for c in plot_feats[:40]:
        try:
            bin_plot(df, x=c, y="log_power", out_png=out_dir / f"binned_log_power_vs_{c}.png")
            bin_plot(df, x=c, y=target_col, out_png=out_dir / f"binned_power_vs_{c}.png")
        except Exception:
            continue

    # 요약 저장
    lines = [
        "특징 신호 요약",
        "=",
        f"샘플 수: {len(df):,}",
        f"사용 피처 수: {len(feats):,}",
        "상위 상관 피처(corr_top30.csv)와 XGB 중요도(xgb_feature_importance.png) 및 binned plot들을 확인하세요.",
    ]
    (out_dir / "feature_signal_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] 결과 저장: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="파생특징 신호 시각화(상관/중요도/빈 평균 곡선)")
    p.add_argument(
        "--input-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache/train_engineered.parquet"),
        help="입력 Parquet 경로(기본: method_07/cache/train_engineered.parquet)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eda/feature_signal"),
        help="출력 디렉토리(기본: method_07/eda/feature_signal)",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="전력소비량(kWh)",
        help="타깃 컬럼명(기본: 전력소비량(kWh))",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input_parquet), Path(args.outdir), target_col=args.target_col)







