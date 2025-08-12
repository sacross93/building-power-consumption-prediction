import argparse
from pathlib import Path
from typing import List, Optional

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


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("%", "pct")
    )


TEST_FRIENDLY_CANDIDATES = [
    # 시간/주기
    "hour", "weekday", "month", "is_weekend",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    # 날씨 원천
    "기온(°C)", "습도(%)", "강수량(mm)", "풍속(m/s)",
    # 열지표/파생(원천으로 계산 가능)
    "THI", "AH_gm3",
    "CDD18", "CDD22", "CDD24", "CDD26", "CDD28", "CDD30",
    "HDD18", "CDH26_3", "CDH26_12", "CDH26_24",
]


def select_predictors(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame]) -> List[str]:
    num_train = set(df_train.select_dtypes(include=[np.number]).columns.tolist())
    allow = list(num_train.intersection(TEST_FRIENDLY_CANDIDATES))
    if df_test is not None:
        num_test = set(df_test.select_dtypes(include=[np.number]).columns.tolist())
        allow = [c for c in allow if c in num_test]
    # 상수 제거
    allow = [c for c in allow if df_train[c].nunique(dropna=True) > 1]
    return sorted(allow)


def binned_plot(df: pd.DataFrame, x: str, y: str, out_png: Path, bins: int = 20) -> None:
    import matplotlib.pyplot as plt
    configure_matplotlib_font()

    s = df[[x, y]].dropna()
    if s.empty:
        return
    q = np.linspace(0, 1, bins + 1)
    edges = s[x].quantile(q).values
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


def heatmap_hour_month(df: pd.DataFrame, target: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib_font()

    if not {"hour", "month", target}.issubset(df.columns):
        return
    s = df[["hour", "month", target]].dropna()
    if s.empty:
        return
    piv = s.pivot_table(index="month", columns="hour", values=target, aggfunc="mean")
    plt.figure(figsize=(9, 5.5))
    sns.heatmap(piv, cmap="YlOrRd")
    plt.title(f"Mean {target} by month×hour")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def quick_xgb_importance(df: pd.DataFrame, features: List[str], target: str, out_png: Path, task: str) -> None:
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier, XGBRegressor
    configure_matplotlib_font()

    X = df[features].copy()
    if task == "binary":
        y = (pd.to_numeric(df[target], errors="coerce").fillna(0.0) > 0).astype(int)
        model = XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=2025,
            n_jobs=0,
        )
    else:
        y = np.where(pd.to_numeric(df[target], errors="coerce").fillna(0.0) > 0,
                     np.log1p(np.maximum(pd.to_numeric(df[target], errors="coerce").fillna(0.0), 0.0)), 0.0)
        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=2025,
            n_jobs=0,
        )
    model.fit(X, y, verbose=False)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    order = np.argsort(imp)[::-1][:30]
    top_feats = [features[i] for i in order]
    top_vals = imp[order]

    plt.figure(figsize=(7.5, 8))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals[::-1])
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel("XGB feature importance")
    plt.title(f"Top importances for {target} ({task})")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def run(train_parquet: Path, out_dir: Path, test_parquet: Optional[Path] = None) -> None:
    print(f"[INFO] 입력(train): {train_parquet}")
    if test_parquet:
        print(f"[INFO] 입력(test) : {test_parquet}")
    print(f"[INFO] 출력: {out_dir}")
    make_dirs(out_dir)

    df_tr = pd.read_parquet(train_parquet)
    df_te = pd.read_parquet(test_parquet) if (test_parquet and test_parquet.exists()) else None

    # 대상 타깃들
    targets = [
        ("일사(MJ/m2)", "solar"),
        ("일조(hr)", "sunshine"),
    ]

    # 사용할 예측자(테스트와 교집합 기반)
    predictors = select_predictors(df_tr, df_te)
    (out_dir / "predictors_used.txt").write_text("\n".join(predictors), encoding="utf-8")

    for tgt, tag in targets:
        if tgt not in df_tr.columns:
            print(f"[WARN] train에 {tgt} 없음. 스킵")
            continue
        subdir = out_dir / tag
        make_dirs(subdir)

        # Heatmap (hour×month)
        heatmap_hour_month(df_tr, tgt, subdir / f"heatmap_{safe_name(tgt)}.png")

        # Binned mean plots against each predictor
        for x in predictors:
            binned_plot(df_tr, x, tgt, subdir / f"binned_{safe_name(tgt)}_vs_{safe_name(x)}.png")

        # Quick XGB importances (binary is_positive, regression log_pos)
        try:
            quick_xgb_importance(df_tr, predictors, tgt, subdir / f"xgb_importance_{tag}_binary.png", task="binary")
        except Exception:
            pass
        try:
            quick_xgb_importance(df_tr, predictors, tgt, subdir / f"xgb_importance_{tag}_reg.png", task="reg")
        except Exception:
            pass

    print(f"[DONE] 결과 저장: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="일사/일조의 시간·날씨 기반 신호 시각화 및 간이 중요도")
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache/train_engineered.parquet"),
        help="train_engineered.parquet 경로",
    )
    p.add_argument(
        "--test-parquet",
        type=str,
        default=None,
        help="test_engineered.parquet 경로(있으면 교집합 예측자만 사용)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eda/solar_proxy"),
        help="출력 디렉토리",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.train_parquet), Path(args.outdir), Path(args.test_parquet) if args.test_parquet else None)


