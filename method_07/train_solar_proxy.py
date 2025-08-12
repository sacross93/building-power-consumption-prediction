import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TEST_FRIENDLY_CANDIDATES: List[str] = [
    # 시간/주기(테스트에서도 생성 가능)
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    # 날씨 원천
    "기온(°C)",
    "습도(%)",
    "강수량(mm)",
    "풍속(m/s)",
    # 열지표/파생(원천으로 계산 가능)
    "THI",
    "AH_gm3",
    "CDD18",
    "CDD22",
    "CDD24",
    "CDD26",
    "CDD28",
    "CDD30",
    "HDD18",
    "CDH26_3",
    "CDH26_12",
    "CDH26_24",
]


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_time_order(df: pd.DataFrame) -> np.ndarray:
    if "num_date_time" in df.columns:
        s = df["num_date_time"].astype(str)
        extracted = s.str.extract(r"(\d{8}\s\d{2})")[0]
        t = pd.to_datetime(extracted, format="%Y%m%d %H", errors="coerce")
        if t.notna().any():
            return t.values
    if "일시" in df.columns:
        t = pd.to_datetime(df["일시"], errors="coerce")
        if t.notna().any():
            return t.values
    return np.arange(len(df))


def select_predictors(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame]) -> List[str]:
    num_train = set(df_train.select_dtypes(include=[np.number]).columns.tolist())
    allow = list(num_train.intersection(TEST_FRIENDLY_CANDIDATES))
    if df_test is not None:
        num_test = set(df_test.select_dtypes(include=[np.number]).columns.tolist())
        allow = [c for c in allow if c in num_test]
    # 상수 제거
    allow = [c for c in allow if df_train[c].nunique(dropna=True) > 1]
    return sorted(allow)


def train_binary(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series, use_gpu: bool) -> Tuple[object, Dict[str, float], float]:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from xgboost import XGBClassifier

    # 불균형 보정
    pos = float(y_tr.mean())
    scale_pos_weight = (1 - pos) / max(pos, 1e-6)

    params: Dict = dict(
        objective="binary:logistic",
        learning_rate=0.05,
        n_estimators=600,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=2025,
        n_jobs=0,
        scale_pos_weight=scale_pos_weight,
    )
    if use_gpu:
        params.update(device="cuda")

    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    prob = model.predict_proba(X_va)[:, 1]
    # 임계값 튜닝: F1 최대화
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y_va, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pred = (prob >= best_thr).astype(int)
    metrics = dict(
        auc=roc_auc_score(y_va, prob),
        f1=f1_score(y_va, pred),
        acc=accuracy_score(y_va, pred),
    )
    return model, metrics, best_thr


def train_regression(
    X_tr: pd.DataFrame, y_tr_log: pd.Series, X_va: pd.DataFrame, y_va_log: pd.Series, use_gpu: bool
) -> Tuple[object, Dict[str, float]]:
    from sklearn.metrics import r2_score, mean_absolute_error
    from xgboost import XGBRegressor

    params: Dict = dict(
        objective="reg:squarederror",
        learning_rate=0.05,
        n_estimators=800,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=2025,
        n_jobs=0,
    )
    if use_gpu:
        params.update(device="cuda")

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)

    yhat_log = model.predict(X_va)
    # 지표: 로그 스페이스 + 원스케일 보조
    r2 = r2_score(y_va_log, yhat_log)
    mae_log = mean_absolute_error(y_va_log, yhat_log)
    metrics = dict(r2=r2, mae_log=mae_log)
    return model, metrics


def run(
    train_parquet: Path,
    out_dir: Path,
    test_parquet: Optional[Path] = None,
    use_gpu: bool = False,
) -> None:
    print(f"[INFO] train: {train_parquet}")
    if test_parquet:
        print(f"[INFO] test : {test_parquet}")
    print(f"[INFO] out  : {out_dir}")
    make_dirs(out_dir)

    df_tr = pd.read_parquet(train_parquet)
    df_te = pd.read_parquet(test_parquet) if (test_parquet and test_parquet.exists()) else None

    predictors = select_predictors(df_tr, df_te)
    (out_dir / "predictors_used.txt").write_text("\n".join(predictors), encoding="utf-8")

    # 시간 순서 분할(80/20)
    order_key = choose_time_order(df_tr)
    order = np.argsort(pd.Series(order_key).values)
    n = len(df_tr)
    split = int(n * 0.8)
    idx_tr, idx_va = order[:split], order[split:]

    def fit_one(target_col: str, tag: str) -> None:
        print(f"\n[INFO] Target: {target_col}")
        if target_col not in df_tr.columns:
            print(f"[WARN] train에 {target_col} 없음. 스킵")
            return
        y = pd.to_numeric(df_tr[target_col], errors="coerce").fillna(0.0)

        # 1) Binary is_positive
        y_bin = (y > 0).astype(int)
        Xtr_bin, Xva_bin = df_tr.loc[idx_tr, predictors], df_tr.loc[idx_va, predictors]
        ytr_bin, yva_bin = y_bin.loc[idx_tr], y_bin.loc[idx_va]
        clf, m_bin, thr = train_binary(Xtr_bin, ytr_bin, Xva_bin, yva_bin, use_gpu)

        # 2) Regression on positive subset (log1p)
        mask_tr_pos = (y.loc[idx_tr] > 0).values
        mask_va_pos = (y.loc[idx_va] > 0).values
        Xtr_reg = df_tr.loc[idx_tr, predictors][mask_tr_pos]
        ytr_reg_log = np.log1p(y.loc[idx_tr][mask_tr_pos].values)
        Xva_reg = df_tr.loc[idx_va, predictors][mask_va_pos]
        yva_reg_log = np.log1p(y.loc[idx_va][mask_va_pos].values)
        reg, m_reg = train_regression(Xtr_reg, ytr_reg_log, Xva_reg, yva_reg_log, use_gpu)

        # 저장: 모델/지표
        clf.get_booster().save_model(str(out_dir / f"{tag}_binary.xgb.json"))
        reg.get_booster().save_model(str(out_dir / f"{tag}_reg.xgb.json"))
        lines = [
            f"predictors={len(predictors)}",
            f"bin_auc={m_bin['auc']:.4f}",
            f"bin_f1={m_bin['f1']:.4f}",
            f"bin_acc={m_bin['acc']:.4f}",
            f"bin_thr={thr:.3f}",
            f"reg_r2={m_reg['r2']:.4f}",
            f"reg_mae_log={m_reg['mae_log']:.4f}",
        ]
        (out_dir / f"{tag}_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

        # 테스트 예측(있을 때만)
        if df_te is not None:
            Xte = df_te[predictors].copy()
            prob = clf.predict_proba(Xte)[:, 1]
            is_pos = (prob >= thr).astype(int)
            yhat_log = reg.predict(Xte)
            yhat_raw = np.expm1(np.maximum(yhat_log, 0.0))
            # 분포 안정화: 음수 하한, 상위 캡(훈련 양수의 99.5%)
            cap = float(np.quantile(np.expm1(ytr_reg_log), 0.995)) if len(ytr_reg_log) else None
            if np.isfinite(cap):
                yhat_raw = np.clip(yhat_raw, 0.0, cap)
            yhat_final = np.where(is_pos > 0, yhat_raw, 0.0)

            # 결과 저장 CSV
            out_cols = pd.DataFrame(
                {
                    f"is_{target_col}_prob": prob,
                    f"is_{target_col}_pred": is_pos,
                    f"{target_col}_log_pos_pred": yhat_log,
                    f"{target_col}_pred": yhat_final,
                }
            )
            out_csv = out_dir / f"{tag}_test_preds.csv"
            out_cols.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[INFO] test 예측 저장: {out_csv}")

    # 학습 타깃들
    fit_one("일사(MJ/m2)", "solar")
    fit_one("일조(hr)", "sunshine")

    print(f"\n[DONE] 저장: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="일사/일조 보조모델(XGBoost, GPU 옵션) 학습 및 검증/예측")
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache/train_engineered.parquet"),
        help="train_engineered.parquet 경로",
    )
    p.add_argument("--test-parquet", type=str, default=None, help="test_engineered.parquet 경로(선택)")
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "out/solar_proxy"))
    p.add_argument("--use-gpu", action="store_true", help="가능하면 GPU(CUDA) 사용")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        train_parquet=Path(args.train_parquet),
        out_dir=Path(args.outdir),
        test_parquet=Path(args.test_parquet) if args.test_parquet else None,
        use_gpu=args.use_gpu,
    )


