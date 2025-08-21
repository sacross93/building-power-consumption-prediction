import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_predictors(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, drop_cols: List[str]) -> List[str]:
    num_train = set(df_train.select_dtypes(include=[np.number]).columns.tolist())
    num_test = set(df_test.select_dtypes(include=[np.number]).columns.tolist())
    candidates = list((num_train & num_test) - set(drop_cols) - {target})
    # 상수 제거
    predictors = [c for c in candidates if df_train[c].nunique(dropna=True) > 1]
    predictors.sort()
    return predictors


def train_regressor(X_tr: pd.DataFrame, y_tr: pd.Series, use_gpu: bool = False):
    from xgboost import XGBRegressor

    params: Dict = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 2025,
        "n_jobs": 0,
    }
    if use_gpu:
        params.update({"device": "cuda"})

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)
    return model


def run(train_parquet: Path, test_parquet: Path, out_parquet: Path, targets: List[str], use_gpu: bool) -> None:
    print(f"[INFO] train: {train_parquet}")
    print(f"[INFO] test : {test_parquet}")
    df_tr = pd.read_parquet(train_parquet)
    df_te = pd.read_parquet(test_parquet)

    # 드롭/금지 컬럼: 타깃, 식별자 등
    forbidden = [
        "전력소비량(kWh)",
        "log_power",
        "건물번호",
        "일시",
        "num_date_time",
    ]

    df_out = df_te.copy()
    trained: List[str] = []
    for tgt in targets:
        if tgt not in df_tr.columns:
            print(f"[WARN] train에 {tgt} 미존재. 스킵")
            continue
        # test에 존재하면 스킵(이미 있음)
        if tgt in df_te.columns:
            # 비어있으면 예측 보강
            if df_te[tgt].notna().any():
                print(f"[INFO] test에 {tgt} 존재 → 스킵")
                continue
        predictors = build_predictors(df_tr, df_te, target=tgt, drop_cols=forbidden)
        if not predictors:
            print(f"[WARN] {tgt} 예측용 공통 피처 없음. 스킵")
            continue
        X_tr = df_tr[predictors]
        y_tr = pd.to_numeric(df_tr[tgt], errors="coerce")
        notnan = y_tr.notna()
        X_tr, y_tr = X_tr[notnan], y_tr[notnan]
        if len(X_tr) == 0:
            print(f"[WARN] {tgt} 학습 데이터 없음. 스킵")
            continue
        print(f"[INFO] {tgt} 모델 학습: predictors={len(predictors)} rows={len(X_tr):,}")
        model = train_regressor(X_tr, y_tr, use_gpu=use_gpu)
        # test 예측
        X_te = df_te[predictors]
        y_hat = model.predict(X_te)
        df_out[tgt] = y_hat
        trained.append(tgt)

    # 파생 재계산(일사/일조 관련)
    def add_two_part(df: pd.DataFrame, base_col: str):
        if base_col in df.columns:
            s = pd.to_numeric(df[base_col], errors="coerce")
            df[f"is_{base_col}_positive"] = (s > 0).astype(int)
            df[f"{base_col}_log_pos"] = np.where(s > 0, np.log1p(np.clip(s, a_min=0, a_max=None)), 0.0)

    if "일사(MJ/m2)" in trained or "일사(MJ/m2)" in df_out.columns:
        add_two_part(df_out, "일사(MJ/m2)")
        # 상호작용(가능 시)
        if {"hour_sin", "hour_cos"}.issubset(df_out.columns):
            df_out["solar_hour_sin"] = pd.to_numeric(df_out["일사(MJ/m2)"], errors="coerce").fillna(0.0) * df_out["hour_sin"]
            df_out["solar_hour_cos"] = pd.to_numeric(df_out["일사(MJ/m2)"], errors="coerce").fillna(0.0) * df_out["hour_cos"]
        if "태양광용량(kW)_utility" in df_out.columns:
            df_out["pv_utility_solar"] = (
                pd.to_numeric(df_out["태양광용량(kW)_utility"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df_out["일사(MJ/m2)"], errors="coerce").fillna(0.0)
            )

    if "일조(hr)" in trained or "일조(hr)" in df_out.columns:
        add_two_part(df_out, "일조(hr)")

    make_dirs(out_parquet.parent)
    df_out.to_parquet(out_parquet, index=False)
    print(f"[DONE] 저장: {out_parquet} (imputed: {trained})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="test에서 누락된 중요 피처를 학습 모델로 보강")
    p.add_argument("--train-parquet", type=str, required=True, help="train_engineered.parquet 경로")
    p.add_argument("--test-parquet", type=str, required=True, help="test_engineered.parquet 경로")
    p.add_argument("--out-parquet", type=str, required=True, help="보강 후 저장 경로")
    p.add_argument(
        "--targets",
        type=str,
        default="일사(MJ/m2),일조(hr)",
        help="보강 대상 타깃들(콤마 구분)",
    )
    p.add_argument("--use-gpu", action="store_true", help="가능하면 GPU(CUDA) 사용")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        train_parquet=Path(args.train_parquet),
        test_parquet=Path(args.test_parquet),
        out_parquet=Path(args.out_parquet),
        targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        use_gpu=args.use_gpu,
    )







