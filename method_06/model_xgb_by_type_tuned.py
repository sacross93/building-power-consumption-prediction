#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
import gc
import json
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.clip(y_pred, 0, None)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)


def ensure_num_date_time(df: pd.DataFrame) -> pd.DataFrame:
    if "num_date_time" not in df.columns and {"건물번호", "일시"}.issubset(df.columns):
        df = df.copy()
        df["num_date_time"] = df["건물번호"].astype(str) + "_" + df["일시"].dt.strftime("%Y%m%d %H")
    return df


def build_features(df: pd.DataFrame, drop_rainfall: bool) -> pd.DataFrame:
    drop_cols = {"일시", "num_date_time", "log_power"}
    if drop_rainfall and "강수량(mm)" in df.columns:
        drop_cols.add("강수량(mm)")
    return df[[c for c in df.columns if c not in drop_cols]]


def train_one_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params_overrides: dict,
    gpus: list[int] | None,
    seeds: list[int],
    drop_rainfall: bool,
    guardrail_alpha: float,
):
    from sklearn.model_selection import KFold
    # 준비
    train_df = ensure_num_date_time(train_df)
    test_df = ensure_num_date_time(test_df)
    target = "전력소비량(kWh)"
    types = train_df["건물유형"].astype(str).unique().tolist()

    preds_test_list = []
    oof_list = []

    base_cols = build_features(train_df, drop_rainfall).columns.tolist()

    # 공통 기본 파라미터
    base_params = dict(
        objective="reg:squarederror",
        learning_rate=0.05,
        n_estimators=5000,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="mae",
        verbosity=0,
    )
    base_params.update(params_overrides or {})
    if gpus:
        base_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})

    for t in types:
        tr_t = train_df[train_df["건물유형"].astype(str) == t].copy()
        te_t = test_df[test_df["건물유형"].astype(str) == t].copy()

        # OHE 건물번호
        tr_t["건물번호"] = tr_t["건물번호"].astype(str)
        te_t["건물번호"] = te_t["건물번호"].astype(str)
        tr_t = pd.get_dummies(tr_t, columns=["건물번호"]) ; te_t = pd.get_dummies(te_t, columns=["건물번호"]) ;
        tr_t, te_t = tr_t.align(te_t, join="left", axis=1, fill_value=0)

        # 수치 피처만
        num_cols = tr_t.select_dtypes(include=[np.number]).columns.tolist()
        fcols = [c for c in num_cols if c not in {target, "log_power"}]
        X = tr_t[fcols].to_numpy(); y = tr_t[target].to_numpy(); X_test = te_t[fcols].to_numpy()

        # 앙상블
        oof = np.zeros(X.shape[0]); preds_te_acc = np.zeros((len(seeds), X_test.shape[0]))
        for si, seed in enumerate(seeds):
            kf = KFold(n_splits=7, shuffle=True, random_state=seed)
            preds_te = np.zeros(X_test.shape[0])
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
                X_tr, X_va = X[tr_idx], X[va_idx]; y_tr, y_va = y[tr_idx], y[va_idx]
                params_fold = {**base_params, "random_state": seed}
                if gpus:
                    gpu_id = gpus[(fold - 1) % len(gpus)]
                    params_fold.update({"gpu_id": int(gpu_id)})
                model = xgb.XGBRegressor(**params_fold)
                try:
                    cb = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=cb, verbose=False)
                except TypeError:
                    model.set_params(n_estimators=min(2000, model.get_params().get('n_estimators', 5000)))
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred_va = model.predict(X_va); pred_te = model.predict(X_test)
                oof[va_idx] += pred_va / len(seeds)
                preds_te += pred_te / kf.get_n_splits()
                gc.collect()
            preds_te_acc[si] = preds_te

        # seed 평균
        te_mean = preds_te_acc.mean(axis=0)
        oof_list.append(pd.DataFrame({"type": t, "y": y, "oof": oof}))
        preds_test_list.append(pd.DataFrame({"type": t, "num_date_time": te_t["num_date_time"], "pred": te_mean}))

    # 전체 OOF
    all_oof = pd.concat(oof_list, ignore_index=True)
    total_smape = smape_np(all_oof["y"].to_numpy(), all_oof["oof"].to_numpy())

    # 제출
    sub = pd.concat(preds_test_list, ignore_index=True)
    sub = sub[["num_date_time", "pred"]].groupby("num_date_time", as_index=False).mean()
    sub.rename(columns={"pred": "answer"}, inplace=True)

    # 가드레일 블렌딩(선택)
    if guardrail_alpha > 0 and {"num_date_time", "building_hour_weekday_mean"}.issubset(test_df.columns):
        sub = sub.merge(test_df[["num_date_time", "building_hour_weekday_mean"]], on="num_date_time", how="left")
        sub["answer"] = (1 - guardrail_alpha) * sub["answer"] + guardrail_alpha * sub["building_hour_weekday_mean"]
        sub.drop(columns=["building_hour_weekday_mean"], inplace=True)
    sub["answer"] = np.clip(sub["answer"], 0, None)
    return total_smape, sub


def main():
    p = argparse.ArgumentParser(description="XGB per-type tuning sweeps (method_06)")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True, help="output directory for submissions & logs")
    p.add_argument("--drop-rainfall", action="store_true")
    p.add_argument("--gpus", type=str, default="")
    p.add_argument("--seeds", type=str, default="2023,2025,2027")
    p.add_argument("--guardrails", type=str, default="0.0,0.1")
    p.add_argument("--depth", type=str, default="8")
    p.add_argument("--min-child-weight", type=str, default="3")
    p.add_argument("--subsample", type=str, default="0.8")
    p.add_argument("--colsample", type=str, default="0.8")
    p.add_argument("--objective", type=str, default="reg:squarederror")
    args = p.parse_args()

    outdir: Path = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    gpus = [int(x) for x in args.gpus.split(',')] if args.gpus.strip() else []
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    guard_list = [float(x) for x in args.guardrails.split(',') if x.strip()]
    depth_list = [int(x) for x in args.depth.split(',') if x.strip()]
    mcw_list = [int(x) for x in args.min_child_weight.split(',') if x.strip()]
    subsample_list = [float(x) for x in args.subsample.split(',') if x.strip()]
    colsample_list = [float(x) for x in args.colsample.split(',') if x.strip()]

    results = []
    best = {"smape": 1e9, "path": None}
    for d, mcw, ss, cs, ga in product(depth_list, mcw_list, subsample_list, colsample_list, guard_list):
        params = {
            "max_depth": d,
            "min_child_weight": mcw,
            "subsample": ss,
            "colsample_bytree": cs,
            "objective": args.objective,
        }
        smape, sub = train_one_setting(train_df, test_df, params, gpus, seeds, args.drop_rainfall, ga)
        tag = f"d{d}_mcw{mcw}_ss{ss}_cs{cs}_ga{ga}"
        sub_path = outdir / f"submission_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        results.append({"tag": tag, "smape": smape, "submission": str(sub_path)})
        if smape < best["smape"]:
            best.update({"smape": smape, "path": str(sub_path), "tag": tag})
        print(f"✔ {tag} OOF SMAPE={smape:.3f}% → {sub_path}")

    # 요약 저장
    with (outdir / "tuning_results.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"🏁 Best: {best}")


if __name__ == "__main__":
    main()

