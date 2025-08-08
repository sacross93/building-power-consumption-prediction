#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
import json
import gc
from typing import List, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold

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


def train_eval_oof(
    train_df: pd.DataFrame,
    params: Dict,
    drop_rainfall: bool,
    seeds: List[int],
    gpus: List[int],
) -> float:
    """Return overall OOF SMAPE across types using per-type XGB with OHE."""
    target = "전력소비량(kWh)"
    types = train_df["건물유형"].astype(str).unique().tolist()
    oof_all = []
    y_all = []

    for t in types:
        tr_t = train_df[train_df["건물유형"].astype(str) == t].copy()
        # OHE building id
        tr_t["건물번호"] = tr_t["건물번호"].astype(str)
        tr_t = pd.get_dummies(tr_t, columns=["건물번호"])  # test는 OOF에 불필요

        # numeric features only
        num_cols = tr_t.select_dtypes(include=[np.number]).columns.tolist()
        fcols = [c for c in num_cols if c not in {target, "log_power"}]
        X = tr_t[fcols].to_numpy()
        y = tr_t[target].to_numpy()

        # seed ensemble OOF
        oof_t = np.zeros(X.shape[0])
        for si, seed in enumerate(seeds):
            kf = KFold(n_splits=7, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                params_fold = {**params, "random_state": seed}
                if gpus:
                    gpu_id = gpus[(fold - 1) % len(gpus)]
                    params_fold.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": int(gpu_id)})
                model = xgb.XGBRegressor(**params_fold)
                try:
                    cb = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=cb, verbose=False)
                except TypeError:
                    model.set_params(n_estimators=min(2000, model.get_params().get('n_estimators', 5000)))
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred_va = model.predict(X_va)
                oof_t[va_idx] += pred_va / len(seeds)
                gc.collect()

        oof_all.append(oof_t)
        y_all.append(y)

    oof_all = np.concatenate(oof_all)
    y_all = np.concatenate(y_all)
    return smape_np(y_all, oof_all)


def make_params_from_trial(trial: optuna.Trial, base: Dict) -> Dict:
    p = dict(base)
    p["max_depth"] = trial.suggest_int("max_depth", 6, 12)
    p["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 6)
    p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    p["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 5.0)
    p["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 1.0)
    p["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
    # n_estimators는 ES로 자동 조절, 상한만 넉넉히
    p["n_estimators"] = 6000
    return p


def main():
    ap = argparse.ArgumentParser(description="Optuna: per-type XGB optimization (method_06)")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--drop-rainfall", action="store_true")
    ap.add_argument("--gpus", type=str, default="")
    ap.add_argument("--opt-seeds", type=str, default="2025", help="seeds used in optimization phase (faster)")
    ap.add_argument("--seeds", type=str, default="2021,2023,2025,2027,2029", help="seeds for final training")
    ap.add_argument("--objective", type=str, default="reg:squarederror")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=0, help="seconds; 0 means no limit")
    ap.add_argument("--study-name", type=str, default="xgb_by_type_optuna")
    ap.add_argument("--guardrail", type=float, default=0.1)
    ap.add_argument("--sampler-seed", type=int, default=2025)
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    gpus = [int(x) for x in args.gpus.split(',')] if args.gpus.strip() else []
    opt_seeds = [int(x) for x in args.opt_seeds.split(',') if x.strip()]
    final_seeds = [int(x) for x in args.seeds.split(',') if x.strip()]

    # Base params
    base_params = dict(
        objective=args.objective,
        learning_rate=0.05,
        n_estimators=6000,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="mae",
        verbosity=0,
    )

    # Optimization objective
    def objective(trial: optuna.Trial) -> float:
        params = make_params_from_trial(trial, base_params)
        score = train_eval_oof(
            ensure_num_date_time(build_features(train_df, args.drop_rainfall)),
            params,
            args.drop_rainfall,
            opt_seeds,
            gpus,
        )
        return score  # minimize SMAPE

    study = optuna.create_study(direction="minimize", study_name=args.study_name, sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout if args.timeout > 0 else None, show_progress_bar=True)

    # Save study summary
    best_params = study.best_params
    best_value = study.best_value
    with (outdir / "optuna_best.json").open("w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_smape": best_value}, f, ensure_ascii=False, indent=2)
    df_trials = study.trials_dataframe()
    df_trials.to_csv(outdir / "optuna_trials.csv", index=False)

    # Final training with best params + full seeds, and guardrail blending
    final_params = make_params_from_trial(optuna.trial.FixedTrial(best_params), base_params)

    # Per-type prediction for submission
    target = "전력소비량(kWh)"
    train_df2 = ensure_num_date_time(train_df)
    test_df2 = ensure_num_date_time(test_df)
    types = train_df2["건물유형"].astype(str).unique().tolist()
    preds_test_list = []
    for t in types:
        tr_t = train_df2[train_df2["건물유형"].astype(str) == t].copy()
        te_t = test_df2[test_df2["건물유형"].astype(str) == t].copy()
        tr_t["건물번호"], te_t["건물번호"] = tr_t["건물번호"].astype(str), te_t["건물번호"].astype(str)
        tr_t = pd.get_dummies(tr_t, columns=["건물번호"]) ; te_t = pd.get_dummies(te_t, columns=["건물번호"]) ;
        tr_t, te_t = tr_t.align(te_t, join="left", axis=1, fill_value=0)
        num_cols = tr_t.select_dtypes(include=[np.number]).columns.tolist()
        fcols = [c for c in num_cols if c not in {target, "log_power"}]
        X = tr_t[fcols].to_numpy(); y = tr_t[target].to_numpy(); X_test = te_t[fcols].to_numpy()
        preds_acc = np.zeros((len(final_seeds), X_test.shape[0]))
        for si, seed in enumerate(final_seeds):
            kf = KFold(n_splits=7, shuffle=True, random_state=seed)
            preds_te = np.zeros(X_test.shape[0])
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                params_fold = {**final_params, "random_state": seed}
                if gpus:
                    gpu_id = gpus[(fold - 1) % len(gpus)]
                    params_fold.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": int(gpu_id)})
                model = xgb.XGBRegressor(**params_fold)
                try:
                    cb = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=cb, verbose=False)
                except TypeError:
                    model.set_params(n_estimators=min(2000, model.get_params().get('n_estimators', 5000)))
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                preds_te += model.predict(X_test) / kf.get_n_splits()
                gc.collect()
            preds_acc[si] = preds_te
        te_mean = preds_acc.mean(axis=0)
        preds_test_list.append(pd.DataFrame({"type": t, "num_date_time": te_t["num_date_time"], "pred": te_mean}))

    sub = pd.concat(preds_test_list, ignore_index=True)
    sub = sub[["num_date_time", "pred"]].groupby("num_date_time", as_index=False).mean()
    sub.rename(columns={"pred": "answer"}, inplace=True)
    # Guardrail blending with building_hour_weekday_mean if present
    if args.guardrail > 0 and {"num_date_time", "building_hour_weekday_mean"}.issubset(test_df2.columns):
        sub = sub.merge(test_df2[["num_date_time", "building_hour_weekday_mean"]], on="num_date_time", how="left")
        sub["answer"] = (1 - args.guardrail) * sub["answer"] + args.guardrail * sub["building_hour_weekday_mean"]
        sub.drop(columns=["building_hour_weekday_mean"], inplace=True)
    sub["answer"] = np.clip(sub["answer"], 0, None)

    # Align to sample_submission order
    sample_path = args.test.parents[2] / "data" / "sample_submission.csv"
    try:
        sample = pd.read_csv(sample_path)
        sub = sample[["num_date_time"]].merge(sub, on="num_date_time", how="left")
    except Exception:
        pass
    out_path = args.outdir / "submission_optuna_best.csv"
    sub.to_csv(out_path, index=False)
    print(f"✅ Optuna best SMAPE (OOF): {best_value:.3f}% | Submission: {out_path}")


if __name__ == "__main__":
    main()

