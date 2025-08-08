#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
import gc
from typing import List, Dict
import os
import shutil

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

SEED = 42


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.maximum(y_pred, 0)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100


def encode_categories(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    df_enc = df.copy()
    for c in cat_cols:
        if str(df_enc[c].dtype) == "category":
            df_enc[c] = df_enc[c].cat.codes.astype("int32")
        else:
            df_enc[c] = df_enc[c].astype("int32")
    return df_enc


def check_gpu_availability() -> Dict[str, str]:
    gpu = {
        "lgb_device": "cpu",
        "xgb_tree_method": "hist",
        "cat_task_type": "CPU",
        "available": False,
    }
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    if has_nvidia_smi:
        gpu.update({
            "lgb_device": "gpu",
            "xgb_tree_method": "gpu_hist",
            "cat_task_type": "GPU",
            "available": True,
        })
    return gpu


def train_fold(X_tr, y_tr, X_val, y_val, categorical_features: List[str], gpu_info: Dict[str, str]):
    models: Dict[str, object] = {}
    preds: Dict[str, np.ndarray] = {}

    print("  ▶ LightGBM 학습 시작 ...")
    try:
        lgb_params = {
            "objective": "regression",
            "metric": "l1",
            "random_state": SEED,
            "learning_rate": 0.05,
            "num_leaves": 256,
            "max_depth": -1,
            "n_estimators": 6000,
            "verbose": -1,
        }
        # LightGBM GPU 시도
        if gpu_info["lgb_device"] == "gpu":
            lgb_params.update({"device_type": "gpu"})
        else:
            lgb_params.update({"device": "cpu"})

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        models["lgb"] = lgb_model
        preds["lgb"] = lgb_model.predict(X_val)
        print("    ✅ LightGBM 완료 (device:", ("gpu" if gpu_info["lgb_device"] == "gpu" else "cpu"), ")")
    except Exception as e:
        print("    ❌ LightGBM GPU 실패, CPU로 재시도:", str(e))
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="l1",
            random_state=SEED,
            learning_rate=0.05,
            num_leaves=256,
            max_depth=-1,
            n_estimators=6000,
            device="cpu",
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        models["lgb"] = lgb_model
        preds["lgb"] = lgb_model.predict(X_val)
        print("    ✅ LightGBM CPU 완료")

    # XGBoost
    X_tr_enc = encode_categories(X_tr, categorical_features)
    X_val_enc = encode_categories(X_val, categorical_features)
    print("  ▶ XGBoost 학습 시작 ...")
    try:
        xgb_params = {
            "objective": "reg:squarederror",
            "tree_method": gpu_info["xgb_tree_method"],
            "random_state": SEED,
            "learning_rate": 0.05,
            "max_depth": 8,
            "n_estimators": 6000,
            "early_stopping_rounds": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "verbosity": 0,
        }
        if gpu_info["xgb_tree_method"] == "gpu_hist":
            xgb_params.update({"predictor": "gpu_predictor", "gpu_id": 0})
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
        models["xgb"] = xgb_model
        preds["xgb"] = xgb_model.predict(X_val_enc)
        print("    ✅ XGBoost 완료 (tree_method:", gpu_info["xgb_tree_method"], ")")
    except Exception as e:
        print("    ❌ XGBoost GPU 실패, CPU로 재시도:", str(e))
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            learning_rate=0.05,
            max_depth=8,
            n_estimators=6000,
            early_stopping_rounds=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            verbosity=0,
        )
        xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
        models["xgb"] = xgb_model
        preds["xgb"] = xgb_model.predict(X_val_enc)
        print("    ✅ XGBoost CPU 완료")

    # CatBoost
    cat_features_idx = [X_tr.columns.get_loc(col) for col in categorical_features]
    print("  ▶ CatBoost 학습 시작 ...")
    try:
        cat_params = {
            "loss_function": "MAE",
            "iterations": 6000,
            "early_stopping_rounds": 300,
            "learning_rate": 0.05,
            "depth": 8,
            "random_seed": SEED,
            "task_type": gpu_info["cat_task_type"],
            "verbose": False,
        }
        if gpu_info["cat_task_type"] == "GPU":
            cat_params.update({"devices": "0"})
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features_idx),
            eval_set=Pool(X_val, y_val, cat_features=cat_features_idx),
            verbose=False,
        )
        models["cat"] = cat_model
        preds["cat"] = cat_model.predict(X_val)
        print("    ✅ CatBoost 완료 (task_type:", gpu_info["cat_task_type"], ")")
    except Exception as e:
        print("    ❌ CatBoost GPU 실패, CPU로 재시도:", str(e))
        cat_model = CatBoostRegressor(
            loss_function="MAE",
            iterations=6000,
            early_stopping_rounds=300,
            learning_rate=0.05,
            depth=8,
            random_seed=SEED,
            task_type="CPU",
            verbose=False,
        )
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features_idx),
            eval_set=Pool(X_val, y_val, cat_features=cat_features_idx),
            verbose=False,
        )
        models["cat"] = cat_model
        preds["cat"] = cat_model.predict(X_val)
        print("    ✅ CatBoost CPU 완료")

    return models, np.column_stack([preds["lgb"], preds["xgb"], preds["cat"]])


def main(train_path: Path, test_path: Path, submission_path: Path):
    print("📦 데이터 로드...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print("🔍 GPU 환경 체크 중...")
    gpu_info = check_gpu_availability()
    if gpu_info["available"]:
        print("🎯 GPU 가속 모드 활성화!",
              f"LGBM={gpu_info['lgb_device']}",
              f"XGB={gpu_info['xgb_tree_method']}",
              f"CAT={gpu_info['cat_task_type']}")
    else:
        print("⚠️ CPU 모드로 실행")

    # 타깃 및 피처
    if "log_power" in train_df.columns:
        target_col = "log_power"
    else:
        raise ValueError("log_power가 필요합니다. 전처리를 먼저 실행하세요.")

    drop_cols = ["전력소비량(kWh)", "일시", "num_date_time"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols + [target_col]]

    # test에 num_date_time 보장
    if "num_date_time" not in test_df.columns:
        if "일시" in test_df.columns and "건물번호" in test_df.columns:
            test_df["num_date_time"] = (
                test_df["건물번호"].astype(str) + "_" + test_df["일시"].dt.strftime("%Y%m%d %H")
            )
        else:
            raise ValueError("test 데이터에 num_date_time 생성 불가")

    # 전역 시간 정렬 보장 (시계열 분할 일관성)
    if "일시" in train_df.columns:
        train_df = train_df.sort_values(["일시"]).reset_index(drop=True)
    if "일시" in test_df.columns:
        test_df = test_df.sort_values(["일시"]).reset_index(drop=True)

    # 카테고리 열 자동 탐지
    categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]
    if "건물번호" in feature_cols and "건물번호" not in categorical_cols:
        categorical_cols.append("건물번호")
    print(f"Categorical features: {categorical_cols}")

    X = train_df[feature_cols]
    y = train_df[target_col]

    print("🚀 5-Fold TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(y), 3))
    test_preds = np.zeros((len(test_df), 3))

    n_splits = 5
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n🚀 Fold {fold+1}/{n_splits}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        print(f"  ▶ Train rows: {len(y_tr)}, Val rows: {len(y_val)}")
        models, pred_val = train_fold(X_tr, y_tr, X_val, y_val, categorical_cols, gpu_info)
        oof_preds[val_idx, :] = pred_val

        # test 예측
        test_df_enc = encode_categories(test_df[feature_cols], categorical_cols)
        fold_test_pred = np.column_stack([
            models["lgb"].predict(test_df[feature_cols]),
            models["xgb"].predict(test_df_enc),
            models["cat"].predict(test_df[feature_cols]),
        ])
        test_preds += fold_test_pred / n_splits
        gc.collect()

    print("\n🔗 메타 모델 학습 (LinearRegression)...")
    meta = LinearRegression()
    meta.fit(oof_preds, y)
    oof_meta = meta.predict(oof_preds)
    score_meta_log = smape_np(y, oof_meta)
    print(f"✅ Meta SMAPE (log-space): {score_meta_log:.3f}%")

    # 원공간 OOF SMAPE 함께 출력
    try:
        y_linear = np.expm1(y)
        oof_meta_linear = np.expm1(oof_meta)
        score_meta_linear = smape_np(y_linear, oof_meta_linear)
        print(f"✅ Meta SMAPE (linear-space): {score_meta_linear:.3f}%")
    except Exception as e:
        print("⚠️ 원공간 OOF 계산 실패:", str(e))

    test_meta = meta.predict(test_preds)
    final_pred_kwh = np.expm1(test_meta)

    submission = pd.DataFrame({
        "num_date_time": test_df["num_date_time"],
        "answer": np.clip(final_pred_kwh, 0, None),
    })

    # sample_submission 순서 강제 정렬 (집합 동일 + 순서 동일 보장)
    try:
        sample_path = Path(test_path).parents[1] / "data" / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            order = sample[["num_date_time"]].merge(submission, on="num_date_time", how="left")
            if order["answer"].isna().any():
                print("⚠️ sample_submission과 키 불일치 행 존재")
            submission = order
    except Exception as e:
        print("⚠️ sample_submission 순서 정렬 실패:", str(e))
    submission.to_csv(submission_path, index=False)
    print(f"🎉 Submission saved to {submission_path}")


def run():
    parser = argparse.ArgumentParser(description="Stacking model (method_06)")
    parser.add_argument("--train", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/cache/train_preprocessed.parquet"))
    parser.add_argument("--test", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/cache/test_preprocessed.parquet"))
    parser.add_argument("--sub", type=Path, default=Path("/home/wlsdud022/ds_test/submission_method06.csv"))
    args = parser.parse_args()
    main(args.train, args.test, args.sub)


if __name__ == "__main__":
    run()

