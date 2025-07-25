import argparse
from pathlib import Path
import warnings
import gc
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import ElasticNetCV

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

SEED = 42

############################################################
# 평가 지표 – SMAPE (competition metric)
############################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.maximum(y_pred, 0)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

############################################################
# 학습 함수 (fold 단위)
############################################################

def train_fold(X_tr, y_tr, X_val, y_val, categorical_features: List[str]):
    """각 모델을 학습하고 validation 예측 반환"""

    ##### LightGBM #####
    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "random_state": SEED,
        "learning_rate": 0.05,
        "num_leaves": 256,
        "max_depth": -1,
        "n_estimators": 8000,
        "device": "gpu",
        "gpu_use_dp": True,
        "max_bin": 255,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )
    pred_lgb = lgb_model.predict(X_val)

    ##### XGBoost #####
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        random_state=SEED,
        learning_rate=0.05,
        max_depth=8,
        n_estimators=8000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        verbosity=0,
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        callbacks=[xgb.callback.EarlyStopping(rounds=300)]
    )
    pred_xgb = xgb_model.predict(X_val)

    ##### CatBoost #####
    cat_features_idx = [X_tr.columns.get_loc(col) for col in categorical_features]
    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,
        random_seed=SEED,
        task_type="GPU",
        devices="0",
        verbose=False,
    )
    cat_model.fit(
        Pool(X_tr, y_tr, cat_features=cat_features_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_features_idx),
        verbose=False,
        early_stopping_rounds=300,
    )
    pred_cat = cat_model.predict(X_val)

    # return models and predictions
    return (lgb_model, xgb_model, cat_model), np.vstack([pred_lgb, pred_xgb, pred_cat]).T

############################################################
# 메인 스크립트
############################################################

def main(train_path: Path, test_path: Path, out_path: Path):
    print("📦 데이터 로드...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 타겟 & 피처 분리
    target_col = "log_power"
    drop_cols = ["전력소비량(kWh)", "일시", "num_date_time"]  # 원본 타겟·시간·ID 열 제외
    feature_cols = [c for c in train_df.columns if c not in drop_cols + [target_col]]

    # 카테고리 피처 자동 감지 (dtype == 'category')
    categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]
    print(f"Categorical features: {categorical_cols}")

    X = train_df[feature_cols]
    y = train_df[target_col]

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_preds = np.zeros((len(train_df), 3))  # 3 base models
    test_preds = np.zeros((len(test_df), 3))
    base_models_per_fold = []

    fold = 0
    for tr_idx, val_idx in tscv.split(X):
        fold += 1
        print(f"\n🚀 Fold {fold}/{n_splits}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        models, pred_val = train_fold(X_tr, y_tr, X_val, y_val, categorical_cols)
        oof_preds[val_idx, :] = pred_val

        # 테스트 예측 (평균)
        fold_test_pred = np.column_stack([m.predict(test_df[feature_cols]) for m in models])
        test_preds += fold_test_pred / n_splits

        base_models_per_fold.append(models)
        gc.collect()

        smape_lgb = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 0]))
        smape_xgb = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 1]))
        smape_cat = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 2]))
        print(f"   SMAPE – LGB {smape_lgb:.3f} | XGB {smape_xgb:.3f} | CAT {smape_cat:.3f}")

    # Meta model (ElasticNet)
    print("\n🔗 스태킹 메타 모델 학습 (ElasticNetCV)...")
    enet = ElasticNetCV(l1_ratio=[0.0, 0.5, 1.0], cv=3, random_state=SEED)
    enet.fit(oof_preds, y)
    oof_meta = enet.predict(oof_preds)
    score_meta = smape_np(np.expm1(y), np.expm1(oof_meta))
    print(f"✅ Meta SMAPE: {score_meta:.3f}%")

    # Test meta predictions
    test_meta = enet.predict(test_preds)
    final_pred_kwh = np.expm1(test_meta) * test_df["연면적(m2)"]

    # 제출 파일 생성
    submission = pd.DataFrame({
        "num_date_time": test_df["num_date_time"],
        "answer": final_pred_kwh.clip(lower=0)
    })
    submission.to_csv(out_path, index=False)
    print(f"🎉 Submission saved to {out_path}")

############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stacking Model Trainer (method_04)")
    parser.add_argument("--train", type=Path, default=Path("method_04/cache/train_preprocessed.parquet"))
    parser.add_argument("--test", type=Path, default=Path("method_04/cache/test_preprocessed.parquet"))
    parser.add_argument("--sub", type=Path, default=Path("submission_stack.csv"))
    args = parser.parse_args()

    main(args.train, args.test, args.sub) 