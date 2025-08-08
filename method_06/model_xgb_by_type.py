#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
import gc
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
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
    # 전처리 산출물에서 사용할 코어 피처만 선별
    drop_cols = {"일시", "num_date_time", "log_power"}
    if drop_rainfall and "강수량(mm)" in df.columns:
        drop_cols.add("강수량(mm)")
    return df[[c for c in df.columns if c not in drop_cols]]


def train_xgb_by_type(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path, drop_rainfall: bool, gpus: list[int] | None) -> None:
    print("📦 데이터 준비...")
    train_df = ensure_num_date_time(train_df)
    test_df = ensure_num_date_time(test_df)

    # 피처/타깃 분리
    target = "전력소비량(kWh)"
    assert target in train_df.columns

    # 유형 목록
    if "건물유형" not in train_df.columns:
        raise ValueError("건물유형 컬럼이 필요합니다. 전처리를 확인하세요.")
    type_values = train_df["건물유형"].astype(str).unique().tolist()
    print(f"건물유형 {len(type_values)}종 학습: {type_values}")

    # 결과 컨테이너
    preds_test_list = []
    oof_list = []
    logs = {"types": []}

    # 공통 피처셋(유형별로 건물번호 OHE 적용)
    base_cols = build_features(train_df, drop_rainfall).columns.tolist()
    print(f"기본 피처 수: {len(base_cols)}")
    if gpus and len(gpus) > 0:
        print(f"🚀 GPU 사용: {gpus} (xgboost gpu_hist)")

    for t in type_values:
        print(f"\n🏷️ 유형 학습: {t}")
        tr_t = train_df[train_df["건물유형"].astype(str) == t].copy()
        te_t = test_df[test_df["건물유형"].astype(str) == t].copy()

        # 건물번호 OHE
        ohe_col = "건물번호"
        tr_t[ohe_col] = tr_t[ohe_col].astype(str)
        te_t[ohe_col] = te_t[ohe_col].astype(str)
        tr_t = pd.get_dummies(tr_t, columns=[ohe_col])
        te_t = pd.get_dummies(te_t, columns=[ohe_col])
        tr_t, te_t = tr_t.align(te_t, join="left", axis=1, fill_value=0)

        # 피처 선택(공통 + OHE 확장 반영) – 비수치 컬럼 제거 및 타깃/시간/키 제외
        num_cols = tr_t.select_dtypes(include=[np.number]).columns.tolist()
        fcols = [c for c in num_cols if c != target]
        X = tr_t[fcols].to_numpy()
        y = tr_t[target].to_numpy()
        X_test = te_t[fcols].to_numpy()

        print(f"유형 {t}: 학습행 {X.shape[0]}, 피처 {X.shape[1]}, 테스트행 {X_test.shape[0]}")

        kf = KFold(n_splits=7, shuffle=True, random_state=2025)
        oof = np.zeros(X.shape[0])
        preds_te = np.zeros(X_test.shape[0])
        fold_scores = []

        # XGB 설정(보수적)
        params = dict(
            objective="reg:squarederror",
            learning_rate=0.05,
            n_estimators=5000,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            tree_method="hist",
            random_state=2025,
            eval_metric="mae",
            verbosity=0,
        )
        # GPU 설정
        if gpus and len(gpus) > 0:
            params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # fold별 GPU 라운드로빈 배정
            fold_gpu = None
            if gpus and len(gpus) > 0:
                fold_gpu = gpus[(fold - 1) % len(gpus)]
                params_fold = {**params, "gpu_id": int(fold_gpu)}
            else:
                params_fold = params
            print(f"  ▶ Fold {fold}: GPU={fold_gpu if fold_gpu is not None else 'CPU'}")
            model = xgb.XGBRegressor(**params_fold)
            # 다양한 xgboost 버전 호환: callbacks 우선, 실패 시 ES 없이 학습
            try:
                cb = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=cb,
                    verbose=False,
                )
            except TypeError:
                # 콜백 미지원 버전: ES 없이 축소 n_estimators로 학습
                model.set_params(n_estimators=min(2000, model.get_params().get('n_estimators', 5000)))
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )

            pred_va = model.predict(X_va)
            pred_te = model.predict(X_test)
            oof[va_idx] = pred_va
            preds_te += pred_te / kf.get_n_splits()

            score = smape_np(y_va, pred_va)
            fold_scores.append(score)

            # 피처 중요도 TOP20 로그
            try:
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-20:][::-1]
                top_feats = [(fcols[i], float(importances[i])) for i in top_idx]
                print(f"Fold {fold} SMAPE={score:.3f}, Top20: {top_feats}")
            except Exception:
                print(f"Fold {fold} SMAPE={score:.3f}")

            gc.collect()

        type_oof = smape_np(y, oof)
        print(f"유형 {t} OOF SMAPE={type_oof:.3f} | folds={np.round(fold_scores,3)}")
        logs["types"].append({"type": t, "oof": float(type_oof), "folds": [float(s) for s in fold_scores]})

        # 결과 수집
        oof_list.append(pd.DataFrame({"type": t, "y": y, "oof": oof}))
        preds_test_list.append(pd.DataFrame({"type": t, "num_date_time": te_t["num_date_time"], "pred": preds_te}))

    # 전체 OOF
    all_oof = pd.concat(oof_list, ignore_index=True)
    total_smape = smape_np(all_oof["y"].to_numpy(), all_oof["oof"].to_numpy())
    print(f"✅ Total OOF SMAPE: {total_smape:.3f}%")

    # 제출 생성(샘플 순서 강제)
    sub = pd.concat(preds_test_list, ignore_index=True)
    sub = sub[["num_date_time", "pred"]].groupby("num_date_time", as_index=False).mean()
    sub.rename(columns={"pred": "answer"}, inplace=True)

    # sample 순서 강제
    sample_path = Path(test_path).parents[1] / "data" / "sample_submission.csv"
    try:
        sample = pd.read_csv(sample_path)
        sub = sample[["num_date_time"]].merge(sub, on="num_date_time", how="left")
    except Exception:
        pass
    sub["answer"] = np.clip(sub["answer"], 0, None)
    sub.to_csv(output_path, index=False)
    print(f"🎉 Saved submission to {output_path}")

    # 로그 저장
    log_path = output_path.with_suffix(".log.json")
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"total_oof": float(total_smape), **logs}, f, ensure_ascii=False, indent=2)


def main(train_path: Path, test_path: Path, submission_path: Path, drop_rainfall: bool, gpus: list[int] | None) -> None:
    print("📥 캐시 로드...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_xgb_by_type(train_df, test_df, submission_path, drop_rainfall, gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost by building type (method_06)")
    parser.add_argument("--train", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/cache/train_preprocessed.parquet"))
    parser.add_argument("--test", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/cache/test_preprocessed.parquet"))
    parser.add_argument("--sub", type=Path, default=Path("/home/wlsdud022/ds_test/method_06/submission_xgb_by_type.csv"))
    parser.add_argument("--drop-rainfall", action="store_true")
    parser.add_argument("--gpus", type=str, default="", help="comma-separated GPU ids to use (e.g., '2,3')")
    args = parser.parse_args()

    # test_path needed inside train_xgb_by_type for sample path
    global test_path
    test_path = args.test
    gpus = [int(x) for x in args.gpus.split(',')] if args.gpus.strip() else []
    main(args.train, args.test, args.sub, args.drop_rainfall, gpus)

