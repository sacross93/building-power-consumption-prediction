import argparse
from pathlib import Path
import warnings
import gc
from typing import List
import os

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
# GPU 상태 확인 및 최적화 함수
############################################################

def check_gpu_availability():
    """GPU 사용 가능 여부와 최적 설정을 확인"""
    gpu_info = {
        "lightgbm_device": "cpu",
        "xgb_tree_method": "hist", 
        "catboost_task_type": "CPU",
        "gpu_available": False
    }
    
    # CUDA 체크 (nvidia-ml-py 대신 직접 체크)
    cuda_available = False
    try:
        # CUDA 라이브러리 경로 체크
        cuda_paths = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
        for path in cuda_paths:
            if os.path.exists(f"{path}/libcuda.so") or os.path.exists(f"{path}/libcuda.so.1"):
                cuda_available = True
                break
        
        if cuda_available:
            print("🎯 CUDA 라이브러리 발견")
            gpu_info["xgb_tree_method"] = "gpu_hist"
            gpu_info["catboost_task_type"] = "GPU"
            gpu_info["gpu_available"] = True
            
            # LightGBM은 OpenCL 백엔드로 시도 (CUDA 빌드 문제 회피)
            gpu_info["lightgbm_device"] = "gpu"  # OpenCL 백엔드 우선 시도
        else:
            print("⚠️ CUDA 라이브러리 없음")
    except Exception as e:
        print(f"⚠️ CUDA 체크 실패: {e}")
    
    # OpenCL 체크
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            devices = []
            for platform in platforms:
                try:
                    platform_devices = platform.get_devices(cl.device_type.GPU)
                    devices.extend(platform_devices)
                except:
                    pass
            
            if devices:
                print(f"🎯 OpenCL GPU 발견: {len(devices)}개 디바이스")
                gpu_info["lightgbm_device"] = "gpu"
                gpu_info["gpu_available"] = True
            else:
                print("⚠️ OpenCL GPU 디바이스 없음")
    except ImportError:
        print("⚠️ pyopencl 없음 - OpenCL GPU 사용 불가")
    except Exception as e:
        print(f"⚠️ OpenCL 체크 실패: {e}")
    
    return gpu_info

############################################################
# Helper: 카테고리형 컬럼을 XGBoost 입력용으로 int 코드 변환
############################################################

def encode_categories(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """category dtype -> int 코드, NaN은 -1"""
    df_enc = df.copy()
    for c in cat_cols:
        if str(df_enc[c].dtype) == "category":
            # cat.codes에서 -1(missing)를 그대로 유지
            df_enc[c] = df_enc[c].cat.codes.astype("int32")
        else:
            # 이미 숫자형인 경우 (건물번호 등) 그대로 유지하되 int32로 변환
            df_enc[c] = df_enc[c].astype("int32")
    return df_enc

############################################################
# 평가 지표 – SMAPE (competition metric)
############################################################

def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.maximum(y_pred, 0)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100

############################################################
# 개선된 학습 함수
############################################################

def train_fold(X_tr, y_tr, X_val, y_val, categorical_features, gpu_info):
    models = {}
    predictions = {}
    # 디버그: 학습/검증 데이터 크기 로그
    print(f"    ▶️ Train rows: {len(y_tr)}, Val rows: {len(y_val)}")
    
    # 1. LightGBM - 일단 CPU로 강제 설정 (GPU 학습 문제 때문)
    print("  🚀 LightGBM 학습...")
    try:
        lgb_params = {
            "objective": "regression",
            "metric": "l1",
            "random_state": SEED,
            "learning_rate": 0.05,
            "num_leaves": 256,
            "max_depth": -1,
            "n_estimators": 8000,
            "device": "cpu",  # 강제 CPU 모드
            "verbose": -1,
        }
        
        # CPU 모드이므로 추가 파라미터 불필요
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
        predictions["lgb"] = lgb_model.predict(X_val)
        print("    ✅ LightGBM 완료 (device: cpu)")
        # 디버그: best iteration 및 예측 분산 확인
        best_iter = getattr(lgb_model, "best_iteration_", lgb_model.n_estimators_)
        pred_std = np.std(predictions["lgb"])
        print(f"       ⮑ best_iter={best_iter}, val_pred_std={pred_std:.4f}")
        if pred_std < 1e-6:
            print("       ⚠️ LightGBM 예측이 상수에 가깝습니다!")
        
    except Exception as e:
        print(f"    ❌ LightGBM GPU 실패, CPU로 재시도: {e}")
        # CPU 백업
        lgb_params_cpu = {
            "objective": "regression",
            "metric": "l1",
            "random_state": SEED,
            "learning_rate": 0.05,
            "num_leaves": 256,
            "max_depth": -1,
            "n_estimators": 8000,
            "device": "cpu",
            "verbose": -1,
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params_cpu)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae", 
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        models["lgb"] = lgb_model
        predictions["lgb"] = lgb_model.predict(X_val)
        print("    ✅ LightGBM CPU 완료")

    # 2. XGBoost with improved GPU settings  
    print("  🚀 XGBoost 학습...")

    # 카테고리 → int 코드 변환 (GPU categorical 미지원 대비)
    X_tr_enc = encode_categories(X_tr, categorical_features)
    X_val_enc = encode_categories(X_val, categorical_features)

    try:
        xgb_params = {
            "objective": "reg:squarederror",
            "tree_method": gpu_info["xgb_tree_method"],
            "random_state": SEED,
            "learning_rate": 0.05,
            "max_depth": 8,
            "n_estimators": 8000,
            "early_stopping_rounds": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "verbosity": 1,  # GPU 사용 여부 확인용
        }
        
        if gpu_info["xgb_tree_method"] == "gpu_hist":
            xgb_params.update({
                "predictor": "gpu_predictor",
                "gpu_id": 0,
            })
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
        
        models["xgb"] = xgb_model
        predictions["xgb"] = xgb_model.predict(X_val_enc)
        print(f"    ✅ XGBoost 완료 (method: {gpu_info['xgb_tree_method']})")
        # 디버그: best iteration 및 예측 분산
        best_iter_xgb = getattr(xgb_model, "best_iteration", None)
        pred_std_xgb = np.std(predictions["xgb"])
        print(f"       ⮑ best_iter={best_iter_xgb}, val_pred_std={pred_std_xgb:.4f}")
        if pred_std_xgb < 1e-6:
            print("       ⚠️ XGBoost 예측이 상수에 가깝습니다!")
        
    except Exception as e:
        print(f"    ❌ XGBoost GPU 실패, CPU로 재시도: {e}")
        # CPU 백업
        xgb_params["tree_method"] = "hist"
        xgb_params.pop("predictor", None)
        xgb_params.pop("gpu_id", None)
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
        models["xgb"] = xgb_model
        predictions["xgb"] = xgb_model.predict(X_val_enc)
        print("    ✅ XGBoost CPU 완료")

    # 3. CatBoost with improved GPU settings
    print("  🚀 CatBoost 학습...")
    try:
        cat_features_idx = [X_tr.columns.get_loc(col) for col in categorical_features]
        cat_params = {
            "loss_function": "MAE",
            "iterations": 8000,
            "early_stopping_rounds": 300,
            "learning_rate": 0.05,
            "depth": 8,
            "random_seed": SEED,
            "task_type": gpu_info["catboost_task_type"],
            "verbose": 100,  # GPU 사용 여부 확인용
        }
        
        if gpu_info["catboost_task_type"] == "GPU":
            cat_params["devices"] = "0"
        
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features_idx),
            eval_set=Pool(X_val, y_val, cat_features=cat_features_idx),
            verbose=False
        )
        
        models["cat"] = cat_model
        predictions["cat"] = cat_model.predict(X_val)
        print(f"    ✅ CatBoost 완료 (task_type: {gpu_info['catboost_task_type']})")
        # 디버그: best iteration 및 예측 분산
        best_iter_cat = cat_model.get_best_iteration()
        pred_std_cat = np.std(predictions["cat"])
        print(f"       ⮑ best_iter={best_iter_cat}, val_pred_std={pred_std_cat:.4f}")
        if pred_std_cat < 1e-6:
            print("       ⚠️ CatBoost 예측이 상수에 가깝습니다!")
        
    except Exception as e:
        print(f"    ❌ CatBoost GPU 실패, CPU로 재시도: {e}")
        # CPU 백업
        cat_params["task_type"] = "CPU"
        cat_params.pop("devices", None)
        
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features_idx),
            eval_set=Pool(X_val, y_val, cat_features=cat_features_idx),
            verbose=False
        )
        models["cat"] = cat_model
        predictions["cat"] = cat_model.predict(X_val)
        print("    ✅ CatBoost CPU 완료")

    return models, np.column_stack([predictions["lgb"], predictions["xgb"], predictions["cat"]])

############################################################
# 메인 스크립트
############################################################

def main(train_path: str, test_path: str, submission_path: str):
    print("🔍 GPU 환경 체크 중...")
    gpu_info = check_gpu_availability()
    
    if gpu_info["gpu_available"]:
        print("🎯 GPU 가속 모드 활성화!")
        print(f"  - LightGBM: {gpu_info['lightgbm_device']}")
        print(f"  - XGBoost: {gpu_info['xgb_tree_method']}")  
        print(f"  - CatBoost: {gpu_info['catboost_task_type']}")
    else:
        print("⚠️ CPU 모드로 실행 (GPU 미사용)")
    print()
    
    print("📦 데이터 로드...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # test 데이터에 num_date_time이 없으면 생성
    if "num_date_time" not in test_df.columns:
        if "일시" in test_df.columns and "건물번호" in test_df.columns:
            test_df["num_date_time"] = (
                test_df["건물번호"].astype(str) + "_" + 
                test_df["일시"].dt.strftime("%Y%m%d %H")
            )
        else:
            raise ValueError("test 데이터에 num_date_time 컬럼이 없고 일시/건물번호로 생성할 수 없습니다.")

    # 타겟 & 피처 분리
    # 전처리에서 생성한 로그 변환 타겟 컬럼명 확인
    if "log_power" in train_df.columns:
        target_col = "log_power"
    elif "log_전력소비량(kWh)" in train_df.columns:
        target_col = "log_전력소비량(kWh)"
    else:
        raise ValueError("로그 변환된 타겟 컬럼을 찾을 수 없습니다. 전처리를 확인하세요.")
    drop_cols = ["전력소비량(kWh)", "일시", "num_date_time"]  # 원본 타겟·시간·ID 열 제외
    feature_cols = [c for c in train_df.columns if c not in drop_cols + [target_col]]

    # 카테고리 피처 자동 감지 (dtype == 'category')
    categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]
    # 건물번호는 필수 범주형 피처로 추가 (전처리에서 category로 설정)
    if "건물번호" in feature_cols and "건물번호" not in categorical_cols:
        categorical_cols.append("건물번호")
    print(f"Categorical features: {categorical_cols}")

    X = train_df[feature_cols]
    y = train_df[target_col]

    print("🚀 5-Fold 교차검증...")
    tscv = TimeSeriesSplit(n_splits=5, test_size=None)
    oof_preds = np.zeros((len(y), 3))  # LGB, XGB, Cat 예측값
    test_preds = np.zeros((len(test_df), 3))
    base_models_per_fold = []  # 각 fold의 모델들을 저장
    
    n_splits = 5
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"🚀 Fold {fold+1}/{n_splits}")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        models, pred_val = train_fold(X_tr, y_tr, X_val, y_val, categorical_cols, gpu_info)
        oof_preds[val_idx, :] = pred_val

        # 테스트 예측 (평균)
        # LightGBM & CatBoost: 원본 test_df 사용, XGBoost: 인코딩 필요
        test_df_enc = encode_categories(test_df[feature_cols], categorical_cols)
        preds_list = []
        for name, model in models.items():
            if name == "xgb":
                preds_list.append(model.predict(test_df_enc))
            else:
                preds_list.append(model.predict(test_df[feature_cols]))
        fold_test_pred = np.column_stack(preds_list)
        test_preds += fold_test_pred / n_splits

        base_models_per_fold.append(models)
        gc.collect()

        smape_lgb = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 0]))
        smape_xgb = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 1]))
        smape_cat = smape_np(np.expm1(y_val), np.expm1(pred_val[:, 2]))
        print(f"   SMAPE – LGB {smape_lgb:.3f} | XGB {smape_xgb:.3f} | CAT {smape_cat:.3f}")

    # Meta model (ElasticNet)
    print("\n🔗 스태킹 메타 모델 학습 (ElasticNetCV)...")
    enet = ElasticNetCV(l1_ratio=[0.01, 0.5, 1.0], cv=3, random_state=SEED)
    enet.fit(oof_preds, y)
    oof_meta = enet.predict(oof_preds)
    score_meta = smape_np(np.expm1(y), np.expm1(oof_meta))
    print(f"✅ Meta SMAPE: {score_meta:.3f}%")

    # Test meta predictions
    test_meta = enet.predict(test_preds)
    # log_power = log1p(전력소비량)이므로 expm1로 역변환하면 됨
    # 연면적을 곱할 필요 없음 (이미 전력소비량 자체를 예측)
    final_pred_kwh = np.expm1(test_meta)

    # 제출 파일 생성
    submission = pd.DataFrame({
        "num_date_time": test_df["num_date_time"],
        "answer": np.clip(final_pred_kwh, 0, None)
    })
    submission.to_csv(submission_path, index=False)
    print(f"🎉 Submission saved to {submission_path}")

############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stacking Model Trainer (method_04)")
    parser.add_argument("--train", type=Path, default=Path("method_04/cache/train_preprocessed.parquet"))
    parser.add_argument("--test", type=Path, default=Path("method_04/cache/test_preprocessed.parquet"))
    parser.add_argument("--sub", type=Path, default=Path("submission_stack.csv"))
    args = parser.parse_args()

    main(args.train, args.test, args.sub) 