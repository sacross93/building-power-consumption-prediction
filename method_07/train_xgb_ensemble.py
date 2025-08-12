import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    # 0/0 방지
    mask = denom > 0
    v = np.zeros_like(denom)
    v[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 200.0 * np.mean(v)  # %


def build_features(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = {
        "일시",
        "num_date_time",
        "건물번호",
        "건물유형",
        target_col,
        "log_power",
    }
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in drop_cols]
    # 상수 제거
    features = [c for c in features if df[c].nunique(dropna=True) > 1]
    return features


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict] = None,
):
    from xgboost import XGBRegressor

    if params is None:
        params = {}
    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=params.get("learning_rate", 0.05),
        n_estimators=params.get("n_estimators", 600),
        max_depth=params.get("max_depth", 8),
        min_child_weight=params.get("min_child_weight", 3),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.5),
        reg_alpha=params.get("reg_alpha", 0.0),
        tree_method=params.get("tree_method", "hist"),
        device=params.get("device", None),
        random_state=params.get("random_state", 2025),
        n_jobs=params.get("n_jobs", 0),
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def inverse_and_clip(y_log_pred: np.ndarray) -> np.ndarray:
    y = np.expm1(y_log_pred)
    y[y < 0] = 0.0
    return y


def choose_time_order(df: pd.DataFrame) -> np.ndarray:
    # 우선순위: num_date_time(숫자 추출) → 일시(datetime) → 원래 인덱스
    if "num_date_time" in df.columns:
        s = df["num_date_time"].astype(str)
        # 뒤의 날짜/시간 패턴 추출: YYYYMMDD HH
        extracted = s.str.extract(r"(\d{8}\s\d{2})")[0]
        t = pd.to_datetime(extracted, format="%Y%m%d %H", errors="coerce")
        if t.notna().any():
            return t.values
    if "일시" in df.columns:
        t = pd.to_datetime(df["일시"], errors="coerce")
        if t.notna().any():
            return t.values
    # fallback: 정수 인덱스
    return np.arange(len(df))


def run(
    input_parquet: Path,
    out_dir: Path,
    sample_rows: int = 0,
    alpha_type_blend: float = 0.5,
    gamma_guardrail: float = 0.1,
    target_col: str = "전력소비량(kWh)",
    use_gpu: bool = False,
    sample_submission: Optional[Path] = None,
    test_parquet: Optional[Path] = None,
):
    print(f"[INFO] 입력: {input_parquet}")
    print(f"[INFO] 출력: {out_dir}")
    make_dirs(out_dir)

    df = pd.read_parquet(input_parquet)
    # 샘플링(옵션)
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=2025).sort_index()

    # 타깃/로그 타깃
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 없음: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    y_log = np.log1p(y)

    # 피처 선택
    features = build_features(df, target_col=target_col)
    X = df[features].copy()

    # 시간 순서 분할(80/20)
    order_key = choose_time_order(df)
    order = np.argsort(pd.Series(order_key).values)
    n = len(df)
    split = int(n * 0.8)
    idx_train = order[:split]
    idx_valid = order[split:]

    X_train, X_valid = X.iloc[idx_train], X.iloc[idx_valid]
    y_train, y_valid = y_log.iloc[idx_train], y_log.iloc[idx_valid]

    # 전역 모델 학습
    print("[INFO] 전역 모델 학습…")
    xgb_params: Dict = {}
    if use_gpu:
        # XGBoost 2.x 권장 설정: device="cuda", tree_method="hist"
        xgb_params.update({"device": "cuda", "tree_method": "hist"})
    global_model = train_xgb(X_train, y_train, X_valid, y_valid, params=xgb_params)

    # 예측: DMatrix 경로 사용으로 device mismatch 경고 방지
    import xgboost as xgb
    dvalid = xgb.DMatrix(X_valid, feature_names=X_valid.columns.tolist())
    y_log_pred_g = global_model.get_booster().predict(dvalid)
    y_pred_g = inverse_and_clip(y_log_pred_g)

    # 유형별 모델 학습(건물유형 컬럼 기반)
    has_type = "건물유형" in df.columns
    y_pred_t = np.zeros_like(y_pred_g)
    type_models: Dict[str, object] = {}
    if has_type:
        print("[INFO] 건물유형별 모델 학습…")
        types = df.iloc[idx_train]["건물유형"].astype(str)
        min_rows = 1000  # 너무 작은 그룹은 스킵
        for t in types.unique():
            tr_mask = (df.iloc[idx_train]["건물유형"].astype(str) == t).values
            va_mask = (df.iloc[idx_valid]["건물유형"].astype(str) == t).values
            if tr_mask.sum() < min_rows or va_mask.sum() == 0:
                continue
            m = train_xgb(
                X_train[tr_mask], y_train[tr_mask], X_valid[va_mask], y_valid[va_mask], params=xgb_params
            )
            type_models[t] = m
            # 부분 검증셋도 DMatrix로 예측
            dvalid_t = xgb.DMatrix(X_valid[va_mask], feature_names=X_valid.columns.tolist())
            y_log_pred_t = m.get_booster().predict(dvalid_t)
            y_pred_t[va_mask] = inverse_and_clip(y_log_pred_t)
    else:
        print("[WARN] '건물유형' 컬럼이 없어 유형별 모델을 건너뜁니다.")

    # 유형 예측이 비어있는 곳은 전역으로 대체
    fallback_mask = y_pred_t <= 0
    y_pred_t[fallback_mask] = y_pred_g[fallback_mask]

    # 블렌딩
    y_pred_blend = (1 - alpha_type_blend) * y_pred_g + alpha_type_blend * y_pred_t

    # 가드레일 블렌딩(베이스라인 존재 시)
    if "building_hour_weekday_mean" in df.columns:
        base_valid = pd.to_numeric(
            df.iloc[idx_valid]["building_hour_weekday_mean"], errors="coerce"
        ).fillna(0.0)
        y_pred_blend = (1 - gamma_guardrail) * y_pred_blend + gamma_guardrail * base_valid.values

    # 검증 성능
    y_true_valid = y.iloc[idx_valid].values
    smape_g = smape(y_true_valid, y_pred_g)
    smape_t = smape(y_true_valid, y_pred_t)
    smape_b = smape(y_true_valid, y_pred_blend)

    print(f"[METRIC] SMAPE Global: {smape_g:.3f}% | Type: {smape_t:.3f}% | Blended: {smape_b:.3f}%")

    # 저장물
    (out_dir / "used_features.txt").write_text("\n".join(features), encoding="utf-8")
    pd.DataFrame(
        {
            "y_true": y_true_valid,
            "y_pred_global": y_pred_g,
            "y_pred_type": y_pred_t,
            "y_pred_blend": y_pred_blend,
        }
    ).to_csv(out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    # 간단 중요도 저장(전역 모델)
    try:
        imp = getattr(global_model, "feature_importances_", None)
        if imp is not None:
            pd.DataFrame({"feature": features, "importance": imp}).sort_values(
                "importance", ascending=False
            ).to_csv(out_dir / "global_feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

    # 요약 메모
    summary = [
        f"SMAPE Global: {smape_g:.3f}%",
        f"SMAPE Type: {smape_t:.3f}%",
        f"SMAPE Blended: {smape_b:.3f}%",
        f"Rows: {len(df):,} | Train: {len(idx_train):,} | Valid: {len(idx_valid):,}",
        f"Features: {len(features):,}",
        f"Alpha(type blend): {alpha_type_blend}",
        f"Gamma(guardrail): {gamma_guardrail}",
        f"Type models: {len(type_models)}",
    ]
    (out_dir / "training_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"[DONE] 저장: {out_dir}")

    # ===== Optional: sample_submission 포맷으로 예측 파일 생성 =====
    def ensure_num_date_time_key(frame: pd.DataFrame) -> Optional[pd.Series]:
        if ("건물번호" in frame.columns) and ("일시" in frame.columns):
            ts = pd.to_datetime(frame["일시"], errors="coerce")
            key = frame["건물번호"].astype(str) + "_" + ts.dt.strftime("%Y%m%d %H")
            return key
        # 대체 키가 없으면 None
        return None

    def predict_for_df(model_g, model_types: Dict[str, object], df_any: pd.DataFrame) -> np.ndarray:
        X_any = df_any[features].copy()
        import xgboost as xgb
        dmat = xgb.DMatrix(X_any, feature_names=X_any.columns.tolist())
        y_log_pred_any_g = model_g.get_booster().predict(dmat)
        y_pred_any_g = inverse_and_clip(y_log_pred_any_g)
        y_pred_any_t = np.zeros_like(y_pred_any_g)
        if has_type and len(model_types) > 0:
            types_any = df_any["건물유형"].astype(str) if "건물유형" in df_any.columns else pd.Series([""] * len(df_any))
            for t, m in model_types.items():
                mask = (types_any == t).values
                if not mask.any():
                    continue
                dmat_t = xgb.DMatrix(X_any[mask], feature_names=X_any.columns.tolist())
                y_log_pred_any_t = m.get_booster().predict(dmat_t)
                y_pred_any_t[mask] = inverse_and_clip(y_log_pred_any_t)
            # fallback
            fb = y_pred_any_t <= 0
            y_pred_any_t[fb] = y_pred_any_g[fb]
        else:
            y_pred_any_t = y_pred_any_g.copy()
        y_pred_any_blend = (1 - alpha_type_blend) * y_pred_any_g + alpha_type_blend * y_pred_any_t
        if "building_hour_weekday_mean" in df_any.columns:
            base_any = pd.to_numeric(df_any["building_hour_weekday_mean"], errors="coerce").fillna(0.0).values
            y_pred_any_blend = (1 - gamma_guardrail) * y_pred_any_blend + gamma_guardrail * base_any
        return y_pred_any_blend

    if sample_submission is not None:
        try:
            ss = pd.read_csv(sample_submission, encoding="utf-8-sig")
            # 우선순위: test_parquet가 있으면 그 데이터로 예측, 없으면 valid 구간으로 근사
            if test_parquet is not None and Path(test_parquet).exists():
                df_test = pd.read_parquet(test_parquet)
                # 피처 세트 정합성 확인
                missing = [c for c in features if c not in df_test.columns]
                if missing:
                    print(f"[WARN] test에 누락 피처 존재: {missing[:5]}... {len(missing)}개")
                y_pred_test = predict_for_df(global_model, type_models, df_test)
                key_test = ensure_num_date_time_key(df_test)
                if key_test is not None and ss.columns[0] in ss.columns:
                    sub = pd.DataFrame({ss.columns[0]: key_test, ss.columns[1]: y_pred_test})
                else:
                    # 키가 없으면 순서대로 채움
                    sub = pd.DataFrame({ss.columns[1]: y_pred_test})
                    sub.insert(0, ss.columns[0], range(len(sub)))
            else:
                # valid 구간으로 근사 생성
                df_valid = df.iloc[idx_valid].copy()
                y_pred_valid = y_pred_blend
                key_valid = ensure_num_date_time_key(df_valid)
                if key_valid is not None:
                    sub = pd.DataFrame({ss.columns[0]: key_valid, ss.columns[1]: y_pred_valid})
                else:
                    sub = pd.DataFrame({ss.columns[1]: y_pred_valid})
                    sub.insert(0, ss.columns[0], range(len(sub)))

            # sample_submission의 순서로 정렬(가능하면)
            if ss.columns[0] in sub.columns:
                sub = ss[[ss.columns[0]]].merge(sub, on=ss.columns[0], how="left")
                # 결측은 0으로 보수적 대체
                sub[ss.columns[1]] = sub[ss.columns[1]].fillna(0.0)
            # 저장
            out_path = out_dir / "submission_like.csv"
            sub.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[DONE] 샘플 제출 포맷 저장: {out_path}")
        except Exception as e:
            print(f"[WARN] sample_submission 생성 실패: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="전역+유형별 XGBoost 앙상블 학습 스크립트")
    p.add_argument(
        "--input-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache/train_engineered.parquet"),
        help="입력 Parquet 경로",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="출력 디렉토리",
    )
    p.add_argument("--sample-rows", type=int, default=0, help="샘플링 행 수(0=전체)")
    p.add_argument("--alpha-type-blend", type=float, default=0.5, help="전역/유형 가중(유형 비중)")
    p.add_argument("--gamma-guardrail", type=float, default=0.1, help="가드레일 베이스라인 가중")
    p.add_argument("--target-col", type=str, default="전력소비량(kWh)")
    p.add_argument("--use-gpu", action="store_true", help="가능하면 GPU(CUDA) 사용")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_parquet=Path(args.input_parquet),
        out_dir=Path(args.outdir),
        sample_rows=args.sample_rows,
        alpha_type_blend=args.alpha_type_blend,
        gamma_guardrail=args.gamma_guardrail,
        target_col=args.target_col,
        use_gpu=args.use_gpu,
    )


