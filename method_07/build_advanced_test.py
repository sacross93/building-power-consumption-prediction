import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TEST_FRIENDLY_CANDIDATES: List[str] = [
    # 시간/주기
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


def select_predictors(df_test: pd.DataFrame) -> List[str]:
    num_test = set(df_test.select_dtypes(include=[np.number]).columns.tolist())
    allow = [c for c in TEST_FRIENDLY_CANDIDATES if c in num_test]
    # 상수 제거
    allow = [c for c in allow if df_test[c].nunique(dropna=True) > 1]
    return allow


def load_thr_from_metrics(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    text = metrics_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("bin_thr="):
            try:
                return float(line.split("=", 1)[1])
            except Exception:
                return 0.5
    return 0.5


def compute_baseline(train_df: pd.DataFrame, target_col: str = "전력소비량(kWh)") -> pd.DataFrame:
    if not {"건물번호", "hour", "weekday", target_col}.issubset(train_df.columns):
        return pd.DataFrame(columns=["건물번호", "hour", "weekday", "building_hour_weekday_mean"])
    grp = (
        train_df.groupby(["건물번호", "hour", "weekday"], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "building_hour_weekday_mean"})
    )
    return grp


def predict_proxy(
    df_test: pd.DataFrame,
    predictors: List[str],
    model_dir: Path,
    tag: str,
    train_df: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """보조모델(이진+회귀)로 예측하여 (prob, log_pred, final_pred) 반환.
    final_pred는 0 하한 + train 양수 99.5% 캡 + 이진 마스크 적용.
    """
    import xgboost as xgb

    bin_path = model_dir / f"{tag}_binary.xgb.json"
    reg_path = model_dir / f"{tag}_reg.xgb.json"
    thr_path = model_dir / f"{tag}_metrics.txt"
    if not bin_path.exists() or not reg_path.exists():
        raise FileNotFoundError(f"보조모델 파일이 없습니다: {bin_path}, {reg_path}")

    clf = xgb.Booster()
    clf.load_model(str(bin_path))
    reg = xgb.Booster()
    reg.load_model(str(reg_path))
    thr = load_thr_from_metrics(thr_path)

    Xte = df_test[predictors].copy()
    dte = xgb.DMatrix(Xte, feature_names=Xte.columns.tolist())

    # 확률(로지스틱 출력)
    prob = clf.predict(dte)
    prob = np.clip(prob, 0.0, 1.0)
    is_pos = (prob >= thr).astype(int)

    # 로그 스페이스 회귀 → 원스케일 변환
    yhat_log = reg.predict(dte)
    yhat_log = np.maximum(yhat_log, 0.0)
    yhat_raw = np.expm1(yhat_log)

    # 상한 캡(훈련 양수 99.5%)
    y_train_pos = pd.to_numeric(train_df[target_col], errors="coerce")
    y_train_pos = y_train_pos[y_train_pos > 0]
    if len(y_train_pos) > 0:
        cap = float(np.quantile(y_train_pos, 0.995))
        yhat_raw = np.clip(yhat_raw, 0.0, cap)
    else:
        yhat_raw = np.clip(yhat_raw, 0.0, None)

    final = np.where(is_pos > 0, yhat_raw, 0.0)
    return prob, yhat_log, final


def run(
    train_parquet: Path,
    test_parquet: Path,
    solar_model_dir: Path,
    out_csv: Path,
) -> None:
    print(f"[INFO] train: {train_parquet}")
    print(f"[INFO] test : {test_parquet}")
    print(f"[INFO] models: {solar_model_dir}")
    print(f"[INFO] out  : {out_csv}")
    make_dirs(out_csv.parent)

    df_tr = pd.read_parquet(train_parquet)
    df_te = pd.read_parquet(test_parquet)

    # 베이스라인 병합(건물×hour×weekday)
    base = compute_baseline(df_tr, target_col="전력소비량(kWh)")
    if not base.empty and {"건물번호", "hour", "weekday"}.issubset(df_te.columns):
        df_te = df_te.merge(base, on=["건물번호", "hour", "weekday"], how="left")
        df_te["building_hour_weekday_mean"] = pd.to_numeric(
            df_te["building_hour_weekday_mean"], errors="coerce"
        ).fillna(0.0)
    else:
        df_te["building_hour_weekday_mean"] = 0.0

    # 예측자 선택(테스트 기준)
    predictors = select_predictors(df_te)
    if not predictors:
        raise ValueError("예측자 후보가 없습니다. test 엔지니어드 파케를 확인하세요.")

    # 일사 예측
    prob_solar, log_solar, solar_pred = predict_proxy(
        df_te, predictors, solar_model_dir, tag="solar", train_df=df_tr, target_col="일사(MJ/m2)"
    )
    df_te["is_일사(MJ/m2)_prob"] = prob_solar
    df_te["is_일사(MJ/m2)_pred"] = (prob_solar >= 0.5).astype(int)  # 저장 표준화(임계는 metrics 참고)
    df_te["일사(MJ/m2)_log_pos_pred"] = log_solar
    df_te["일사(MJ/m2)_pred"] = solar_pred

    # 일조 예측
    prob_sun, log_sun, sun_pred = predict_proxy(
        df_te, predictors, solar_model_dir, tag="sunshine", train_df=df_tr, target_col="일조(hr)"
    )
    df_te["is_일조(hr)_prob"] = prob_sun
    df_te["is_일조(hr)_pred"] = (prob_sun >= 0.5).astype(int)
    df_te["일조(hr)_log_pos_pred"] = log_sun
    df_te["일조(hr)_pred"] = sun_pred

    # 파생: 시간 상호작용, PV 상호작용
    if "hour_sin" in df_te.columns:
        df_te["solar_hour_sin_pred"] = pd.to_numeric(df_te["일사(MJ/m2)_pred"], errors="coerce").fillna(0.0) * df_te[
            "hour_sin"
        ]
        df_te["solar_hour_cos_pred"] = pd.to_numeric(df_te["일사(MJ/m2)_pred"], errors="coerce").fillna(0.0) * df_te[
            "hour_cos"
        ]
    if "태양광용량(kW)_utility" in df_te.columns:
        df_te["pv_utility_solar_pred"] = (
            pd.to_numeric(df_te["태양광용량(kW)_utility"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df_te["일사(MJ/m2)_pred"], errors="coerce").fillna(0.0)
        )

    # 타깃 의존 파생 제거(안전)
    for c in ["load_dev", "load_ratio", "전력소비량(kWh)", "log_power"]:
        if c in df_te.columns:
            df_te.drop(columns=[c], inplace=True)

    # 저장
    df_te.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] 저장: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="advanced_test.csv 생성 – 보조 일사/일조 예측 + 베이스라인 병합")
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache/train_engineered.parquet"),
    )
    p.add_argument(
        "--test-parquet",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache_test/test_engineered.parquet"),
    )
    p.add_argument(
        "--solar-model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out/solar_proxy"),
        help="train_solar_proxy.py가 저장한 모델/지표 디렉토리",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "advanced_test.csv"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        train_parquet=Path(args.train_parquet),
        test_parquet=Path(args.test_parquet),
        solar_model_dir=Path(args.solar_model_dir),
        out_csv=Path(args.out_csv),
    )


