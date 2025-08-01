# -*- coding: utf-8 -*-
"""
Ultimate Tuning Solution (v2‑full)
=================================
완전 실행 가능한 파이프라인. `python ultimate_tuning_solution.py` 만으로 학습 → 검증 → 예측 CSV 생성.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor, StackingRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0. Utility
# -----------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-9)) * 100


class TimeSeriesCV:
    """Forward‑chaining splitter with robust window sizing."""

    def __init__(self, n_splits: int = 3, test_size_days: int = 7, gap_days: int = 0):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days

    def split(self, df: pd.DataFrame, date_col: str):
        df = df.sort_values(date_col).reset_index(drop=True)
        first, last = df[date_col].min(), df[date_col].max()
        span = (last - first).days + 1
        fold_size = (span - self.test_size_days - self.gap_days) // self.n_splits
        for i in range(self.n_splits):
            train_end = first + pd.Timedelta(days=fold_size * (i + 1) - 1)
            val_start = train_end + pd.Timedelta(days=self.gap_days + 1)
            val_end = val_start + pd.Timedelta(days=self.test_size_days - 1)
            tr_idx = df[df[date_col] <= train_end].index
            va_idx = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)].index
            if len(va_idx) and len(tr_idx):
                yield tr_idx, va_idx

# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------

COL_MAPPING = {"num_date_time": "num_date_time", "건물번호": "building_id", "일시": "datetime", "기온(°C)": "temp", "강수량(mm)": "rainfall", "풍속(m/s)": "wind_speed", "습도(%)": "humidity", "일조(hr)": "sunshine_hours", "일사(MJ/m2)": "solar_radiation", "연면적(m2)": "total_area", "냉방면적(m2)": "cooling_area", "태양광용량(kW)": "pv_capacity", "건물유형": "building_type"}
NUMERIC_BUILDING_COLS = ["total_area", "cooling_area", "pv_capacity"]

def load_data(train_p: Path, test_p: Path, binfo_p: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test, binfo = pd.read_csv(train_p), pd.read_csv(test_p), pd.read_csv(binfo_p)
    for df in (train, test, binfo):
        df.rename(columns=COL_MAPPING, inplace=True)
    for df in (train, test):
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H")
    for col in NUMERIC_BUILDING_COLS:
        binfo[col] = pd.to_numeric(binfo[col].replace("-", np.nan))
    train = train.merge(binfo, on="building_id", how="left")
    test = test.merge(binfo, on="building_id", how="left")
    return train, test

# -----------------------------------------------------------------------------
# 2. Basic features
# -----------------------------------------------------------------------------

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]
    df["year"], df["month"], df["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    df["hour"], df["weekday"] = dt.dt.hour, dt.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_peak_flag"] = (((df["hour"].between(7, 9)) | (df["hour"].between(11, 14)) | (df["hour"].between(17, 19))) & (df["is_weekend"] == 0)).astype(int)
    return df

# -----------------------------------------------------------------------------
# 3. Advanced FE
# -----------------------------------------------------------------------------

class AdvancedFE:
    def __init__(self):
        self.clusters: Dict[int, int] = {}

    def _build_clusters(self, df: pd.DataFrame):
        feats = ["total_area", "cooling_area", "pv_capacity"]
                feats = ["total_area", "cooling_area", "pv_capacity"]
        tmp = (df.groupby("building_id")[feats]
                  .median()
                  .apply(np.log1p))
        # NaN → 컬럼별 중앙값으로
        tmp = tmp.fillna(tmp.median())
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.clusters = dict(zip(tmp.index, km.fit_predict(StandardScaler().fit_transform(tmp)))) = dict(zip(tmp.index, km.fit_predict(StandardScaler().fit_transform(tmp))))

    def _transform_one(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["building_cluster"] = df["building_id"].map(self.clusters).fillna(-1).astype(int)
        for col, div in (("quarter_hour", 15), ("half_hour", 30)):
            df[col] = (df["datetime"].dt.minute // div).astype(int)
            if df[col].nunique() == 1:
                df.drop(columns=[col], inplace=True)
        df["season_fine"] = pd.cut(df["datetime"].dt.dayofyear, bins=[0,79,171,263,354,366], labels=["late_winter","spring","summer","autumn","early_winter"])
        if {"temp","humidity","wind_speed"}.issubset(df.columns):
            svp = 6.105*np.exp(np.minimum(17.27*df["temp"]/(237.7+df["temp"]),50))
            df["apparent_temp"] = df["temp"] + 0.33*(df["humidity"]*svp/100) - 0.7*df["wind_speed"] - 4
            df["temp_humidity_hour"] = df["temp"]*df["humidity"]*df["hour"] / 10000
        df["morning_rush"] = ((df["hour"].between(7,9)) & (df["is_weekend"]==0)).astype(int)
        return df

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        self._build_clusters(train)
        return self._transform_one(train), self._transform_one(test)

# -----------------------------------------------------------------------------
# 4. Preprocessor
# -----------------------------------------------------------------------------

class Preprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.num_raw: List[str] = []
        self.cat_cols: List[str] = []
        self.enc: Dict[str, LabelEncoder] = {}

    def _identify(self, df: pd.DataFrame, target: str):
        self.num_raw = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        self.cat_cols = [c for c in df.columns if str(df[c].dtype) in ("object","category")]

    def fit_transform(self, tr: pd.DataFrame, te: pd.DataFrame, target="전력소비량(kWh)"):
        self._identify(tr, target)
        tr_sc = pd.DataFrame(self.scaler.fit_transform(tr[self.num_raw]), columns=[f"sc_{c}" for c in self.num_raw], index=tr.index)
        te_sc = pd.DataFrame(self.scaler.transform(te[self.num_raw]), columns=[f"sc_{c}" for c in self.num_raw], index=te.index)
        X_tr = pd.concat([tr[self.num_raw], tr_sc], axis=1)
        X_te = pd.concat([te[self.num_raw], te_sc], axis=1)
        for col in self.cat_cols:
            le = LabelEncoder(); le.fit(tr[col].astype(str).fillna("missing"))
            self.enc[col] = le
            X_tr[col] = le.transform(tr[col].astype(str).fillna("missing"))
            X_te[col] = te[col].astype(str).fillna("missing").map(lambda v: v if v in le.classes_ else "missing")
            X_te[col] = le.transform(X_te[col])
        y = np.log1p(tr[target]) if target in tr else None
        return X_tr, X_te, y

# -----------------------------------------------------------------------------
# 5. Ultimate Tuner
# -----------------------------------------------------------------------------

class UltimateTuner:
    def __init__(self, trials:int=30):
        self.trials = trials
        self.params: Dict[str,Dict] = {}
        self.models: Dict[str,object] = {}
        self.results: Dict[str,Dict] = {}

    # ------------------------------ HPO -------------------------------------
    def _hpo(self, X, y, dt_series):
        cv_splits = list(TimeSeriesCV(2,7,1).split(pd.DataFrame({"dt":dt_series}),"dt"))
        def cv_score(model_ctor):
            scores=[]
            for tr,va in cv_splits:
                model = model_ctor()
                model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], early_stopping_rounds=50, verbose=False)
                p=np.expm1(model.predict(X.iloc[va])); t=np.expm1(y.iloc[va])
                scores.append(smape(t,p))
            return np.mean(scores)
        def tune(name,obj):
            st=optuna.create_study(direction="minimize",study_name=name)
            st.optimize(obj,n_trials=self.trials, show_progress_bar=False)
            self.params[name]=st.best_params

        def obj_xgb(trial):
            p={"max_depth":trial.suggest_int("max_depth",5,10),"n_estimators":trial.suggest_int("n_estimators",300,600,100),"learning_rate":trial.suggest_float("lr",0.03,0.15),"subsample":trial.suggest_float("subsample",0.7,0.9),"colsample_bytree":trial.suggest_float("colsample",0.7,0.9),"reg_alpha":trial.suggest_float("ra",0.5,3),"reg_lambda":trial.suggest_float("rl",0.5,3),"tree_method":"gpu_hist","gpu_id":0,"verbosity":0}
            return cv_score(lambda: xgb.XGBRegressor(**p))
        def obj_lgb(trial):
            p={"max_depth":trial.suggest_int("max_depth",5,10),"n_estimators":trial.suggest_int("n_estimators",300,600,100),"learning_rate":trial.suggest_float("lr",0.03,0.15),"subsample":trial.suggest_float("subsample",0.7,0.9),"colsample_bytree":trial.suggest_float("colsample",0.7,0.9),"reg_alpha":trial.suggest_float("ra",0.5,3),"reg_lambda":trial.suggest_float("rl",0.5,3),"num_leaves":trial.suggest_int("nl",50,150),"device_type":"gpu","verbosity":-1}
            return cv_score(lambda: lgb.LGBMRegressor(**p))
        def obj_cb(trial):
            p={"depth":trial.suggest_int("depth",5,10),"iterations":trial.suggest_int("iters",300,600,100),"learning_rate":trial.suggest_float("lr",0.03,0.15),"subsample":trial.suggest_float("subsample",0.7,0.9),"reg_lambda":trial.suggest_float("rl",0.5,3),"task_type":"GPU","verbose":0}
            return cv_score(lambda: cb.CatBoostRegressor(**p))
        tune("xgb",obj_xgb); tune("lgb",obj_lgb); tune("cb",obj_cb)

    # --------------------------- model building ------------------------------
    def _build_models(self):
        self.models["xgb"] = xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0, **self.params["xgb"])
        self.models["lgb"] = lgb.LGBMRegressor(device_type="gpu", **self.params["lgb"])
        self.models["cb"]  = cb.CatBoostRegressor(task_type="GPU", verbose=0, **self.params["cb"])
        # optimize ensemble weights & ridge α
        def ens_obj(trial):
            w1, w2 = trial.suggest_float("w1",0.2,0.6), trial.suggest_float("w2",0.2,0.6)
            w3 = 1 - w1 - w2 if 0 < 1 - w1 - w2 else 0.2
            alpha = trial.suggest_float("alpha",0.1,5.0)
            vot = VotingRegressor([("x",self.models["xgb"]),("l",self.models["lgb"]),("c",self.models["cb"])],weights=[w1,w2,w3])
            stk = StackingRegressor(estimators=[("x",self.models["xgb"]),("l",self.models["lgb"]),("c",self.models["cb"])],final_estimator=Ridge(alpha=alpha),cv=3)
            # quick CV with small subset
            samp = np.random.choice(len(self.X), size=int(0.2*len(self.X)), replace=False)
            vot.fit(self.X.iloc[samp], self.y.iloc[samp]); p = np.expm1(vot.predict(self.X.iloc[samp])); t = np.expm1(self.y.iloc[samp])
            return smape(t,p)
        study = optuna.create_study(direction="minimize", study_name="ensemble")
        study.optimize(ens_obj, n_trials=15, show_progress_bar=False)
        w1, w2 = study.best_params["w1"], study.best_params["w2"]
        w3 = 1 - w1 - w2 if 0 < 1 - w1 - w2 else 0.2
        alpha = study.best_params["alpha"]
        self.models["voting"] = VotingRegressor([("x",self.models["xgb"]),("l",self.models["lgb"]),("c",self.models["cb"])],weights=[w1,w2,w3])
        self.models["stack"] = StackingRegressor(estimators=[("x",self.models["xgb"]),("l",self.models["lgb"]),("c",self.models["cb"])],final_estimator=Ridge(alpha=alpha),cv=3)

    # ------------------------------ validate ---------------------------------
    def _validate(self, dt_series):
        cv = list(TimeSeriesCV(3,7,1).split(pd.DataFrame({"dt":dt_series}),"dt"))
        for name, model in self.models.items():
            scores=[]
            for tr,va in cv:
                model.fit(self.X.iloc[tr], self.y.iloc[tr])
                p=np.expm1(model.predict(self.X.iloc[va])); t=np.expm1(self.y.iloc[va])
                scores.append(smape(t,p))
            self.results[name]={"mean":float(np.mean(scores)), "std":float(np.std(scores))}

    # ------------------------------ run --------------------------------------
    def run(self, X, y, dt_series):
        self.X, self.y = X, y  # for ensemble tuning quick access
        self._hpo(X,y,dt_series)
        self._build_models()
        self._validate(dt_series)
        best = min(self.results, key=lambda k: self.results[k]["mean"])
        best_model = self.models[best]
        best_model.fit(X,y)
        return best, best_model, self.results

# -----------------------------------------------------------------------------
# 6. End‑to‑End Pipeline
# -----------------------------------------------------------------------------

def run_pipeline(data_dir: str|Path="../data", trials:int=30):
    data_dir = Path(data_dir)
    train, test = load_data(data_dir/"train.csv", data_dir/"test.csv", data_dir/"building_info.csv")
    train = basic_features(train); test = basic_features(test)
    adv = AdvancedFE(); train, test = adv.fit_transform(train, test)
    prep = Preprocessor(); X_tr, X_te, y_tr = prep.fit_transform(train, test)
    tuner = UltimateTuner(trials=trials)
    best_name, best_model, results = tuner.run(X_tr, y_tr, train["datetime"].reset_index(drop=True))
    preds = np.maximum(np.expm1(best_model.predict(X_te)),0)
    sub = pd.DataFrame({"num_date_time": test["num_date_time"], "answer": preds})
    fname = f"submission_{best_name}.csv"; sub.to_csv(fname, index=False)
    print(f"✔ Saved {fname} | Best SMAPE≈{results[best_name]['mean']:.3f}")

# -----------------------------------------------------------------------------
# 7. main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(data_dir="../data", trials=25)
