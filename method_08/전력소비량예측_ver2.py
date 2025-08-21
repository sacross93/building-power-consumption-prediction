import pandas as pd 
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

import math
import matplotlib.pyplot as plt
import seaborn as sns

# SMAPE 계산 함수 (옵션: 마스킹, 분모 하한)
def smape(y_true, y_pred, eps: float = 1e-6, denom_floor: float | None = None, mask: np.ndarray | None = None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    if denom_floor is not None:
        denom = np.maximum(denom, denom_floor)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

# 한글 폰트 및 마이너스
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# GPU 사용 가능 여부 점검 (XGBoost로 직접 확인)
def detect_gpu_available() -> bool:
    try:
        X_small = np.random.rand(128, 8).astype(np.float32)
        y_small = np.random.rand(128).astype(np.float32)
        dtrain = xgb.DMatrix(X_small, label=y_small)
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'max_depth': 2,
            'nthread': 1,
            'verbosity': 0,
        }
        xgb.train(params, dtrain, num_boost_round=1)
        return True
    except Exception as e:
        print('GPU 사용 불가로 판단:', str(e))
        return False

# 데이터 로드
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

print(train_df.shape, test_df.shape, submission.shape)

# 학습 시 사용할 입력 컬럼은 test_df 기준으로 제한 (동일 파생 보장)
TEST_BASE_INPUT_COLS = [c for c in test_df.columns if c not in ['num_date_time']]


# 1) 날짜 파생변수 생성
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if '일시' in out.columns:
        if not np.issubdtype(out['일시'].dtype, np.datetime64):
            out['일시'] = pd.to_datetime(out['일시'], errors='coerce')
        out['year'] = out['일시'].dt.year
        out['month'] = out['일시'].dt.month
        out['day'] = out['일시'].dt.day
        out['hour'] = out['일시'].dt.hour
        out['dayofweek'] = out['일시'].dt.dayofweek
        out['quarter'] = out['일시'].dt.quarter
        out['is_weekend'] = out['dayofweek'].isin([5, 6]).astype(int)

        def season(m):
            if m in [12, 1, 2]:
                return 0
            if m in [3, 4, 5]:
                return 1
            if m in [6, 7, 8]:
                return 2
            return 3
        out['season'] = out['month'].apply(season)
    return out

# 2) 특징/타깃 구성
def build_feature_target(df: pd.DataFrame, target_col: str):
    # 모델 입력은 test_df에 존재하는 컬럼으로 제한하여 일관성 확보
    drop_targets = ['일사(MJ/m2)', '일조(hr)']
    allowed_cols = [c for c in TEST_BASE_INPUT_COLS if c not in drop_targets]
    base_cols = [c for c in df.columns if c in allowed_cols]

    # 파생변수 생성
    fe = add_time_features(df[base_cols])

    # 원본 '일시' 제거(파생변수로 대체)
    if '일시' in fe.columns:
        fe = fe.drop(columns=['일시'])

    # 건물번호를 범주형으로 캐스팅(정수 순서 의미 제거)
    if '건물번호' in fe.columns:
        fe['건물번호'] = fe['건물번호'].astype('category')

    # 수치/범주 자동 분리
    num_cols = fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = fe.select_dtypes(exclude=[np.number]).columns.tolist()

    X = fe
    y = df[target_col]
    return X, y, num_cols, cat_cols

# 3) 전처리 + XGB 파이프라인
def make_pipeline(num_cols, cat_cols, use_gpu: bool = False):
    num_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False는 버전 이슈 가능 → 기본 설정(희소) 유지
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    pre = ColumnTransformer(
        transformers=[
            ('num', num_tf, num_cols),
            ('cat', cat_tf, cat_cols)
        ],
        remainder='drop'
    )

    model_params = dict(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method='gpu_hist' if use_gpu else 'hist',
        random_state=42,
        n_jobs=-1,
    )
    if use_gpu:
        # 일부 환경에서 필요
        model_params['predictor'] = 'gpu_predictor'
    model = xgb.XGBRegressor(**model_params)

    pipe = Pipeline(steps=[
        ('pre', pre),
        ('model', model)
    ])
    return pipe

# 4) 학습 및 평가
def fit_and_eval(df_train_full: pd.DataFrame, target_col: str, test_size=0.2, random_state=42, split_mode: str = 'time'):
    # 두 타깃 모두 있는 행만 사용해 학습(사용자 의도 유지)
    train_df_nonnull = df_train_full.dropna(subset=['일사(MJ/m2)', '일조(hr)'])

    X, y, num_cols, cat_cols = build_feature_target(train_df_nonnull, target_col)

    # 시간 기반 분할(default): 과거 -> 학습, 미래 -> 검증
    if split_mode == 'time':
        if '일시' in train_df_nonnull.columns:
            ts = pd.to_datetime(train_df_nonnull['일시'], errors='coerce')
        else:
            # 안전장치: 파생에서 만든 year/month/hour 등으로 대체 불가 시 랜덤 분할로 폴백
            ts = pd.Series(np.arange(len(train_df_nonnull)), index=train_df_nonnull.index)
        order = np.argsort(ts.values)
        idx_sorted = train_df_nonnull.index.values[order]
        split_point = int(len(idx_sorted) * (1.0 - test_size))
        tr_idx = idx_sorted[:split_point]
        te_idx = idx_sorted[split_point:]
        X_tr, X_te = X.loc[tr_idx], X.loc[te_idx]
        y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
        val_times = ts.loc[te_idx]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # 랜덤 분할 시에도 val_times 구성 가능하면 구성
        if '일시' in train_df_nonnull.columns:
            ts = pd.to_datetime(train_df_nonnull['일시'], errors='coerce')
            val_times = ts.loc[y_te.index]
        else:
            val_times = pd.Series(index=y_te.index, dtype='datetime64[ns]')

    use_gpu = detect_gpu_available()
    print(f"GPU 사용 감지 결과: {use_gpu}")
    pipe = make_pipeline(num_cols, cat_cols, use_gpu=use_gpu)
    try:
        pipe.fit(X_tr, y_tr)
    except xgb.core.XGBoostError as e:
        print('GPU 학습 시도 실패. CPU로 재시도합니다. reason=', str(e))
        pipe = make_pipeline(num_cols, cat_cols, use_gpu=False)
        pipe.fit(X_tr, y_tr)
    except Exception as e:
        print('GPU 학습 시도 중 예기치 못한 오류. CPU로 재시도합니다. reason=', str(e))
        pipe = make_pipeline(num_cols, cat_cols, use_gpu=False)
        pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    # 구버전 scikit-learn 호환: squared 인자 미지원 환경 대응
    rmse = mean_squared_error(y_te, pred) ** 0.5
    r2 = r2_score(y_te, pred)

    # sMAPE: 기본, 주간만, 분모 하한
    sm_standard = smape(y_te, pred)
    # 주간(유효구간): 6~18시 기준
    daytime_mask = None
    try:
        val_times_dt = pd.to_datetime(val_times, errors='coerce')
        hours = val_times_dt.dt.hour
        daytime_mask = (hours >= 6) & (hours <= 18)
    except Exception:
        daytime_mask = None
    sm_daytime = smape(y_te, pred, mask=daytime_mask) if daytime_mask is not None else np.nan
    # 분모 하한(폭주 방지): 1.0
    sm_floor = smape(y_te, pred, denom_floor=1.0)

    return pipe, {
        'rmse': rmse,
        'r2': r2,
        'smape': sm_standard,
        'smape_daytime': sm_daytime,
        'smape_floor': sm_floor,
        'n_train': len(X_tr),
        'n_valid': len(X_te)
    }

# 5) 모델 학습
model_irr, metrics_irr = fit_and_eval(train_df, target_col='일사(MJ/m2)')  # irradiance
model_sun, metrics_sun = fit_and_eval(train_df, target_col='일조(hr)')     # sunshine duration

print("=== 일사(MJ/m2) 모델 ===")
print(metrics_irr)
print("=== 일조(hr) 모델 ===")
print(metrics_sun)


# 6) 결측치 채우기
def predict_for_df(model, df_like: pd.DataFrame):
    # 예측 시에도 train과 동일하게 test_df 기준 컬럼만 사용
    base_cols = [c for c in TEST_BASE_INPUT_COLS if c in df_like.columns]
    X_all = add_time_features(df_like[base_cols])
    if '일시' in X_all.columns:
        X_all = X_all.drop(columns=['일시'])
    if '건물번호' in X_all.columns:
        X_all['건물번호'] = X_all['건물번호'].astype('category')
    return model.predict(X_all)

filled_df = train_df.copy()

# (a) 일사 결측 채우기
mask_irr_na = filled_df['일사(MJ/m2)'].isna()
if mask_irr_na.any():
    filled_df.loc[mask_irr_na, '일사(MJ/m2)'] = predict_for_df(model_irr, filled_df[mask_irr_na])

# (b) 일조 결측 채우기
mask_sun_na = filled_df['일조(hr)'].isna()
if mask_sun_na.any():
    filled_df.loc[mask_sun_na, '일조(hr)'] = predict_for_df(model_sun, filled_df[mask_sun_na])

# 7) 결과 확인
print("결측 보정 전/후 결측 개수")
print({
    '일사_before_na': train_df['일사(MJ/m2)'].isna().sum(),
    '일사_after_na':  filled_df['일사(MJ/m2)'].isna().sum(),
    '일조_before_na': train_df['일조(hr)'].isna().sum(),
    '일조_after_na':  filled_df['일조(hr)'].isna().sum(),
})
# 이후 파이프라인에 filled_df 사용
# train_df = filled_df

# 8) test_df에 일사/일조 예측 컬럼 추가
test_df['일사(MJ/m2)'] = predict_for_df(model_irr, test_df)
test_df['일조(hr)'] = predict_for_df(model_sun, test_df)
# 저장용: 날짜 파생변수 컬럼도 test_df에 병합
_test_fe = add_time_features(test_df)
for _col in ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter', 'is_weekend', 'season']:
    if _col in _test_fe.columns and _col not in test_df.columns:
        test_df[_col] = _test_fe[_col]
print("test_df 예측 컬럼 추가 후 info:")
print(test_df.info())

# test_df 한글인코딩 저장
train_df.to_csv('./data/train_filled.csv', index=False, encoding='cp949')
test_df.to_csv('./data/test_filled.csv', index=False, encoding='cp949')