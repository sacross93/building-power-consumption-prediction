import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import warnings
import random as rn

# --- 기본 설정 ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

# --- 사용자 정의 함수 ---

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * abs(y_pred - y_true) / (abs(y_pred) + abs(y_true))) * 100

def weighted_mse(alpha=1):
    def weighted_mse_fixed(predt, dtrain):
        label = dtrain.get_label()
        residual = (label - predt).astype("float")
        grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * alpha, 2.0)
        return grad, hess
    return weighted_mse_fixed

def custom_smape_metric(predt, dtrain):
    label = dtrain.get_label()
    predt_exp = np.exp(predt)
    label_exp = np.exp(label)
    return 'custom_smape', smape(label_exp, predt_exp)

def main():
    print("1. 데이터 로딩 및 기본 전처리...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    building_info = pd.read_csv('data/building_info.csv')

    # 건물 유형 영문 변환 및 설비 피처 생성
    translation_dict = {
        '건물기타': 'Other Buildings', '공공': 'Public', '대학교': 'University', '백화점및아울렛': 'Department Store',
        '병원': 'Hospital', '상업': 'Commercial', '아파트': 'Apartment', '연구소': 'Research Institute',
        '데이터센터': 'IDC', '호텔및리조트': 'Hotel'
    }
    building_info['건물유형'] = building_info['건물유형'].replace(translation_dict)
    building_info['태양광용량(kW)'] = building_info['태양광용량(kW)'].replace('-', '0').astype(float)
    building_info['ESS저장용량(kWh)'] = building_info['ESS저장용량(kWh)'].replace('-', '0').astype(float)
    building_info['solar_power_utility'] = np.where(building_info['태양광용량(kW)'] > 0, 1, 0)
    building_info['ess_utility'] = np.where(building_info['ESS저장용량(kWh)'] > 0, 1, 0)
    
    train = pd.merge(train, building_info.drop(columns=['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']), on='건물번호', how='left')
    test = pd.merge(test, building_info.drop(columns=['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']), on='건물번호', how='left')

    # 0인 전력소비량 제거
    train = train[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

    print("2. 피처 엔지니어링...")
    
    # --- 2-1. 시간 관련 피처 생성 ---
    for df in [train, test]:
        df['date_time'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        df['holiday'] = np.where(df['day_of_week'] >= 5, 1, 0)
        df['THI'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)'] / 100) * (9/5 * df['기온(°C)'] - 26) + 32
        df['WCT'] = 13.12 + 0.6125 * df['기온(°C)'] - 11.37 * (df['풍속(m/s)']**0.16) + 0.3965 * (df['풍속(m/s)']**0.16) * df['기온(°C)']

    # --- 2-2. 과거 데이터 기반 통계 피처 생성 (Train 데이터 기준) ---
    power_mean = pd.pivot_table(train, values='전력소비량(kWh)', index=['건물번호', 'hour', 'day_of_week'], aggfunc=np.mean).reset_index()
    power_mean.rename(columns={'전력소비량(kWh)': 'day_hour_mean'}, inplace=True)
    
    power_std = pd.pivot_table(train, values='전력소비량(kWh)', index=['건물번호', 'hour', 'day_of_week'], aggfunc=np.std).reset_index()
    power_std.rename(columns={'전력소비량(kWh)': 'day_hour_std'}, inplace=True)

    # --- 2-3. 통계 피처 병합 ---
    train = pd.merge(train, power_mean, on=['건물번호', 'hour', 'day_of_week'], how='left')
    test = pd.merge(test, power_mean, on=['건물번호', 'hour', 'day_of_week'], how='left')
    train = pd.merge(train, power_std, on=['건물번호', 'hour', 'day_of_week'], how='left')
    test = pd.merge(test, power_std, on=['건물번호', 'hour', 'day_of_week'], how='left')

    # 사용할 피처 선택
    features = [
        '건물번호', '기온(°C)', '풍속(m/s)', '습도(%)',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week',
        'holiday', 'THI', 'WCT',
        'day_hour_mean', 'day_hour_std',
        'solar_power_utility', 'ess_utility'
    ]
    target = '전력소비량(kWh)'

    # K-Fold 설정
    KFOLD_SPLITS = 5
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # 결과 저장을 위한 배열 초기화
    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])
    
    print("3. 건물 유형별 모델 학습 및 교차 검증...")
    for b_type in train['건물유형'].unique():
        print(f"\n--- 건물 유형: {b_type} ---")
        
        train_b_type_idx = train[train['건물유형'] == b_type].index
        test_b_type_idx = test[test['건물유형'] == b_type].index
        
        X = train.loc[train_b_type_idx, features]
        y = train.loc[train_b_type_idx, target]
        X_test = test.loc[test_b_type_idx, features]

        y_log = np.log1p(y)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold}/{KFOLD_SPLITS} 학습 시작...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model = xgb.XGBRegressor(
                learning_rate=0.05, n_estimators=5000, max_depth=10,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                random_state=RANDOM_SEED, objective=weighted_mse(alpha=3),
                tree_method="gpu_hist", gpu_id=0, n_jobs=-1,
                eval_metric=custom_smape_metric
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                verbose=False
            )
            
            val_preds = model.predict(X_val)
            oof_preds[X_val.index] = np.expm1(val_preds)
            test_preds[X_test.index] += np.expm1(model.predict(X_test)) / KFOLD_SPLITS

    total_smape = smape(train[target], oof_preds)
    print(f"\n--- 최종 OOF SMAPE: {total_smape:.4f} ---")

    print("4. 제출 파일 생성...")
    submission = pd.read_csv('data/sample_submission.csv')
    submission['answer'] = test_preds
    submission.to_csv('method03/submission.csv', index=False)
    print("method03/submission.csv 파일이 성공적으로 생성되었습니다.")

if __name__ == '__main__':
    main()
# End of script
