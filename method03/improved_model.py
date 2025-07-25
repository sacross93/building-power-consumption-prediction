import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings
import holidays

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------
# 주석 설명: 코드 로직 및 개선 사항 (v4.1 - Gemini Pro)
# ------------------------------------------------------------------------------------------------
# [개선점 4] 시각 자료 분석 기반 피처 엔지니어링 및 타겟 변환:
#    (이전 버전과 동일)
# [수정점 1] 피처 생성 로직 수정:
#    - KeyError 방지를 위해 훈련/테스트 데이터를 통합한 후 시차 피처를 일괄 생성하고, 다시 분리하는 방식으로 수정.
# ------------------------------------------------------------------------------------------------

def smape(y_true, y_pred):
    epsilon = 1e-10
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    numerator = 2 * np.abs(y_pred_exp - y_true_exp)
    denominator = np.abs(y_true_exp) + np.abs(y_pred_exp) + epsilon
    return np.mean(numerator / denominator) * 100

def create_features(df):
    df = df.sort_values(by=['건물번호', '일시']).reset_index(drop=True)

    # 1. 시간 관련 피처
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_year'] = df['일시'].dt.dayofyear

    # 2. 공휴일 피처
    kr_holidays = holidays.KR()
    df['is_holiday'] = df['일시'].dt.date.apply(lambda x: 1 if x in kr_holidays else 0).astype(int)

    # 3. 날씨 관련 피처
    df['THI'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    df['CDD'] = np.maximum(0, df['기온(°C)'] - 21)
    df['HDD'] = np.maximum(0, 18 - df['기온(°C)'])

    # 4. 상호작용 및 이동통계 피처
    df['temp_x_hour'] = df['기온(°C)'] * df['hour']
    df['temp_rolling_mean_6'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['temp_rolling_std_6'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=6, min_periods=1).std()).fillna(0)
    df['humidity_rolling_mean_6'] = df.groupby('건물번호')['습도(%)'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['humidity_rolling_std_6'] = df.groupby('건물번호')['습도(%)'].transform(lambda x: x.rolling(window=6, min_periods=1).std()).fillna(0)

    # 5. 시차 변수 (Data Leakage 방지)
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df[f'power_lag_{lag}'] = df.groupby('건물번호')['전력소비량(kWh)'].transform(lambda x: x.shift(lag))
        df[f'temp_lag_{lag}'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.shift(lag))

    return df

def main():
    print("데이터 로딩 시작...")
    train_df = pd.read_csv('data/train.csv', parse_dates=['일시'])
    test_df = pd.read_csv('data/test.csv', parse_dates=['일시'])
    building_info_df = pd.read_csv('data/building_info.csv')
    print("데이터 로딩 완료.")

    print("건물 정보 전처리 시작...")
    numeric_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in numeric_cols:
        building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)
        building_info_df[col] = np.log1p(building_info_df[col])
    print("건물 정보 전처리 완료.")

    print("피처 엔지니어링 시작...")
    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')
    
    test_df['전력소비량(kWh)'] = np.nan
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    combined_df = create_features(combined_df)
    
    train_processed_df = combined_df[~combined_df['전력소비량(kWh)'].isna()].copy()
    test_processed_df = combined_df[combined_df['전력소비량(kWh)'].isna()].copy()
    print("피처 엔지니어링 완료.")

    # 타겟 변수 로그 변환
    train_processed_df['전력소비량(kWh)'] = np.log1p(train_processed_df['전력소비량(kWh)'])

    features = [
        '건물번호', '기온(°C)', '풍속(m/s)', '습도(%)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        'month', 'day', 'hour', 'weekday', 'is_weekend', 'is_holiday',
        'hour_sin', 'hour_cos', 'THI', 'temp_x_hour',
        'temp_rolling_mean_6', 'temp_rolling_std_6',
        'humidity_rolling_mean_6', 'humidity_rolling_std_6',
        'day_of_year', 'CDD', 'HDD',
        'power_lag_1', 'temp_lag_1', 'power_lag_2', 'temp_lag_2', 'power_lag_3', 'temp_lag_3',
        'power_lag_24', 'temp_lag_24', 'power_lag_48', 'temp_lag_48', 'power_lag_168', 'temp_lag_168'
    ]
    
    categorical_features_for_model = ['건물번호']
    train_processed_df['건물번호'] = train_processed_df['건물번호'].astype('category')
    train_processed_df['건물유형'] = train_processed_df['건물유형'].astype('category')
    test_processed_df['건물번호'] = test_processed_df['건물번호'].astype('category')
    test_processed_df['건물유형'] = test_processed_df['건물유형'].astype('category')

    print("건물 유형별 모델 학습 및 검증 시작...")
    
    building_types = train_processed_df['건물유형'].cat.categories
    total_predictions = []
    
    for b_type in building_types:
        print(f"--- {b_type} 유형 모델 학습 ---")
        
        type_train_df = train_processed_df[train_processed_df['건물유형'] == b_type].copy()
        type_test_df = test_processed_df[test_processed_df['건물유형'] == b_type].copy()

        type_train_df.dropna(subset=[col for col in features if 'power_lag' in col], inplace=True)
        
        if type_train_df.empty:
            print(f"{b_type} 유형은 학습할 데이터가 충분하지 않습니다. 건너뜁니다.")
            continue

        X_train = type_train_df[features]
        y_train = type_train_df['전력소비량(kWh)']
        X_test = type_test_df[features]

        # --- 데이터 진단 코드 시작 ---
        print(f"--- In-depth analysis for building type: {b_type} ---")
        print(f"Shape of training data for this type (after dropna): {X_train.shape}")

        if X_train.shape[0] < 50: # 데이터가 너무 적으면 경고
            print("!!! WARNING: Very few data points after processing. This is likely the cause of the issue.")

        print("\n>>> Feature variance check (number of unique values):")
        for col in X_train.columns:
            # 고유값이 1개인 피처(상수)가 있는지 확인
            nunique = X_train[col].nunique()
            if nunique == 1:
                print(f"!!! CONSTANT FEATURE: '{col}' has only 1 unique value.")
            # elif nunique < 5:
            #     print(f"- '{col}': {nunique} unique values (very low variance)")

        print("\n>>> Target variable description:")
        print(y_train.describe())
        print("-----------------------------------------------------\n")
        # --- 데이터 진단 코드 끝 ---

        lgb_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'random_state': 42,
            'device': 'gpu',
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'num_leaves': 32,
            'max_depth': 8,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        final_model = lgb.LGBMRegressor(**lgb_params)
        final_model.fit(X_train, y_train)
        
        if not X_test.empty:
            preds = final_model.predict(X_test)
            preds = np.expm1(preds)
            preds[preds < 0] = 0
            
            temp_submission = pd.DataFrame({'num_date_time': type_test_df['num_date_time'], 'answer': preds})
            total_predictions.append(temp_submission)

    print("-----------------------------------------")

    print("결과 취합 및 저장 시작...")
    if total_predictions:
        final_submission = pd.concat(total_predictions, ignore_index=True)
        
        sample_submission = pd.read_csv('data/sample_submission.csv')
        sample_submission = sample_submission.drop(columns=['answer'])
        final_submission = pd.merge(sample_submission, final_submission, on='num_date_time', how='left')
        
        final_submission['answer'].fillna(0, inplace=True)

        final_submission.to_csv('method03/submission.csv', index=False)
        print("method03/submission.csv 파일이 성공적으로 생성되었습니다.")
    else:
        print("예측이 생성되지 않았습니다.")

if __name__ == '__main__':
    main()