import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings
import holidays

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------
# 주석 설명: 코드 로직 및 개선 사항 (v4 - Gemini Pro)
# ------------------------------------------------------------------------------------------------
# [개선점 4] 시각 자료 분석 기반 피처 엔지니어링 및 타겟 변환:
#    1. **타겟 변수 로그 변환:** 전력소비량의 분포가 오른쪽으로 심하게 치우쳐 있으므로, np.log1p를 적용하여 타겟 변수를 정규분포에 가깝게 변환합니다.
#       이는 모델이 이상치에 덜 민감하게 만들고, SMAPE와 같은 상대적 오차 지표에 더 잘 최적화되도록 돕습니다. 예측 후에는 np.expm1로 원래 스케일로 복원합니다.
#    2. **단기 시차 피처 추가:** ACF/PACF 분석 결과, 1-3시간의 단기 자기상관성이 높게 나타났습니다. 기존의 장기 시차(24, 48, 168)에 더해
#       'power_lag_1', 'power_lag_2', 'power_lag_3' 및 'temp_lag_1', 'temp_lag_2', 'temp_lag_3'을 추가하여 단기 동적 변화를 포착합니다.
#    3. **비선형 온도 피처 (HDD/CDD):** 온도와 전력 소비 간의 U자형 관계를 명시적으로 모델링하기 위해 난방도일(HDD)과 냉방도일(CDD)을 추가합니다.
#       이는 특정 기준 온도보다 높거나 낮을 때의 난방/냉방 수요를 직접적으로 나타내는 강력한 피처입니다.
#    4. **계절 추세 피처:** 월(month) 피처보다 더 연속적인 시간의 흐름을 나타내기 위해 'day_of_year'를 추가하여 여름철 점진적인 소비량 증가 추세를 모델이 학습하도록 돕습니다.
#    5. **치우친 수치형 피처 로그 변환:** '연면적', '냉방면적', '태양광용량' 등 오른쪽으로 치우친 분포를 가진 건물 정보 피처들에 np.log1p를 적용하여 데이터 분포를 안정화시킵니다.
#    6. **날씨 이동 통계 확장:** 기존 온도 이동 통계에 더해 '습도'에 대한 이동 평균/표준편차를 추가하여 최근 습도 추세 및 변동성을 포착합니다.
# ------------------------------------------------------------------------------------------------

def smape(y_true, y_pred):
    epsilon = 1e-10
    # 로그 변환된 타겟으로 훈련했으므로, 원래 값으로 되돌려서 SMAPE를 계산해야 합니다.
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
        'humidity_rolling_mean_6', 'humidity_rolling_std_6', # 습도 이동 통계 추가
        'day_of_year', 'CDD', 'HDD', # 새로운 시간/온도 피처 추가
        'power_lag_1', 'temp_lag_1', 'power_lag_2', 'temp_lag_2', 'power_lag_3', 'temp_lag_3', # 단기 시차
        'power_lag_24', 'temp_lag_24', 'power_lag_48', 'temp_lag_48', 'power_lag_168', 'temp_lag_168' # 장기 시차
    ]
    # 일부 건물 유형에는 특정 시차 피처가 모두 NaN일 수 있으므로, 모델 학습 전에 확인
    
    categorical_features_for_model = ['건물번호', '건물유형']
    train_df['건물번호'] = train_df['건물번호'].astype('category')
    train_df['건물유형'] = train_df['건물유형'].astype('category')
    test_df['건물번호'] = test_df['건물번호'].astype('category')
    test_df['건물유형'] = test_df['건물유형'].astype('category')


    print("건물 유형별 모델 학습 및 검증 시작...")
    
    building_types = train_df['건물유형'].unique()
    total_predictions = []
    
    for b_type in building_types:
        print(f"--- {b_type} 유형 모델 학습 ---")
        
        type_train_df = train_df[train_df['건물유형'] == b_type].copy()
        type_test_df = test_df[test_df['건물유형'] == b_type].copy()

        # 시차 변수 생성 후 NaN 값이 있는 행 제거
        type_train_df.dropna(subset=[col for col in features if 'power_lag' in col], inplace=True)
        
        if type_train_df.empty:
            print(f"{b_type} 유형은 학습할 데이터가 충분하지 않습니다. 건너뜁니다.")
            continue

        X_train = type_train_df[features]
        y_train = type_train_df['전력소비량(kWh)']
        X_test = type_test_df[features]

        lgb_params = {
            'objective': 'regression_l1', # MAE
            'metric': 'mae',
            'random_state': 42,
            'device': 'gpu', # GPU 사용 설정
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
        final_model.fit(X_train, y_train, categorical_feature=categorical_features_for_model)
        
        if not X_test.empty:
            preds = final_model.predict(X_test)
            # [개선점 4.1] 예측값 역변환
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
        
        # 일부 건물 유형이 학습되지 않아 예측이 없는 경우를 대비해 채우기
        final_submission['answer'].fillna(0, inplace=True)

        final_submission.to_csv('submission_improved.csv', index=False)
        print("submission_improved.csv 파일이 성공적으로 생성되었습니다.")
    else:
        print("예측이 생성되지 않았습니다.")

if __name__ == '__main__':
    main()
