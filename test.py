
import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings
import holidays

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------
# 주석 설명: 코드 로직 및 개선 사항 (v3)
# ------------------------------------------------------------------------------------------------
# [개선점 3] 추가 피처 엔지니어링 및 하이퍼파라미터 튜닝:
#    - `holidays` 라이브러리를 사용하여 '공휴일' 피처를 추가합니다. 공휴일은 주말과 유사하게 전력 소비 패턴에 큰 영향을 미치므로 중요한 변수입니다.
#    - 날씨와 시간의 상호작용을 더 잘 모델링하기 위해 '기온 * 시간', '습도 * 시간'과 같은 상호작용 피처를 추가합니다.
#    - 날씨의 단기적 추세를 반영하기 위해 6시간 이동 평균 및 표준편차(`rolling`) 피처를 추가합니다.
#    - LightGBM 모델의 하이퍼파라미터를 데이터 양과 특성에 맞게 일부 조정하여 과적합을 방지하고 성능을 개선합니다. (예: `n_estimators` 증가, `learning_rate` 감소)
#
# 1. 데이터 로딩 및 기본 전처리: 이전과 동일
# 2. 피처 엔지니어링 통합 수행:
#    - `create_features` 함수 내에 공휴일, 상호작용, 이동 평균 피처 생성 로직을 추가합니다.
# 3. SMAPE 평가 함수: 이전과 동일
# 4. 건물 유형별 모델링 및 검증:
#    - 개선된 피처셋으로 각 건물 유형별 모델을 학습하고 검증합니다.
#    - Early Stopping을 통해 찾은 최적의 트리 개수를 최종 학습에 사용하여 과적합을 방지합니다.
# 5. 최종 예측 및 제출:
#    - 최적화된 모델로 최종 예측을 수행하고 제출 파일을 생성합니다.
# ------------------------------------------------------------------------------------------------

def smape(y_true, y_pred):
    epsilon = 1e-10
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
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
    
    # 2. 공휴일 피처
    kr_holidays = holidays.KR()
    df['is_holiday'] = df['일시'].dt.date.apply(lambda x: 1 if x in kr_holidays else 0).astype(int)
    
    # 3. 날씨 관련 피처
    df['THI'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    
    # 4. 상호작용 및 이동통계 피처
    df['temp_x_hour'] = df['기온(°C)'] * df['hour']
    df['temp_rolling_mean_6'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['temp_rolling_std_6'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=6, min_periods=1).std()).fillna(0)

    # 5. 시차 변수 (Data Leakage 방지)
    lags = [24, 48, 168]
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

    print("데이터 전처리 및 병합 시작...")
    numeric_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in numeric_cols:
        building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)

    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')
    
    test_df['전력소비량(kWh)'] = np.nan # test 데이터의 전력소비량은 NaN으로 초기화
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print("데이터 전처리 및 병합 완료.")

    print("피처 엔지니어링 시작...")
    combined_df = create_features(combined_df)
    
    combined_df['건물번호'] = combined_df['건물번호'].astype('category')
    combined_df['건물유형'] = combined_df['건물유형'].astype('category')
    
    train_processed_df = combined_df[~combined_df['전력소비량(kWh)'].isna()].copy()
    test_processed_df = combined_df[combined_df['전력소비량(kWh)'].isna()].copy()
    
    print("피처 엔지니어링 완료.")

    features = [
        '건물번호', '기온(°C)', '풍속(m/s)', '습도(%)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        'month', 'day', 'hour', 'weekday', 'is_weekend', 'is_holiday',
        'hour_sin', 'hour_cos', 'THI', 'temp_x_hour', 
        'temp_rolling_mean_6', 'temp_rolling_std_6',
        'power_lag_24', 'temp_lag_24', 'power_lag_48', 'temp_lag_48', 'power_lag_168', 'temp_lag_168'
    ]
    categorical_features_for_model = ['건물번호']

    print("건물 유형별 모델 학습 및 검증 시작...")
    
    building_types = train_processed_df['건물유형'].unique()
    total_predictions = []
    total_smape_score = 0
    
    for b_type in building_types:
        print(f"--- {b_type} 유형 모델 학습 ---")
        
        type_train_df = train_processed_df[train_processed_df['건물유형'] == b_type].copy()
        
        split_date = pd.to_datetime('2024-08-18 00:00:00')
        train_val_df = type_train_df[type_train_df['일시'] < split_date]
        valid_df = type_train_df[type_train_df['일시'] >= split_date]

        X_train_val = train_val_df[features]
        y_train_val = train_val_df['전력소비량(kWh)']
        X_valid = valid_df[features]
        y_valid = valid_df['전력소비량(kWh)']

        lgb_params = {
            'objective': 'regression_l1',
            'random_state': 42,
            'n_estimators': 2000, # 학습 횟수 증가
            'learning_rate': 0.02, # 학습률 감소
            'num_leaves': 32,
            'max_depth': 8,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'reg_alpha': 0.1, # L1 정규화
            'reg_lambda': 0.1 # L2 정규화
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        
        lgb_model.fit(X_train_val, y_train_val, eval_set=[(X_valid, y_valid)], 
                      callbacks=[lgb.early_stopping(100, verbose=False)], 
                      categorical_feature=categorical_features_for_model)
        
        best_iter = lgb_model.best_iteration_
        valid_preds = lgb_model.predict(X_valid, num_iteration=best_iter)
        valid_preds[valid_preds < 0] = 0
        type_smape = smape(y_valid.values, valid_preds)
        total_smape_score += type_smape
        print(f"{b_type} 유형 검증 SMAPE: {type_smape:.4f} (최적 반복: {best_iter})")

        X_train_full = type_train_df[features]
        y_train_full = type_train_df['전력소비량(kWh)']
        
        final_model = lgb.LGBMRegressor(**lgb_params)
        # 최적 반복 횟수로 전체 데이터 재학습
        final_model.fit(X_train_full, y_train_full, categorical_feature=categorical_features_for_model)
        
        type_test_df = test_processed_df[test_processed_df['건물유형'] == b_type]
        
        if not type_test_df.empty:
            X_test = type_test_df[features]
            preds = final_model.predict(X_test)
            preds[preds < 0] = 0
            
            temp_submission = pd.DataFrame({'num_date_time': type_test_df['num_date_time'], 'answer': preds})
            total_predictions.append(temp_submission)

    print("-----------------------------------------")
    print(f"평균 검증 SMAPE 점수: {total_smape_score / len(building_types):.4f}")
    print("-----------------------------------------")

    print("결과 취합 및 저장 시작...")
    final_submission = pd.concat(total_predictions, ignore_index=True)
    
    sample_submission = pd.read_csv('data/sample_submission.csv')
    sample_submission = sample_submission.drop(columns=['answer'])
    final_submission = pd.merge(sample_submission, final_submission, on='num_date_time', how='left')
    
    final_submission.to_csv('submission.csv', index=False)
    print("submission.csv 파일이 성공적으로 생성되었습니다.")

if __name__ == '__main__':
    main()
