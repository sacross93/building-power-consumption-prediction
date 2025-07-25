#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 개선 버전: test.py 참고한 체계적 접근법
깔끔한 데이터 플로우 + 건물별 개별 모델 + 적절한 피처 엔지니어링
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import holidays
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """SMAPE 계산 (test.py와 동일)"""
    epsilon = 1e-10
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(numerator / denominator) * 100

def create_features(df):
    """체계적인 피처 엔지니어링 (test.py 기반 + 개선)"""
    df = df.sort_values(by=['건물번호', '일시']).reset_index(drop=True)
    
    # 1. 시간 관련 피처
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 주기성 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # 2. 한국 공휴일 피처 (test.py에서 가져옴)
    kr_holidays = holidays.KR(years=[2024])
    df['is_holiday'] = df['일시'].dt.date.apply(lambda x: 1 if x in kr_holidays else 0).astype(int)
    
    # 3. 온도-습도 불쾌지수 (THI) - test.py에서 가져옴
    df['THI'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    
    # 4. 날씨 상호작용 피처
    df['temp_x_hour'] = df['기온(°C)'] * df['hour']
    df['humidity_x_hour'] = df['습도(%)'] * df['hour']
    df['temp_x_humidity'] = df['기온(°C)'] * df['습도(%)']
    
    # 5. 날씨 이동 통계 (6시간 윈도우)
    df['temp_rolling_mean_6'] = df.groupby('건물번호')['기온(°C)'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean())
    df['temp_rolling_std_6'] = df.groupby('건물번호')['기온(°C)'].transform(
        lambda x: x.rolling(window=6, min_periods=1).std()).fillna(0)
    df['humidity_rolling_mean_6'] = df.groupby('건물번호')['습도(%)'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean())
    
    # 6. 시차 변수 (Data Leakage 방지)
    if '전력소비량(kWh)' in df.columns:
        lags = [24, 48, 168]  # 1일, 2일, 1주일
        for lag in lags:
            df[f'power_lag_{lag}'] = df.groupby('건물번호')['전력소비량(kWh)'].transform(
                lambda x: x.shift(lag))
            df[f'temp_lag_{lag}'] = df.groupby('건물번호')['기온(°C)'].transform(
                lambda x: x.shift(lag))
    
    # 7. 건물 면적 비율
    df['area_ratio'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1e-6)
    
    return df

def main():
    print("🚀 최종 개선: test.py 참고한 체계적 접근법")
    print("=" * 60)
    
    # 1. 데이터 로딩
    print("📊 데이터 로딩 중...")
    train_df = pd.read_csv('data/train.csv', parse_dates=['일시'])
    test_df = pd.read_csv('data/test.csv', parse_dates=['일시'])
    building_info_df = pd.read_csv('data/building_info.csv')
    print("✅ 데이터 로딩 완료")

    # 2. 건물 정보 전처리
    print("🔧 데이터 전처리 중...")
    numeric_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in numeric_cols:
        building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)

    # 3. 데이터 병합
    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')
    
    # test 데이터의 전력소비량은 NaN으로 초기화
    test_df['전력소비량(kWh)'] = np.nan
    
    # 전체 데이터 통합하여 피처 엔지니어링
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print("✅ 데이터 전처리 완료")

    # 4. 피처 엔지니어링
    print("⚙️ 피처 엔지니어링 중...")
    combined_df = create_features(combined_df)
    
    # 범주형 변수 처리
    combined_df['건물번호'] = combined_df['건물번호'].astype('category')
    combined_df['건물유형'] = combined_df['건물유형'].astype('category')
    
    # 학습/테스트 데이터 분리
    train_processed_df = combined_df[~combined_df['전력소비량(kWh)'].isna()].copy()
    test_processed_df = combined_df[combined_df['전력소비량(kWh)'].isna()].copy()
    print("✅ 피처 엔지니어링 완료")

    # 5. 피처 리스트 정의 (test.py 스타일)
    features = [
        '건물번호', '기온(°C)', '풍속(m/s)', '습도(%)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        'month', 'day', 'hour', 'weekday', 'is_weekend', 'is_holiday',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
        'THI', 'temp_x_hour', 'humidity_x_hour', 'temp_x_humidity', 'area_ratio',
        'temp_rolling_mean_6', 'temp_rolling_std_6', 'humidity_rolling_mean_6',
        'power_lag_24', 'temp_lag_24', 'power_lag_48', 'temp_lag_48', 'power_lag_168', 'temp_lag_168'
    ]
    
    categorical_features_for_model = ['건물번호']
    
    print(f"📈 사용할 피처 수: {len(features)}개")
    print(f"📊 학습 데이터: {train_processed_df.shape}")
    print(f"🏢 건물 유형: {train_processed_df['건물유형'].nunique()}개")

    # 6. 건물 유형별 모델 학습 및 검증
    print("\n🏗️ 건물 유형별 모델 학습 시작...")
    
    building_types = train_processed_df['건물유형'].unique()
    total_predictions = []
    total_smape_score = 0
    building_results = []
    
    for b_type in building_types:
        print(f"\n🏢 {b_type} 유형 모델 학습...")
        
        type_train_df = train_processed_df[train_processed_df['건물유형'] == b_type].copy()
        
        # 시간 기반 분할 (test.py 방식)
        split_date = pd.to_datetime('2024-08-18 00:00:00')
        train_val_df = type_train_df[type_train_df['일시'] < split_date]
        valid_df = type_train_df[type_train_df['일시'] >= split_date]
        
        if len(valid_df) == 0:
            print(f"   ⚠️ 검증 데이터 없음 - 전체 데이터로 학습")
            train_val_df = type_train_df
            valid_df = type_train_df.tail(100)  # 임시 검증용

        X_train_val = train_val_df[features]
        y_train_val = train_val_df['전력소비량(kWh)']
        X_valid = valid_df[features]
        y_valid = valid_df['전력소비량(kWh)']
        
        # XGBoost를 위해 범주형 데이터 처리
        X_train_val_xgb = X_train_val.copy()
        X_valid_xgb = X_valid.copy()
        X_train_val_xgb['건물번호'] = X_train_val_xgb['건물번호'].cat.codes
        X_valid_xgb['건물번호'] = X_valid_xgb['건물번호'].cat.codes
        
        print(f"   📊 학습: {len(train_val_df)}개, 검증: {len(valid_df)}개")

        # XGBoost 파라미터 (메인 모델)
        xgb_params = {
            'objective': 'reg:absoluteerror',
            'random_state': 42,
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0
        }
        
        # LightGBM 파라미터 (앙상블용)
        lgb_params = {
            'objective': 'regression_l1',
            'random_state': 42,
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'num_leaves': 32,
            'max_depth': 8,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }
        
        # 1. XGBoost 모델 학습 및 검증
        print(f"   🔹 XGBoost 모델 학습...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        
        # XGBoost 훈련 (early stopping 없이)
        xgb_model.fit(X_train_val_xgb, y_train_val)
        
        xgb_best_iter = xgb_model.n_estimators  # 전체 반복 사용
        xgb_valid_preds = xgb_model.predict(X_valid_xgb)
        xgb_valid_preds = np.maximum(xgb_valid_preds, 0)
        xgb_smape = smape(y_valid.values, xgb_valid_preds)
        
        # 2. LightGBM 모델 학습 및 검증
        print(f"   🔹 LightGBM 모델 학습...")
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        
        lgb_model.fit(
            X_train_val, y_train_val, 
            eval_set=[(X_valid, y_valid)], 
            callbacks=[lgb.early_stopping(100, verbose=False)], 
            categorical_feature=categorical_features_for_model
        )
        
        lgb_best_iter = lgb_model.best_iteration_
        lgb_valid_preds = lgb_model.predict(X_valid, num_iteration=lgb_best_iter)
        lgb_valid_preds = np.maximum(lgb_valid_preds, 0)
        lgb_smape = smape(y_valid.values, lgb_valid_preds)
        
        # 3. 앙상블 예측 (XGBoost 70% + LightGBM 30%)
        ensemble_preds = 0.7 * xgb_valid_preds + 0.3 * lgb_valid_preds
        ensemble_smape = smape(y_valid.values, ensemble_preds)
        
        # 최고 성능 모델 선택
        best_smape = min(xgb_smape, lgb_smape, ensemble_smape)
        if best_smape == xgb_smape:
            best_model_name = "XGBoost"
            best_iter = xgb_best_iter
            type_smape = xgb_smape
        elif best_smape == lgb_smape:
            best_model_name = "LightGBM"
            best_iter = lgb_best_iter
            type_smape = lgb_smape
        else:
            best_model_name = "Ensemble"
            best_iter = max(xgb_best_iter, lgb_best_iter)
            type_smape = ensemble_smape
        
        total_smape_score += type_smape
        building_results.append((b_type, type_smape, best_iter, len(type_train_df), best_model_name))
        
        print(f"   📊 XGBoost SMAPE: {xgb_smape:.2f}% (반복: {xgb_best_iter})")
        print(f"   📊 LightGBM SMAPE: {lgb_smape:.2f}% (반복: {lgb_best_iter})")
        print(f"   📊 Ensemble SMAPE: {ensemble_smape:.2f}%")
        print(f"   🏆 최고 성능: {best_model_name} ({type_smape:.2f}%)")

        # 전체 데이터로 최종 모델 학습
        X_train_full = type_train_df[features]
        y_train_full = type_train_df['전력소비량(kWh)']
        
        # XGBoost용 데이터 준비
        X_train_full_xgb = X_train_full.copy()
        X_train_full_xgb['건물번호'] = X_train_full_xgb['건물번호'].cat.codes
        
        # XGBoost 최종 모델
        final_xgb_params = xgb_params.copy()
        final_xgb_params['n_estimators'] = xgb_best_iter
        final_xgb_model = xgb.XGBRegressor(**final_xgb_params)
        final_xgb_model.fit(X_train_full_xgb, y_train_full)
        
        # LightGBM 최종 모델
        final_lgb_params = lgb_params.copy()
        final_lgb_params['n_estimators'] = lgb_best_iter
        final_lgb_model = lgb.LGBMRegressor(**final_lgb_params)
        final_lgb_model.fit(X_train_full, y_train_full, categorical_feature=categorical_features_for_model)
        
        # 테스트 예측
        type_test_df = test_processed_df[test_processed_df['건물유형'] == b_type]
        
        if not type_test_df.empty:
            X_test = type_test_df[features]
            
            # XGBoost용 테스트 데이터 준비
            X_test_xgb = X_test.copy()
            X_test_xgb['건물번호'] = X_test_xgb['건물번호'].cat.codes
            
            # XGBoost 예측
            xgb_test_preds = final_xgb_model.predict(X_test_xgb)
            xgb_test_preds = np.maximum(xgb_test_preds, 0)
            
            # LightGBM 예측
            lgb_test_preds = final_lgb_model.predict(X_test)
            lgb_test_preds = np.maximum(lgb_test_preds, 0)
            
            # 앙상블 예측 (검증에서 최고 성능이었던 방식 사용)
            if best_model_name == "XGBoost":
                final_preds = xgb_test_preds
            elif best_model_name == "LightGBM":
                final_preds = lgb_test_preds
            else:  # Ensemble
                final_preds = 0.7 * xgb_test_preds + 0.3 * lgb_test_preds
            
            temp_submission = pd.DataFrame({
                'num_date_time': type_test_df['num_date_time'], 
                'answer': final_preds
            })
            total_predictions.append(temp_submission)
            print(f"   📤 테스트 예측: {len(final_preds)}개 ({best_model_name} 방식)")

    # 7. 결과 취합
    print(f"\n🎯 건물별 성능 결과:")
    print("=" * 80)
    for b_type, smape_score, best_iter, data_count, best_model in building_results:
        print(f"   {b_type:20s}: SMAPE {smape_score:6.2f}% (데이터: {data_count:5d}개, 반복: {best_iter:4d}, 모델: {best_model})")
    
    avg_smape = total_smape_score / len(building_types)
    print("=" * 80)
    print(f"🏆 평균 검증 SMAPE: {avg_smape:.2f}%")
    print("=" * 80)

    # 8. 최종 제출 파일 생성
    print("\n📋 제출 파일 생성 중...")
    final_submission = pd.concat(total_predictions, ignore_index=True)
    
    # sample_submission과 맞춤
    sample_submission = pd.read_csv('data/sample_submission.csv')
    sample_submission = sample_submission.drop(columns=['answer'])
    final_submission = pd.merge(sample_submission, final_submission, on='num_date_time', how='left')
    
    # 결측값 처리 (혹시 누락된 데이터가 있다면)
    if final_submission['answer'].isna().sum() > 0:
        print(f"   ⚠️ 결측값 {final_submission['answer'].isna().sum()}개 발견 - 평균값으로 대체")
        final_submission['answer'].fillna(final_submission['answer'].mean(), inplace=True)
    
    final_submission.to_csv('submission_final.csv', index=False)
    
    print(f"✅ submission_final.csv 저장 완료! ({len(final_submission)}행)")
    print(f"📊 예측값 범위: {final_submission['answer'].min():.1f} ~ {final_submission['answer'].max():.1f}")
    print(f"📈 예측값 평균: {final_submission['answer'].mean():.1f}")
    
    # 개선 효과 요약
    print(f"\n📊 최종 개선 효과:")
    print(f"   🔹 피처 수: {len(features)}개 (적절한 수준)")
    print(f"   🔹 한국 공휴일: 적용됨")
    print(f"   🔹 THI 불쾌지수: 적용됨")
    print(f"   🔹 건물별 개별 모델: {len(building_types)}개 유형")
    print(f"   🔹 시간 기반 검증: 적용됨")
    print(f"   🔹 Early Stopping: 적용됨")

if __name__ == "__main__":
    main() 