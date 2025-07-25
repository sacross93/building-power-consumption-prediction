import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import os
import json
from datetime import datetime

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

print("="*60)
print("🚀 XGBoost Model Training for Power Consumption Prediction")
print("="*60)

def smape(y_true, y_pred):
    """SMAPE 계산 함수"""
    epsilon = 1e-10
    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return np.mean(numerator / denominator) * 100

def create_features(df):
    """피처 엔지니어링 수행"""
    print("🔧 피처 엔지니어링 수행 중...")
    
    df = df.sort_values(by=['건물번호', '일시']).reset_index(drop=True)
    
    # 1. 시간 관련 피처
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 2. 공휴일 피처 추가
    kr_holidays = holidays.KR()
    df['is_holiday'] = df['일시'].dt.date.apply(lambda x: 1 if x in kr_holidays else 0).astype(int)
    
    # 3. 날씨 관련 피처 (THI - Temperature Humidity Index)
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
    
    # 6. 계절성 피처 추가
    df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # 겨울
                                    3: 1, 4: 1, 5: 1,    # 봄
                                    6: 2, 7: 2, 8: 2,    # 여름
                                    9: 3, 10: 3, 11: 3}) # 가을
    
    # 7. 더 많은 시간 피처
    df['dayofyear'] = df['일시'].dt.dayofyear
    df['week_of_year'] = df['일시'].dt.isocalendar().week
    df['is_month_start'] = df['일시'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['일시'].dt.is_month_end.astype(int)
    
    print(f"✅ 피처 엔지니어링 완료 - 총 {len(df.columns)}개 컬럼")
    return df

def prepare_data():
    """데이터 로딩 및 전처리"""
    print("\n📊 데이터 로딩 및 전처리 시작...")
    
    # 데이터 로드
    train_df = pd.read_csv('data/train.csv', parse_dates=['일시'])
    test_df = pd.read_csv('data/test.csv', parse_dates=['일시'])
    building_info_df = pd.read_csv('data/building_info.csv')
    
    print(f"✅ 훈련 데이터: {train_df.shape}")
    print(f"✅ 테스트 데이터: {test_df.shape}")
    print(f"✅ 건물 정보: {building_info_df.shape}")

    # 건물 정보 수치형 변환
    numeric_cols = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in numeric_cols:
        building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)

    # 건물 정보 병합
    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')
    
    # 테스트 데이터에 더미 타겟 변수 추가
    test_df['전력소비량(kWh)'] = np.nan
    
    # 전체 데이터 결합하여 피처 엔지니어링
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = create_features(combined_df)
    
    # 카테고리컬 변수 처리 - XGBoost를 위해 수치형으로 변환
    from sklearn.preprocessing import LabelEncoder
    
    # 건물번호를 수치형으로 변환
    le_building = LabelEncoder()
    combined_df['건물번호'] = le_building.fit_transform(combined_df['건물번호'])
    
    # 건물유형은 문자열로 유지 (나중에 필터링용으로 사용)
    combined_df['건물유형'] = combined_df['건물유형'].astype(str)
    
    # 훈련/테스트 데이터 분리
    train_processed_df = combined_df[~combined_df['전력소비량(kWh)'].isna()].copy()
    test_processed_df = combined_df[combined_df['전력소비량(kWh)'].isna()].copy()
    
    print(f"✅ 전처리 완료 - 훈련: {train_processed_df.shape}, 테스트: {test_processed_df.shape}")
    
    return train_processed_df, test_processed_df

def get_feature_list():
    """사용할 피처 리스트 반환"""
    features = [
        '건물번호', '기온(°C)', '풍속(m/s)', '습도(%)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        'month', 'day', 'hour', 'weekday', 'is_weekend', 'is_holiday',
        'hour_sin', 'hour_cos', 'THI', 'temp_x_hour', 
        'temp_rolling_mean_6', 'temp_rolling_std_6',
        'power_lag_24', 'temp_lag_24', 'power_lag_48', 'temp_lag_48', 'power_lag_168', 'temp_lag_168',
        'season', 'dayofyear', 'week_of_year', 'is_month_start', 'is_month_end'
    ]
    return features

def train_building_type_models(train_df, test_df):
    """건물 유형별 XGBoost 모델 훈련"""
    print("\n🏗️ 건물 유형별 XGBoost 모델 훈련 시작...")
    
    features = get_feature_list()
    
    building_types = train_df['건물유형'].unique()
    total_predictions = []
    total_smape_score = 0
    model_results = {}
    
    # 결과 저장 폴더 생성
    os.makedirs('result', exist_ok=True)
    
    for b_type in building_types:
        print(f"\n--- {b_type} 유형 모델 훈련 ---")
        
        # 건물 유형별 데이터 필터링
        type_train_df = train_df[train_df['건물유형'] == b_type].copy()
        
        # 시계열 분할 (검증용)
        split_date = pd.to_datetime('2024-08-18 00:00:00')
        train_val_df = type_train_df[type_train_df['일시'] < split_date]
        valid_df = type_train_df[type_train_df['일시'] >= split_date]

        X_train_val = train_val_df[features]
        y_train_val = train_val_df['전력소비량(kWh)']
        X_valid = valid_df[features]
        y_valid = valid_df['전력소비량(kWh)']

        # XGBoost 파라미터 설정
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': 2000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        # 모델 훈련 (검증용)
        xgb_params_with_early_stop = xgb_params.copy()
        xgb_params_with_early_stop['early_stopping_rounds'] = 100
        
        xgb_model = XGBRegressor(**xgb_params_with_early_stop)
        xgb_model.fit(
            X_train_val, y_train_val,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        # 검증 성능 평가
        best_iter = xgb_model.get_booster().best_iteration
        valid_preds = xgb_model.predict(X_valid)
        valid_preds[valid_preds < 0] = 0
        type_smape = smape(y_valid.values, valid_preds)
        total_smape_score += type_smape
        
        print(f"✅ {b_type} 검증 SMAPE: {type_smape:.4f} (최적 반복: {best_iter})")

        # 전체 데이터로 최종 모델 훈련
        X_train_full = type_train_df[features]
        y_train_full = type_train_df['전력소비량(kWh)']
        
        # 최적 반복수로 재훈련
        final_xgb_params = xgb_params.copy()
        final_xgb_params['n_estimators'] = best_iter + 50  # 여유분 추가
        
        final_model = XGBRegressor(**final_xgb_params)
        final_model.fit(X_train_full, y_train_full)
        
        # 모델 저장
        model_path = f'result/xgboost_model_{b_type}.pkl'
        joblib.dump(final_model, model_path)
        
        # 해당 건물 유형의 테스트 데이터 예측
        type_test_df = test_df[test_df['건물유형'] == b_type]
        
        if not type_test_df.empty:
            X_test = type_test_df[features]
            preds = final_model.predict(X_test)
            preds[preds < 0] = 0  # 음수 예측값 제거
            
            temp_submission = pd.DataFrame({
                'num_date_time': type_test_df['num_date_time'], 
                'answer': preds
            })
            total_predictions.append(temp_submission)
            
            print(f"✅ {b_type} 예측 완료: {len(preds)}개 샘플")
        
        # 모델 결과 저장
        model_results[b_type] = {
            'smape': type_smape,
            'best_iteration': best_iter,
            'train_samples': len(X_train_full),
            'test_samples': len(type_test_df) if not type_test_df.empty else 0
        }

    print(f"\n📊 전체 평균 검증 SMAPE: {total_smape_score / len(building_types):.4f}")
    
    return total_predictions, model_results

def create_submission_file(predictions, model_results):
    """제출 파일 생성"""
    print("\n📄 제출 파일 생성 중...")
    
    # 모든 예측 결과 결합
    final_submission = pd.concat(predictions, ignore_index=True)
    
    # sample_submission과 형식 맞추기
    sample_submission = pd.read_csv('data/sample_submission.csv')
    sample_submission = sample_submission.drop(columns=['answer'])
    final_submission = pd.merge(sample_submission, final_submission, on='num_date_time', how='left')
    
    # 누락된 값 확인 및 처리
    if final_submission['answer'].isna().sum() > 0:
        print(f"⚠️ 누락된 예측값 {final_submission['answer'].isna().sum()}개를 평균값으로 대체")
        final_submission['answer'].fillna(final_submission['answer'].mean(), inplace=True)
    
    # 제출 파일 저장
    submission_path = 'result/submission.csv'
    final_submission.to_csv(submission_path, index=False)
    
    print(f"✅ 제출 파일 저장 완료: {submission_path}")
    print(f"✅ 제출 파일 크기: {final_submission.shape}")
    print(f"✅ 예측값 범위: {final_submission['answer'].min():.2f} ~ {final_submission['answer'].max():.2f}")
    
    return final_submission

def create_analysis_report(model_results, submission_df):
    """분석 리포트 생성"""
    print("\n📊 분석 리포트 생성 중...")
    
    # 모델 성능 요약
    report = {
        'model_summary': {
            'total_building_types': len(model_results),
            'average_smape': np.mean([result['smape'] for result in model_results.values()]),
            'model_details': model_results
        },
        'prediction_summary': {
            'total_predictions': len(submission_df),
            'prediction_stats': {
                'min': float(submission_df['answer'].min()),
                'max': float(submission_df['answer'].max()),
                'mean': float(submission_df['answer'].mean()),
                'std': float(submission_df['answer'].std())
            }
        },
        'training_info': {
            'model_type': 'XGBoost',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_used': get_feature_list()
        }
    }
    
    # JSON 리포트 저장
    report_path = 'result/analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 분석 리포트 저장: {report_path}")
    
    # 시각화 생성
    create_visualizations(model_results, submission_df)
    
    return report

def create_visualizations(model_results, submission_df):
    """시각화 생성"""
    print("🎨 시각화 생성 중...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 건물 유형별 SMAPE 성능
    plt.subplot(2, 3, 1)
    building_types = list(model_results.keys())
    smape_scores = [model_results[bt]['smape'] for bt in building_types]
    
    plt.bar(range(len(building_types)), smape_scores, color='skyblue', alpha=0.7)
    plt.xlabel('Building Type')
    plt.ylabel('SMAPE')
    plt.title('SMAPE by Building Type')
    plt.xticks(range(len(building_types)), [bt[:10] for bt in building_types], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 예측값 분포
    plt.subplot(2, 3, 2)
    plt.hist(submission_df['answer'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Power Consumption (kWh)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.grid(True, alpha=0.3)
    
    # 3. 건물 유형별 훈련 샘플 수
    plt.subplot(2, 3, 3)
    train_samples = [model_results[bt]['train_samples'] for bt in building_types]
    plt.bar(range(len(building_types)), train_samples, color='lightgreen', alpha=0.7)
    plt.xlabel('Building Type')
    plt.ylabel('Training Samples')
    plt.title('Training Samples by Building Type')
    plt.xticks(range(len(building_types)), [bt[:10] for bt in building_types], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. 모델 성능 요약
    plt.subplot(2, 3, 4)
    avg_smape = np.mean(smape_scores)
    plt.text(0.1, 0.8, f'Average SMAPE: {avg_smape:.4f}', fontsize=14, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Best SMAPE: {min(smape_scores):.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Worst SMAPE: {max(smape_scores):.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Total Models: {len(building_types)}', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Model Performance Summary')
    plt.axis('off')
    
    # 5. 예측값 통계
    plt.subplot(2, 3, 5)
    stats = submission_df['answer'].describe()
    y_pos = np.arange(len(stats))
    plt.barh(y_pos, stats.values, color='coral', alpha=0.7)
    plt.yticks(y_pos, [f'{stat}: {val:.2f}' for stat, val in zip(stats.index, stats.values)])
    plt.xlabel('Value')
    plt.title('Prediction Statistics')
    plt.grid(True, alpha=0.3)
    
    # 6. 건물 유형별 예측 개수
    plt.subplot(2, 3, 6)
    test_samples = [model_results[bt]['test_samples'] for bt in building_types]
    plt.pie(test_samples, labels=[bt[:10] for bt in building_types], autopct='%1.1f%%', startangle=90)
    plt.title('Test Samples Distribution')
    
    plt.tight_layout()
    
    # 시각화 저장
    viz_path = 'result/model_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 시각화 저장: {viz_path}")

def main():
    """메인 실행 함수"""
    try:
        print("🏃‍♂️ XGBoost 전력소비량 예측 모델 시작!")
        
        # 1. 데이터 준비
        train_df, test_df = prepare_data()
        
        # 2. 건물 유형별 모델 훈련
        predictions, model_results = train_building_type_models(train_df, test_df)
        
        # 3. 제출 파일 생성
        submission_df = create_submission_file(predictions, model_results)
        
        # 4. 분석 리포트 생성
        report = create_analysis_report(model_results, submission_df)
        
        # 최종 결과 출력
        print("\n" + "="*60)
        print("🎉 XGBoost 모델링 완료!")
        print("="*60)
        print(f"📊 결과 요약:")
        print(f"   • 건물 유형 수: {len(model_results)}")
        print(f"   • 평균 SMAPE: {report['model_summary']['average_smape']:.4f}")
        print(f"   • 총 예측 샘플: {len(submission_df)}")
        print(f"")
        print(f"📁 생성된 파일들:")
        print(f"   • result/submission.csv (제출 파일)")
        print(f"   • result/analysis_report.json (분석 리포트)")
        print(f"   • result/model_analysis.png (시각화)")
        print(f"   • result/xgboost_model_*.pkl (건물별 모델)")
        print(f"")
        print("🚀 submission.csv 파일이 준비되었습니다!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 