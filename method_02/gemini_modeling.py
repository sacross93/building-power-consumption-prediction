import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# SMAPE 계산 함수
def smape(y_true, y_pred):
    """ Symmetric Mean Absolute Percentage Error """
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def train_and_predict(preprocessed_path, data_dir, model_output_dir, submission_output_dir):
    """
    Trains a LightGBM model, evaluates it, and creates a submission file.
    """
    # 1. 데이터 로딩
    print("1. Loading data...")
    train_df = pd.read_csv(preprocessed_path)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    building_info_df = pd.read_csv(os.path.join(data_dir, 'building_info.csv'))
    submission_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    # 2. 테스트 데이터 전처리
    print("2. Preprocessing test data...")
    # 컬럼명 영문으로 변경 (test.csv는 sunshine, solar_radiation 컬럼이 없음)
    test_df.columns = ['num_date_time', 'building_number', 'datetime', 'temperature', 'precipitation', 
                       'windspeed', 'humidity']
    # 누락된 컬럼 추가 및 0으로 채우기
    test_df['sunshine'] = 0
    test_df['solar_radiation'] = 0
    building_info_df.columns = ['building_number', 'building_type', 'total_area', 'cooling_area', 
                                'solar_capacity', 'ess_capacity', 'pcs_capacity']

    # 타입 변환 및 정리
    test_df['datetime'] = pd.to_datetime(test_df['datetime'], format='%Y%m%d %H')
    for col in ['solar_capacity', 'ess_capacity', 'pcs_capacity']:
        building_info_df[col] = building_info_df[col].replace('-', 0).astype(float)

    # 데이터 병합
    test_df = pd.merge(test_df, building_info_df, on='building_number', how='left')

    # 피처 엔지니어링 (훈련 데이터와 동일하게)
    test_df['month'] = test_df['datetime'].dt.month
    test_df['day'] = test_df['datetime'].dt.day
    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['dayofweek'] = test_df['datetime'].dt.dayofweek
    test_df['is_weekend'] = test_df['dayofweek'].isin([5, 6]).astype(int)
    holidays = [pd.Timestamp('2024-06-06'), pd.Timestamp('2024-08-15')]
    test_df['is_holiday'] = test_df['datetime'].dt.date.isin([d.date() for d in holidays]).astype(int)
    test_df['discomfort_index'] = test_df['temperature'] - 0.55 * (1 - 0.01 * test_df['humidity']) * (test_df['temperature'] - 14.5)
    test_df = pd.get_dummies(test_df, columns=['building_type'], prefix='building')

    # 훈련 데이터와 테스트 데이터의 컬럼 맞추기
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        if col != 'power_consumption': # 타겟 변수는 제외
            test_df[col] = 0
    test_df = test_df[train_df.drop(columns=['power_consumption']).columns] # 순서 맞추기

    # 3. 모델 학습 및 평가
    print("3. Training and evaluating model with TimeSeriesSplit...")
    X = train_df.drop(columns=['power_consumption'])
    y = train_df['power_consumption']

    # LightGBM 파라미터 (GPU 사용)
    lgbm_params = {
        'objective': 'regression_l1', # MAE
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'device': 'gpu' # GPU 사용 설정
    }

    ts_cv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    smape_scores = []

    for fold, (train_index, val_index) in enumerate(ts_cv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        smape_val = smape(y_val, preds)
        rmse_scores.append(rmse)
        smape_scores.append(smape_val)
        print(f"Fold {fold+1} | RMSE: {rmse:.4f} | SMAPE: {smape_val:.4f}")

    print(f"\nAverage CV RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Average CV SMAPE: {np.mean(smape_scores):.4f}\n")

    # 4. 최종 모델 학습 (전체 데이터 사용)
    print("4. Training final model on all data...")
    final_model = lgb.LGBMRegressor(**lgbm_params)
    final_model.fit(X, y)

    # 5. 예측 및 제출 파일 생성
    print("5. Predicting on test data and creating submission file...")
    predictions = final_model.predict(test_df)
    predictions[predictions < 0] = 0 # 전력 사용량은 0보다 작을 수 없음

    submission_df['answer'] = predictions

    # 결과 폴더 생성
    if not os.path.exists(submission_output_dir):
        os.makedirs(submission_output_dir)
    
    submission_path = os.path.join(submission_output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

    # 6. 특성 중요도 시각화 및 저장
    print("6. Saving feature importance plot...")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plot_path = os.path.join(model_output_dir, 'feature_importance_gemini.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_file = os.path.join(current_dir, '../', 'preprocessed_data_gemini', 'preprocessed_gemini.csv')
    data_directory = os.path.join(current_dir, '../', 'data')
    model_results_dir = os.path.join(current_dir, '../', 'modeling_results')
    submission_results_dir = os.path.join(current_dir, '../', 'result_gemini')

    train_and_predict(preprocessed_file, data_directory, model_results_dir, submission_results_dir)
