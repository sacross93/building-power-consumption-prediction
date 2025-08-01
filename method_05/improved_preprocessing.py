"""
Improved Data Preprocessing Based on Visualization Analysis
==========================================================

시각화 분석 결과를 바탕으로 개선된 전처리 파이프라인:
1. 타겟 변수 log 변환
2. VIF 기반 다중공선성 제거
3. 피처 스케일링
4. 이상값 처리
5. 고급 피처 엔지니어링
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features


class ImprovedPreprocessor:
    """개선된 전처리 클래스."""
    
    def __init__(self, scaler_type='standard'):
        """
        Parameters:
        -----------
        scaler_type : str
            'standard' (기본값), 'robust', 'minmax', 'quantile' 중 선택
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.selected_features = None
        self.outlier_bounds = {}
        self.target_transformer = None
        
    def detect_outliers_iqr(self, data, column, factor=1.5):
        """IQR 기반 이상값 탐지."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        return outliers, lower_bound, upper_bound
    
    def remove_multicollinearity(self, X, threshold=10.0):
        """VIF 기반 다중공선성 제거."""
        print(f"Removing multicollinearity (VIF > {threshold})...")
        
        # 수치형 변수만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].copy()
        
        # 무한값이나 NaN 제거
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        
        if X_numeric.empty:
            print("No valid numeric columns for VIF analysis")
            return X.columns.tolist()
        
        # 상수 컬럼 제거
        constant_cols = [col for col in X_numeric.columns if X_numeric[col].var() == 0]
        X_numeric = X_numeric.drop(columns=constant_cols)
        
        if X_numeric.empty:
            print("No non-constant columns for VIF analysis")
            return X.columns.tolist()
        
        features_to_keep = list(X_numeric.columns)
        
        while True:
            if len(features_to_keep) <= 1:
                break
                
            try:
                # 숫자형 데이터만 필터링하고 무한값/NaN 처리
                X_vif = X_numeric[features_to_keep].copy()
                
                # 무한값과 NaN을 0으로 대체
                X_vif = X_vif.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 상수 컬럼 제거 (분산이 0인 컬럼)
                constant_cols = X_vif.columns[X_vif.var() == 0].tolist()
                if constant_cols:
                    features_to_keep = [f for f in features_to_keep if f not in constant_cols]
                    print(f"Removed constant columns: {constant_cols}")
                    continue
                
                # VIF 계산
                vif_df = pd.DataFrame()
                vif_df["Feature"] = features_to_keep
                vif_df["VIF"] = [
                    variance_inflation_factor(X_vif.values, i) 
                    for i in range(len(features_to_keep))
                ]
                
                # 무한값 VIF 처리
                vif_df["VIF"] = vif_df["VIF"].replace([np.inf, -np.inf], 999999)
                
                # 최고 VIF가 임계값보다 낮으면 중단
                max_vif = vif_df["VIF"].max()
                if max_vif <= threshold:
                    break
                
                # 가장 높은 VIF를 가진 피처 제거
                feature_to_remove = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
                features_to_keep.remove(feature_to_remove)
                print(f"Removed {feature_to_remove} (VIF: {max_vif:.2f})")
                
            except Exception as e:
                print(f"VIF calculation error: {e}")
                break
        
        # 비수치형 컬럼들도 포함
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        final_features = features_to_keep + non_numeric_cols
        
        print(f"Features after multicollinearity removal: {len(final_features)}")
        return final_features
    
    def transform_target(self, y, method='log1p'):
        """타겟 변수 변환."""
        print(f"Applying {method} transformation to target variable...")
        
        if method == 'log1p':
            y_transformed = np.log1p(y)
        elif method == 'sqrt':
            y_transformed = np.sqrt(np.maximum(y, 0))
        else:
            y_transformed = y
            
        self.target_transformer = method
        return y_transformed
    
    def inverse_transform_target(self, y_transformed):
        """타겟 변수 역변환."""
        if self.target_transformer == 'log1p':
            return np.expm1(y_transformed)
        elif self.target_transformer == 'sqrt':
            return np.power(y_transformed, 2)
        else:
            return y_transformed
    
    def create_advanced_features(self, df):
        """고급 피처 엔지니어링."""
        print("Creating advanced features...")
        df = df.copy()
        
        # 온도 degree-day 계산
        if 'temp' in df.columns:
            # Cooling Degree Days (기준: 18°C)
            df['cdd'] = np.maximum(df['temp'] - 18, 0)
            # Heating Degree Days (기준: 18°C)  
            df['hdd'] = np.maximum(18 - df['temp'], 0)
            
            # 온도 변동성 (일일 범위 추정)
            df['temp_volatility'] = np.abs(df['temp'] - df['temp'].rolling(24, min_periods=1).mean())
            
            # 계절별 온도 편차
            seasonal_temp = df.groupby('month')['temp'].transform('mean')
            df['temp_seasonal_deviation'] = df['temp'] - seasonal_temp
        
        # 습도 관련 고급 피처
        if 'humidity' in df.columns and 'temp' in df.columns:
            # 불쾌지수 (실제 공식)
            df['heat_index'] = df['temp'] + 0.348 * df['humidity'] - 42.379
            
            # 습도 구간별 인디케이터
            df['humidity_very_low'] = (df['humidity'] < 20).astype(int)
            df['humidity_optimal'] = ((df['humidity'] >= 40) & (df['humidity'] <= 60)).astype(int)
            df['humidity_very_high'] = (df['humidity'] > 80).astype(int)
        
        # 건물 효율성 고급 피처
        if 'total_area' in df.columns and 'pv_capacity' in df.columns:
            # 건물 에너지 효율성 점수
            df['energy_efficiency_score'] = (
                df['pv_capacity'] / np.maximum(df['total_area'], 1) * 1000 +
                df['area_ratio'] * 50
            )
            
            # 건물 크기별 카테고리 (더 세분화)
            df['building_size_category'] = pd.cut(
                df['total_area'], 
                bins=[0, 1000, 5000, 20000, 50000, np.inf],
                labels=['very_small', 'small', 'medium', 'large', 'very_large']
            )
        
        # 시간 기반 고급 피처
        if 'datetime' in df.columns:
            # 한국 공휴일 (간단한 버전)
            df['is_holiday'] = (
                ((df['month'] == 8) & (df['day'] == 15)) |  # 광복절
                ((df['month'] == 6) & (df['day'] == 6))     # 현충일
            ).astype(int)
            
            # 계절 구분 (더 정확한)
            df['season_detailed'] = pd.cut(
                df['month'], 
                bins=[0, 2, 5, 8, 11, 12],
                labels=['winter', 'spring', 'summer', 'fall', 'winter2'],
                ordered=False
            )
            
            # 주차 정보
            df['week_of_year'] = df['datetime'].dt.isocalendar().week
            
            # 월 내 주차
            df['week_of_month'] = (df['day'] - 1) // 7 + 1
        
        # 건물 타입별 특화 피처
        if 'building_type' in df.columns:
            # IDC 건물의 야간 특성
            df['idc_night_operation'] = (
                (df['building_type'] == 'IDC(전화국)') & 
                ((df['hour'] < 6) | (df['hour'] > 22))
            ).astype(int)
            
            # 백화점의 영업시간 특성
            df['department_store_hours'] = (
                (df['building_type'] == '백화점') &
                (df['hour'].between(10, 21)) &
                (df['weekday'] < 5)
            ).astype(int)
            
            # 학교의 방학 기간 (여름방학 추정)
            df['school_vacation'] = (
                (df['building_type'] == '학교') &
                (df['month'].isin([7, 8]))
            ).astype(int)
        
        print(f"Advanced features created. New shape: {df.shape}")
        return df
    
    def fit_transform(self, train_df, test_df, target_col='전력소비량(kWh)'):
        """전체 전처리 파이프라인 적용."""
        print("=" * 80)
        print("IMPROVED PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # 1. 기본 피처 엔지니어링 (기존 solution.py) - 이미 적용된 경우 건너뛰기
        print("1. Basic feature engineering...")
        if 'sunshine_est' in train_df.columns:
            # 이미 고급 피처 엔지니어링이 적용된 데이터
            print("   Already processed data detected, skipping basic FE...")
            train_processed, test_processed = train_df.copy(), test_df.copy()
        else:
            # 원본 데이터, 기본 피처 엔지니어링 적용
            train_processed, test_processed = engineer_features(train_df.copy(), test_df.copy())
        
        # 2. 고급 피처 엔지니어링
        print("2. Advanced feature engineering...")
        train_processed = self.create_advanced_features(train_processed)
        test_processed = self.create_advanced_features(test_processed)
        
        # 3. 타겟 변수 이상값 탐지 및 처리
        print("3. Target outlier detection...")
        if target_col in train_processed.columns:
            outliers, lower, upper = self.detect_outliers_iqr(train_processed, target_col, factor=2.0)
            print(f"Found {outliers.sum()} outliers in target variable")
            print(f"Outlier bounds: {lower:.2f} - {upper:.2f}")
            
            # 이상값 캡핑 (완전 제거보다는 캡핑)
            train_processed[target_col] = np.clip(
                train_processed[target_col], 
                lower, upper
            )
            
            self.outlier_bounds[target_col] = (lower, upper)
        
        # 4. 타겟 변수 변환
        print("4. Target transformation...")
        if target_col in train_processed.columns:
            y_original = train_processed[target_col].copy()
            y_transformed = self.transform_target(y_original, method='log1p')
            train_processed[f'{target_col}_log'] = y_transformed
            
            # 변환 효과 출력
            print(f"Original target - Mean: {y_original.mean():.2f}, Std: {y_original.std():.2f}")
            print(f"Transformed target - Mean: {y_transformed.mean():.2f}, Std: {y_transformed.std():.2f}")
        
        # 5. 피처 준비
        print("5. Feature preparation...")
        drop_cols = [target_col, f'{target_col}_log', '일시', 'num_date_time', 'datetime']
        feature_cols = [c for c in train_processed.columns if c not in drop_cols and c in test_processed.columns]
        
        X_train = train_processed[feature_cols].copy()
        X_test = test_processed[feature_cols].copy()
        
        # 6. 카테고리 변수 처리
        print("6. Categorical encoding...")
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # 라벨 인코딩
            all_categories = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
            category_map = {cat: idx for idx, cat in enumerate(all_categories)}
            
            X_train[col] = X_train[col].astype(str).map(category_map).fillna(-1)
            X_test[col] = X_test[col].astype(str).map(category_map).fillna(-1)
        
        # 7. 다중공선성 제거
        print("7. Multicollinearity removal...")
        selected_features = self.remove_multicollinearity(X_train, threshold=10.0)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        self.selected_features = selected_features
        
        # 8. 피처 스케일링
        print("8. Feature scaling...")
        scaler_map = {
            'standard': StandardScaler(),
            'robust': RobustScaler(), 
            'minmax': MinMaxScaler(),
            'quantile': QuantileTransformer(n_quantiles=100, random_state=42)
        }
        
        self.scaler = scaler_map.get(self.scaler_type, StandardScaler())
        print(f"   Using {self.scaler.__class__.__name__}...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
        
        # 결과 출력
        print(f"\n✅ Preprocessing completed!")
        print(f"Original features: {len(feature_cols)}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Final train shape: {X_train_scaled.shape}")
        print(f"Final test shape: {X_test_scaled.shape}")
        
        # 타겟 변수 반환
        if target_col in train_processed.columns:
            return X_train_scaled, X_test_scaled, train_processed[f'{target_col}_log']
        else:
            return X_train_scaled, X_test_scaled, None
    
    def create_preprocessing_report(self, output_dir='./visualizations/'):
        """전처리 개선 결과 리포트 생성."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / 'improved_preprocessing_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Improved Preprocessing Report\n\n")
            
            f.write("## Applied Improvements\n\n")
            
            f.write("### 1. Target Variable Transformation\n")
            f.write(f"- Method: {self.target_transformer}\n")
            f.write("- Reduces right-skewness for better model training\n\n")
            
            if self.outlier_bounds:
                f.write("### 2. Outlier Treatment\n")
                for col, bounds in self.outlier_bounds.items():
                    f.write(f"- {col}: Clipped to [{bounds[0]:.2f}, {bounds[1]:.2f}]\n")
                f.write("\n")
            
            f.write("### 3. Multicollinearity Removal\n")
            if self.selected_features:
                f.write(f"- Features selected: {len(self.selected_features)}\n")
                f.write(f"- VIF threshold: 10.0\n\n")
            
            f.write("### 4. Feature Scaling\n")
            f.write(f"- Method: {type(self.scaler).__name__}\n")
            f.write("- Robust to outliers, normalizes feature scales\n\n")
            
            f.write("### 5. Advanced Feature Engineering\n")
            f.write("- Cooling/Heating Degree Days\n")
            f.write("- Heat Index calculation\n")
            f.write("- Building efficiency scores\n")
            f.write("- Holiday indicators\n")
            f.write("- Building-type specific features\n\n")
            
            f.write("## Expected Benefits\n\n")
            f.write("1. **Better model convergence** from target transformation\n")
            f.write("2. **Reduced overfitting** from multicollinearity removal\n")
            f.write("3. **Improved gradient flow** from feature scaling\n")
            f.write("4. **Enhanced feature richness** from advanced engineering\n")
            f.write("5. **More robust predictions** from outlier handling\n")
        
        print(f"📄 Preprocessing report saved: {report_path}")


def test_improved_preprocessing():
    """개선된 전처리 테스트."""
    print("Testing improved preprocessing pipeline...")
    
    # 데이터 로드
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 개선된 전처리 적용
    preprocessor = ImprovedPreprocessor()
    X_train, X_test, y_train = preprocessor.fit_transform(train_df, test_df)
    
    # 결과 분석
    print("\n" + "=" * 60)
    print("PREPROCESSING RESULTS ANALYSIS")
    print("=" * 60)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    if y_train is not None:
        print(f"Target variable shape: {y_train.shape}")
        print(f"Target stats - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    
    # 피처 중요도 분석 (간단한 상관관계)
    if y_train is not None:
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        print(f"\nTop 10 feature correlations with transformed target:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            print(f"{i+1:2d}. {feature}: {corr:.4f}")
    
    # 리포트 생성
    preprocessor.create_preprocessing_report()
    
    return X_train, X_test, y_train, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, preprocessor = test_improved_preprocessing()
    print("\n🎯 Improved preprocessing completed successfully!")