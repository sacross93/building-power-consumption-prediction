"""
Importance Sampling: Train 샘플을 Test 분포에 맞춰 리웨이팅

Train 데이터의 각 샘플에 Test 분포와의 유사도에 기반한 가중치를 부여하여
Distribution shift 문제를 완화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ImportanceSampler:
    """Test 분포 기반 Train 샘플 리웨이팅"""
    
    def __init__(self, method: str = 'density_ratio', normalize: bool = True):
        """
        Args:
            method: 'density_ratio', 'distance_based', 'quantile_based'
            normalize: 가중치 정규화 여부
        """
        self.method = method
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, train_features: pd.DataFrame, test_features: pd.DataFrame,
            feature_cols: Optional[List[str]] = None) -> 'ImportanceSampler':
        """Train/Test 분포 학습"""
        
        if feature_cols is None:
            feature_cols = ['기온(°C)', '습도(%)', '풍속(m/s)']
            feature_cols = [col for col in feature_cols 
                          if col in train_features.columns and col in test_features.columns]
        
        self.feature_cols = feature_cols
        
        # 특징 추출 및 정규화
        X_train = train_features[feature_cols].dropna()
        X_test = test_features[feature_cols].dropna()
        
        # 결합하여 스케일러 학습
        X_combined = pd.concat([X_train, X_test], axis=0)
        self.scaler.fit(X_combined)
        
        self.X_train_scaled = self.scaler.transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        self.train_index = X_train.index
        self.test_index = X_test.index
        
        self.fitted = True
        return self
    
    def calculate_weights(self, train_features: pd.DataFrame) -> np.ndarray:
        """가중치 계산"""
        if not self.fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다.")
        
        X_train = train_features[self.feature_cols].dropna()
        X_train_scaled = self.scaler.transform(X_train)
        
        if self.method == 'density_ratio':
            weights = self._density_ratio_weights(X_train_scaled)
        elif self.method == 'distance_based':
            weights = self._distance_based_weights(X_train_scaled)
        elif self.method == 'quantile_based':
            weights = self._quantile_based_weights(X_train, train_features[self.feature_cols])
        else:
            raise ValueError(f"지원하지 않는 방법: {self.method}")
        
        # 정규화
        if self.normalize:
            weights = weights / weights.mean()
        
        # 극단값 클리핑
        weights = np.clip(weights, 0.1, 5.0)
        
        return weights
    
    def _density_ratio_weights(self, X_train_scaled: np.ndarray) -> np.ndarray:
        """밀도 비율 기반 가중치"""
        from sklearn.neighbors import KernelDensity
        
        # KDE로 Train/Test 밀도 추정
        bandwidth = 0.5
        
        kde_train = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_test = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        
        kde_train.fit(self.X_train_scaled)
        kde_test.fit(self.X_test_scaled)
        
        # 각 Train 샘플의 밀도 비율 계산
        log_density_train = kde_train.score_samples(X_train_scaled)
        log_density_test = kde_test.score_samples(X_train_scaled)
        
        # 비율 계산 (test density / train density)
        density_ratio = np.exp(log_density_test - log_density_train)
        
        return density_ratio
    
    def _distance_based_weights(self, X_train_scaled: np.ndarray) -> np.ndarray:
        """거리 기반 가중치"""
        # 각 Train 샘플에서 가장 가까운 Test 샘플까지의 거리
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs.fit(self.X_test_scaled)
        
        distances, _ = nbrs.kneighbors(X_train_scaled)
        distances = distances.flatten()
        
        # 거리를 가중치로 변환 (가까울수록 높은 가중치)
        max_distance = np.percentile(distances, 95)  # 극단값 제어
        weights = np.exp(-distances / max_distance)
        
        return weights
    
    def _quantile_based_weights(self, X_train: pd.DataFrame, 
                              X_train_full: pd.DataFrame) -> np.ndarray:
        """분위수 기반 가중치"""
        weights = np.ones(len(X_train))
        
        for col in self.feature_cols:
            if col not in X_train_full.columns:
                continue
                
            train_values = X_train_full[col].dropna()
            test_values = pd.read_csv("../method_07/test_building_merged.csv", 
                                    encoding='utf-8-sig')[col].dropna()
            
            # Test 분포의 중앙 50% 범위
            test_q25 = test_values.quantile(0.25)
            test_q75 = test_values.quantile(0.75)
            
            # 각 Train 샘플이 Test 중앙 범위에 얼마나 가까운지
            col_values = X_train[col].values
            
            # Test 중앙 범위 내: 가중치 증가
            in_core_range = (col_values >= test_q25) & (col_values <= test_q75)
            
            # Test 범위 밖: 가중치 감소
            test_min, test_max = test_values.min(), test_values.max()
            out_of_range = (col_values < test_min) | (col_values > test_max)
            
            col_weights = np.ones(len(col_values))
            col_weights[in_core_range] *= 1.5  # 중앙 범위 내 샘플 가중치 증가
            col_weights[out_of_range] *= 0.5   # 범위 밖 샘플 가중치 감소
            
            weights *= col_weights
        
        return weights

def calculate_seasonal_weights(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """계절적 특성 기반 가중치"""
    
    # 시간 정보 추출
    train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H', errors='coerce')
    test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
    
    train['month'] = train['datetime'].dt.month
    train['hour'] = train['datetime'].dt.hour
    train['weekday'] = train['datetime'].dt.weekday
    
    test['month'] = test['datetime'].dt.month
    test['hour'] = test['datetime'].dt.hour
    test['weekday'] = test['datetime'].dt.weekday
    
    # Test의 시간 분포
    test_month_dist = test['month'].value_counts(normalize=True)
    test_hour_dist = test['hour'].value_counts(normalize=True)
    test_weekday_dist = test['weekday'].value_counts(normalize=True)
    
    # Train의 시간 분포
    train_month_dist = train['month'].value_counts(normalize=True)
    train_hour_dist = train['hour'].value_counts(normalize=True)
    train_weekday_dist = train['weekday'].value_counts(normalize=True)
    
    # 각 Train 샘플에 대한 가중치 계산
    weights = np.ones(len(train))
    
    for i, row in train.iterrows():
        month_weight = test_month_dist.get(row['month'], 0) / train_month_dist.get(row['month'], 1e-6)
        hour_weight = test_hour_dist.get(row['hour'], 0) / train_hour_dist.get(row['hour'], 1e-6)
        weekday_weight = test_weekday_dist.get(row['weekday'], 0) / train_weekday_dist.get(row['weekday'], 1e-6)
        
        # 가중치 조합 (곱셈보다는 평균)
        combined_weight = (month_weight + hour_weight + weekday_weight) / 3
        weights[i] = combined_weight
    
    # 정규화 및 클리핑
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 3.0)
    
    return weights

def calculate_building_weights(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """건물 특성 기반 가중치"""
    
    # 건물유형별 분포
    test_building_dist = test['건물유형'].value_counts(normalize=True)
    train_building_dist = train['건물유형'].value_counts(normalize=True)
    
    # 면적 범위별 가중치
    if '연면적(m2)' in train.columns and '연면적(m2)' in test.columns:
        test_area_q25 = test['연면적(m2)'].quantile(0.25)
        test_area_q75 = test['연면적(m2)'].quantile(0.75)
    else:
        test_area_q25, test_area_q75 = 0, np.inf
    
    weights = np.ones(len(train))
    
    for i, row in train.iterrows():
        # 건물유형 가중치
        building_type = row['건물유형']
        type_weight = test_building_dist.get(building_type, 0) / train_building_dist.get(building_type, 1e-6)
        
        # 면적 가중치
        area_weight = 1.0
        if '연면적(m2)' in train.columns:
            area = row['연면적(m2)']
            if test_area_q25 <= area <= test_area_q75:
                area_weight = 1.2  # Test 중앙 범위 내 건물 가중치 증가
            else:
                area_weight = 0.8  # 범위 밖 건물 가중치 감소
        
        weights[i] = (type_weight + area_weight) / 2
    
    # 정규화 및 클리핑
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.3, 2.5)
    
    return weights

def visualize_weights(weights: np.ndarray, train: pd.DataFrame, 
                     save_path: str = "eda/weight_distribution.png"):
    """가중치 분포 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 가중치 히스토그램
    axes[0,0].hist(weights, bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title('가중치 분포')
    axes[0,0].set_xlabel('가중치')
    axes[0,0].set_ylabel('빈도')
    
    # 가중치 vs 기온
    if '기온(°C)' in train.columns:
        axes[0,1].scatter(train['기온(°C)'], weights, alpha=0.5, s=1)
        axes[0,1].set_title('가중치 vs 기온')
        axes[0,1].set_xlabel('기온(°C)')
        axes[0,1].set_ylabel('가중치')
    
    # 가중치 vs 월
    if 'datetime' in train.columns or '일시' in train.columns:
        if 'datetime' not in train.columns:
            train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H', errors='coerce')
        train['month'] = train['datetime'].dt.month
        
        month_weights = train.groupby('month').apply(lambda x: weights[x.index].mean())
        axes[1,0].bar(month_weights.index, month_weights.values)
        axes[1,0].set_title('월별 평균 가중치')
        axes[1,0].set_xlabel('월')
        axes[1,0].set_ylabel('평균 가중치')
    
    # 가중치 vs 건물유형
    if '건물유형' in train.columns:
        type_weights = train.groupby('건물유형').apply(lambda x: weights[x.index].mean())
        axes[1,1].bar(range(len(type_weights)), type_weights.values)
        axes[1,1].set_title('건물유형별 평균 가중치')
        axes[1,1].set_xlabel('건물유형')
        axes[1,1].set_ylabel('평균 가중치')
        axes[1,1].set_xticks(range(len(type_weights)))
        axes[1,1].set_xticklabels(type_weights.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def combine_weights(weights_list: List[np.ndarray], 
                   weights_names: List[str],
                   combination_method: str = 'geometric_mean') -> np.ndarray:
    """여러 가중치 조합"""
    
    if combination_method == 'geometric_mean':
        # 기하평균
        combined = np.ones(len(weights_list[0]))
        for weights in weights_list:
            combined *= weights
        combined = combined ** (1.0 / len(weights_list))
        
    elif combination_method == 'arithmetic_mean':
        # 산술평균
        combined = np.mean(weights_list, axis=0)
        
    elif combination_method == 'weighted_average':
        # 가중평균 (첫 번째 가중치에 더 큰 비중)
        weights_importance = [0.4, 0.3, 0.3][:len(weights_list)]
        combined = np.average(weights_list, axis=0, weights=weights_importance)
        
    else:
        raise ValueError(f"지원하지 않는 조합 방법: {combination_method}")
    
    # 정규화
    combined = combined / combined.mean()
    
    return combined

def main():
    """메인 실행 함수"""
    
    print("🔄 Importance Sampling 가중치 계산 시작...")
    
    # 데이터 로드
    train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    test = pd.read_csv("../method_07/test_building_merged.csv", encoding='utf-8-sig')
    
    print(f"Train: {len(train):,}행, Test: {len(test):,}행")
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    # 1. 기후 변수 기반 가중치
    print("1. 기후 변수 기반 가중치 계산...")
    sampler = ImportanceSampler(method='distance_based')
    sampler.fit(train, test)
    weather_weights = sampler.calculate_weights(train)
    
    # 2. 계절적 특성 가중치
    print("2. 계절적 특성 가중치 계산...")
    seasonal_weights = calculate_seasonal_weights(train, test)
    
    # 3. 건물 특성 가중치
    print("3. 건물 특성 가중치 계산...")
    building_weights = calculate_building_weights(train, test)
    
    # 4. 가중치 조합
    print("4. 가중치 조합...")
    all_weights = [weather_weights, seasonal_weights, building_weights]
    weight_names = ['weather', 'seasonal', 'building']
    
    combined_weights = combine_weights(all_weights, weight_names, 'geometric_mean')
    
    # 5. 시각화
    print("5. 가중치 분포 시각화...")
    visualize_weights(combined_weights, train, f"{eda_dir}/combined_weights_distribution.png")
    
    # 개별 가중치도 시각화
    for weights, name in zip(all_weights, weight_names):
        visualize_weights(weights, train, f"{eda_dir}/{name}_weights_distribution.png")
    
    # 6. 가중치 저장
    print("6. 가중치 저장...")
    weights_df = pd.DataFrame({
        'index': train.index,
        'weather_weights': weather_weights,
        'seasonal_weights': seasonal_weights,
        'building_weights': building_weights,
        'combined_weights': combined_weights
    })
    
    weights_df.to_csv(f"{eda_dir}/importance_sampling_weights.csv", index=False, encoding='utf-8-sig')
    
    # 7. 통계 요약
    print("\n📊 가중치 통계:")
    print(f"   - 평균: {combined_weights.mean():.3f}")
    print(f"   - 표준편차: {combined_weights.std():.3f}")
    print(f"   - 최소값: {combined_weights.min():.3f}")
    print(f"   - 최대값: {combined_weights.max():.3f}")
    print(f"   - 5% 분위수: {np.percentile(combined_weights, 5):.3f}")
    print(f"   - 95% 분위수: {np.percentile(combined_weights, 95):.3f}")
    
    # 가중치가 높은/낮은 샘플 분석
    high_weight_indices = np.where(combined_weights > np.percentile(combined_weights, 90))[0]
    low_weight_indices = np.where(combined_weights < np.percentile(combined_weights, 10))[0]
    
    print(f"\n🔍 고가중치 샘플 (상위 10%): {len(high_weight_indices)}개")
    if len(high_weight_indices) > 0:
        high_weight_sample = train.iloc[high_weight_indices]
        if '기온(°C)' in train.columns:
            print(f"   - 평균 기온: {high_weight_sample['기온(°C)'].mean():.1f}°C")
        if '건물유형' in train.columns:
            print(f"   - 주요 건물유형: {high_weight_sample['건물유형'].mode().iloc[0]}")
    
    print(f"\n🔍 저가중치 샘플 (하위 10%): {len(low_weight_indices)}개")
    if len(low_weight_indices) > 0:
        low_weight_sample = train.iloc[low_weight_indices]
        if '기온(°C)' in train.columns:
            print(f"   - 평균 기온: {low_weight_sample['기온(°C)'].mean():.1f}°C")
        if '건물유형' in train.columns:
            print(f"   - 주요 건물유형: {low_weight_sample['건물유형'].mode().iloc[0]}")
    
    print(f"\n✅ Importance Sampling 완료! 결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    
    return combined_weights

if __name__ == "__main__":
    weights = main()