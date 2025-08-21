"""
Test Distribution-Based CV Strategy: Test 분포 기반 교차검증 전략

기존 시간순 분할 대신 Test 데이터와 유사한 분포를 가진 validation set을 구성하여
Train-Test gap을 최소화하는 CV 전략을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Iterator
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class TestSimilarityCV(BaseCrossValidator):
    """Test 분포와 유사한 validation set을 생성하는 CV"""
    
    def __init__(self, n_splits: int = 5, test_similarity_ratio: float = 0.7,
                 similarity_features: Optional[List[str]] = None,
                 random_state: int = 2025):
        """
        Args:
            n_splits: CV 폴드 수
            test_similarity_ratio: validation이 test와 얼마나 유사해야 하는지 (0~1)
            similarity_features: 유사도 계산에 사용할 피처들
            random_state: 랜덤 시드
        """
        self.n_splits = n_splits
        self.test_similarity_ratio = test_similarity_ratio
        self.similarity_features = similarity_features
        self.random_state = random_state
        self.test_reference = None
        
    def set_test_reference(self, test_data: pd.DataFrame):
        """참조할 test 데이터 설정"""
        self.test_reference = test_data.copy()
        
        # 기본 유사도 피처 설정
        if self.similarity_features is None:
            available_features = ['기온(°C)', '습도(%)', '풍속(m/s)']
            self.similarity_features = [f for f in available_features 
                                      if f in test_data.columns]
    
    def _calculate_test_similarity(self, train_data: pd.DataFrame) -> np.ndarray:
        """각 train 샘플의 test 분포와의 유사도 계산"""
        
        if self.test_reference is None:
            raise ValueError("test_reference가 설정되지 않았습니다. set_test_reference()를 먼저 호출하세요.")
        
        # 유사도 계산을 위한 피처 추출
        train_features = train_data[self.similarity_features].dropna()
        test_features = self.test_reference[self.similarity_features].dropna()
        
        # 표준화
        scaler = StandardScaler()
        combined_features = pd.concat([train_features, test_features])
        scaler.fit(combined_features)
        
        train_scaled = scaler.transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        # 각 train 샘플에서 test 분포 중심까지의 거리
        test_center = np.mean(test_scaled, axis=0)
        distances = np.linalg.norm(train_scaled - test_center, axis=1)
        
        # 거리를 유사도로 변환 (작은 거리 = 높은 유사도)
        max_distance = np.percentile(distances, 95)
        similarities = np.exp(-distances / max_distance)
        
        # 원본 인덱스에 맞춰 확장
        full_similarities = np.zeros(len(train_data))
        full_similarities[train_features.index] = similarities
        
        return full_similarities
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """CV 분할 생성"""
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Test 유사도 계산
        similarities = self._calculate_test_similarity(X)
        
        # 시간 정보 추출 (기존 시간 정보도 고려)
        X_with_time = X.copy()
        if '일시' in X.columns:
            X_with_time['datetime'] = pd.to_datetime(X['일시'], format='%Y%m%d %H', errors='coerce')
            X_with_time['month'] = X_with_time['datetime'].dt.month
            X_with_time['hour'] = X_with_time['datetime'].dt.hour
        
        np.random.seed(self.random_state)
        
        for fold in range(self.n_splits):
            # 각 폴드마다 다른 시드 사용
            fold_seed = self.random_state + fold
            np.random.seed(fold_seed)
            
            # Test 유사한 샘플들을 validation으로 우선 선택
            similarity_threshold = np.percentile(similarities, 
                                               (1 - self.test_similarity_ratio) * 100)
            
            high_similarity_indices = indices[similarities >= similarity_threshold]
            low_similarity_indices = indices[similarities < similarity_threshold]
            
            # Validation 크기 결정 (전체의 20%)
            val_size = n_samples // 5
            
            # 고유사도 샘플에서 우선 선택
            if len(high_similarity_indices) >= val_size:
                val_indices = np.random.choice(high_similarity_indices, val_size, replace=False)
            else:
                # 고유사도 샘플이 부족하면 저유사도에서 보충
                remaining_size = val_size - len(high_similarity_indices)
                additional_indices = np.random.choice(low_similarity_indices, 
                                                    remaining_size, replace=False)
                val_indices = np.concatenate([high_similarity_indices, additional_indices])
            
            train_indices = np.setdiff1d(indices, val_indices)
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class ClimateBasedCV(BaseCrossValidator):
    """기후 조건 기반 CV"""
    
    def __init__(self, n_splits: int = 5, climate_strategy: str = 'temperature_based',
                 random_state: int = 2025):
        """
        Args:
            climate_strategy: 'temperature_based', 'season_based', 'clustering_based'
        """
        self.n_splits = n_splits
        self.climate_strategy = climate_strategy
        self.random_state = random_state
        self.test_reference = None
    
    def set_test_reference(self, test_data: pd.DataFrame):
        """참조할 test 데이터 설정"""
        self.test_reference = test_data.copy()
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """기후 조건 기반 분할"""
        
        if self.test_reference is None:
            raise ValueError("test_reference가 설정되지 않았습니다.")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.climate_strategy == 'temperature_based':
            yield from self._temperature_based_split(X, indices)
        elif self.climate_strategy == 'season_based':
            yield from self._season_based_split(X, indices)
        elif self.climate_strategy == 'clustering_based':
            yield from self._clustering_based_split(X, indices)
    
    def _temperature_based_split(self, X: pd.DataFrame, indices: np.ndarray):
        """온도 기준 분할"""
        
        if '기온(°C)' not in X.columns:
            # 온도 정보가 없으면 랜덤 분할
            yield from self._random_split(indices)
            return
        
        # Test 온도 분포 파악
        test_temp = self.test_reference['기온(°C)'].dropna()
        test_temp_mean = test_temp.mean()
        test_temp_std = test_temp.std()
        
        # Train을 온도 기준으로 그룹화
        train_temp = X['기온(°C)']
        
        # Test와 유사한 온도 범위 정의
        temp_lower = test_temp_mean - test_temp_std
        temp_upper = test_temp_mean + test_temp_std
        
        # 온도 그룹 분류
        similar_temp_mask = (train_temp >= temp_lower) & (train_temp <= temp_upper)
        
        np.random.seed(self.random_state)
        
        for fold in range(self.n_splits):
            fold_seed = self.random_state + fold
            np.random.seed(fold_seed)
            
            # 유사 온도 샘플에서 validation 우선 선택
            similar_indices = indices[similar_temp_mask]
            different_indices = indices[~similar_temp_mask]
            
            val_size = len(indices) // 5
            
            if len(similar_indices) >= val_size:
                val_indices = np.random.choice(similar_indices, val_size, replace=False)
            else:
                remaining_size = val_size - len(similar_indices)
                if len(different_indices) >= remaining_size:
                    additional_indices = np.random.choice(different_indices, 
                                                        remaining_size, replace=False)
                    val_indices = np.concatenate([similar_indices, additional_indices])
                else:
                    val_indices = indices[:val_size]  # fallback
            
            train_indices = np.setdiff1d(indices, val_indices)
            yield train_indices, val_indices
    
    def _season_based_split(self, X: pd.DataFrame, indices: np.ndarray):
        """계절 기준 분할"""
        
        X_with_time = X.copy()
        if '일시' in X.columns:
            X_with_time['datetime'] = pd.to_datetime(X['일시'], format='%Y%m%d %H', errors='coerce')
            X_with_time['month'] = X_with_time['datetime'].dt.month
        else:
            yield from self._random_split(indices)
            return
        
        # Test 월 분포
        test_months = set(self.test_reference['일시'].str[:6].str[-2:].astype(int))
        
        # Train에서 test 월과 같은 월의 샘플들
        train_months = X_with_time['month']
        similar_month_mask = train_months.isin(test_months)
        
        np.random.seed(self.random_state)
        
        for fold in range(self.n_splits):
            fold_seed = self.random_state + fold
            np.random.seed(fold_seed)
            
            similar_indices = indices[similar_month_mask]
            different_indices = indices[~similar_month_mask]
            
            val_size = len(indices) // 5
            
            if len(similar_indices) >= val_size:
                val_indices = np.random.choice(similar_indices, val_size, replace=False)
            else:
                remaining_size = val_size - len(similar_indices)
                if len(different_indices) >= remaining_size:
                    additional_indices = np.random.choice(different_indices, 
                                                        remaining_size, replace=False)
                    val_indices = np.concatenate([similar_indices, additional_indices])
                else:
                    val_indices = indices[:val_size]
            
            train_indices = np.setdiff1d(indices, val_indices)
            yield train_indices, val_indices
    
    def _clustering_based_split(self, X: pd.DataFrame, indices: np.ndarray):
        """클러스터링 기반 분할"""
        
        climate_features = ['기온(°C)', '습도(%)', '풍속(m/s)']
        climate_features = [f for f in climate_features if f in X.columns]
        
        if not climate_features:
            yield from self._random_split(indices)
            return
        
        # 기후 특성으로 클러스터링
        X_climate = X[climate_features].dropna()
        test_climate = self.test_reference[climate_features].dropna()
        
        # 표준화
        scaler = StandardScaler()
        combined_climate = pd.concat([X_climate, test_climate])
        scaler.fit(combined_climate)
        
        X_scaled = scaler.transform(X_climate)
        test_scaled = scaler.transform(test_climate)
        
        # 클러스터링
        n_clusters = min(10, len(X_climate) // 100)  # 적절한 클러스터 수
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        
        # Train과 Test를 함께 클러스터링
        all_scaled = np.vstack([X_scaled, test_scaled])
        cluster_labels = kmeans.fit_predict(all_scaled)
        
        train_clusters = cluster_labels[:len(X_scaled)]
        test_clusters = cluster_labels[len(X_scaled):]
        
        # Test가 많이 속한 클러스터 찾기
        test_cluster_counts = pd.Series(test_clusters).value_counts()
        dominant_test_clusters = set(test_cluster_counts.head(3).index)
        
        # Train에서 test와 같은 클러스터에 속한 샘플들
        similar_cluster_mask = pd.Series(train_clusters, index=X_climate.index).isin(dominant_test_clusters)
        full_mask = pd.Series(False, index=X.index)
        full_mask.loc[similar_cluster_mask.index] = similar_cluster_mask
        
        np.random.seed(self.random_state)
        
        for fold in range(self.n_splits):
            fold_seed = self.random_state + fold
            np.random.seed(fold_seed)
            
            similar_indices = indices[full_mask]
            different_indices = indices[~full_mask]
            
            val_size = len(indices) // 5
            
            if len(similar_indices) >= val_size:
                val_indices = np.random.choice(similar_indices, val_size, replace=False)
            else:
                remaining_size = val_size - len(similar_indices)
                if len(different_indices) >= remaining_size:
                    additional_indices = np.random.choice(different_indices, 
                                                        remaining_size, replace=False)
                    val_indices = np.concatenate([similar_indices, additional_indices])
                else:
                    val_indices = indices[:val_size]
            
            train_indices = np.setdiff1d(indices, val_indices)
            yield train_indices, val_indices
    
    def _random_split(self, indices: np.ndarray):
        """Fallback 랜덤 분할"""
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(indices)
        
        for fold in range(self.n_splits):
            start = fold * len(indices) // self.n_splits
            end = (fold + 1) * len(indices) // self.n_splits
            val_indices = shuffled_indices[start:end]
            train_indices = np.concatenate([shuffled_indices[:start], shuffled_indices[end:]])
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def analyze_cv_strategies(train: pd.DataFrame, test: pd.DataFrame,
                         target_col: str = '전력소비량(kWh)',
                         save_dir: str = "eda") -> Dict:
    """다양한 CV 전략 분석 및 비교"""
    
    print("📊 CV 전략 분석 시작...")
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. 기존 시간순 분할
    print("1. 시간순 분할 분석...")
    time_based_stats = analyze_time_based_cv(train, test, target_col)
    
    # 2. Test 유사도 기반 분할
    print("2. Test 유사도 기반 분할 분석...")
    similarity_cv = TestSimilarityCV(n_splits=5, test_similarity_ratio=0.7)
    similarity_cv.set_test_reference(test)
    similarity_stats = analyze_cv_strategy(train, test, similarity_cv, target_col, "test_similarity")
    
    # 3. 기후 조건 기반 분할
    print("3. 기후 조건 기반 분할 분석...")
    climate_strategies = ['temperature_based', 'season_based', 'clustering_based']
    climate_stats = {}
    
    for strategy in climate_strategies:
        climate_cv = ClimateBasedCV(n_splits=5, climate_strategy=strategy)
        climate_cv.set_test_reference(test)
        climate_stats[strategy] = analyze_cv_strategy(train, test, climate_cv, target_col, strategy)
    
    # 결과 통합
    all_stats = {
        'time_based': time_based_stats,
        'test_similarity': similarity_stats,
        **climate_stats
    }
    
    # 전략별 비교 시각화
    print("4. CV 전략 비교 시각화...")
    visualize_cv_comparison(train, test, all_stats, save_dir)
    
    # 리포트 생성
    print("5. CV 분석 리포트 생성...")
    generate_cv_report(all_stats, f"{save_dir}/cv_strategy_analysis_report.txt")
    
    print(f"✅ CV 전략 분석 완료! 결과는 {save_dir}/ 디렉토리에 저장되었습니다.")
    
    return all_stats

def analyze_time_based_cv(train: pd.DataFrame, test: pd.DataFrame, 
                         target_col: str) -> Dict:
    """기존 시간순 CV 분석"""
    
    # 시간 정보 추출
    train_with_time = train.copy()
    test_with_time = test.copy()
    
    train_with_time['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H', errors='coerce')
    test_with_time['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
    
    # 시간순 정렬
    train_sorted = train_with_time.sort_values('datetime')
    
    # 80/20 분할
    split_idx = int(len(train_sorted) * 0.8)
    train_part = train_sorted.iloc[:split_idx]
    val_part = train_sorted.iloc[split_idx:]
    
    # 분포 차이 분석
    stats = {
        'train_size': len(train_part),
        'val_size': len(val_part),
        'train_period': (train_part['datetime'].min(), train_part['datetime'].max()),
        'val_period': (val_part['datetime'].min(), val_part['datetime'].max()),
        'test_period': (test_with_time['datetime'].min(), test_with_time['datetime'].max())
    }
    
    # 기후 변수 분포 차이
    climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
    stats['climate_distribution'] = {}
    
    for var in climate_vars:
        if var in train.columns and var in test.columns:
            stats['climate_distribution'][var] = {
                'train_mean': train_part[var].mean(),
                'val_mean': val_part[var].mean(),
                'test_mean': test_with_time[var].mean(),
                'val_test_diff': abs(val_part[var].mean() - test_with_time[var].mean()),
                'train_test_diff': abs(train_part[var].mean() - test_with_time[var].mean())
            }
    
    # 타겟 분포 (있는 경우)
    if target_col in train.columns:
        stats['target_distribution'] = {
            'train_mean': train_part[target_col].mean(),
            'val_mean': val_part[target_col].mean(),
            'train_std': train_part[target_col].std(),
            'val_std': val_part[target_col].std()
        }
    
    return stats

def analyze_cv_strategy(train: pd.DataFrame, test: pd.DataFrame, 
                       cv_strategy: BaseCrossValidator, target_col: str,
                       strategy_name: str) -> Dict:
    """특정 CV 전략 분석"""
    
    stats = {
        'strategy_name': strategy_name,
        'n_splits': cv_strategy.get_n_splits(),
        'fold_stats': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(train)):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]
        
        fold_stat = {
            'fold': fold_idx,
            'train_size': len(fold_train),
            'val_size': len(fold_val),
            'climate_similarity': {}
        }
        
        # 기후 변수 유사도
        climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
        
        for var in climate_vars:
            if var in train.columns and var in test.columns:
                val_mean = fold_val[var].mean()
                test_mean = test[var].mean()
                val_std = fold_val[var].std()
                test_std = test[var].std()
                
                # 평균 차이와 분포 차이
                mean_diff = abs(val_mean - test_mean)
                std_ratio = val_std / test_std if test_std > 0 else 1.0
                
                fold_stat['climate_similarity'][var] = {
                    'mean_diff': mean_diff,
                    'std_ratio': std_ratio,
                    'val_mean': val_mean,
                    'test_mean': test_mean
                }
        
        # 타겟 분포 (있는 경우)
        if target_col in train.columns:
            fold_stat['target_stats'] = {
                'train_mean': fold_train[target_col].mean(),
                'val_mean': fold_val[target_col].mean(),
                'train_std': fold_train[target_col].std(),
                'val_std': fold_val[target_col].std()
            }
        
        stats['fold_stats'].append(fold_stat)
    
    # 전체 통계 계산
    climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
    stats['average_similarity'] = {}
    
    for var in climate_vars:
        if var in train.columns and var in test.columns:
            mean_diffs = [fold['climate_similarity'].get(var, {}).get('mean_diff', 0) 
                         for fold in stats['fold_stats']]
            stats['average_similarity'][var] = np.mean(mean_diffs)
    
    return stats

def visualize_cv_comparison(train: pd.DataFrame, test: pd.DataFrame, 
                           all_stats: Dict, save_dir: str):
    """CV 전략 비교 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 기후 변수별 Test와의 유사도 비교
    climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
    
    for i, var in enumerate(climate_vars):
        if i >= 3:
            break
            
        strategy_names = []
        similarities = []
        
        for strategy_name, stats in all_stats.items():
            if strategy_name == 'time_based':
                if 'climate_distribution' in stats and var in stats['climate_distribution']:
                    similarity = stats['climate_distribution'][var]['val_test_diff']
                    strategy_names.append(strategy_name)
                    similarities.append(similarity)
            else:
                if 'average_similarity' in stats and var in stats['average_similarity']:
                    similarity = stats['average_similarity'][var]
                    strategy_names.append(strategy_name)
                    similarities.append(similarity)
        
        if strategy_names and similarities:
            axes[0, i].bar(strategy_names, similarities)
            axes[0, i].set_title(f'{var} Test 유사도\n(낮을수록 좋음)')
            axes[0, i].set_ylabel('평균 차이')
            axes[0, i].tick_params(axis='x', rotation=45)
    
    # 2. 전략별 validation 크기 분포
    axes[1, 0].set_title('전략별 Validation 크기')
    for strategy_name, stats in all_stats.items():
        if strategy_name == 'time_based':
            val_sizes = [stats['val_size']]
        else:
            val_sizes = [fold['val_size'] for fold in stats['fold_stats']]
        
        axes[1, 0].boxplot(val_sizes, positions=[list(all_stats.keys()).index(strategy_name)], 
                          widths=0.6, patch_artist=True)
    
    axes[1, 0].set_xticks(range(len(all_stats)))
    axes[1, 0].set_xticklabels(all_stats.keys(), rotation=45)
    
    # 3. 종합 유사도 점수
    strategy_names = []
    similarity_scores = []
    
    for strategy_name, stats in all_stats.items():
        if strategy_name == 'time_based':
            if 'climate_distribution' in stats:
                score = np.mean([stats['climate_distribution'][var]['val_test_diff'] 
                               for var in climate_vars 
                               if var in stats['climate_distribution']])
                strategy_names.append(strategy_name)
                similarity_scores.append(1 / (1 + score))  # 점수 변환 (높을수록 좋음)
        else:
            if 'average_similarity' in stats:
                score = np.mean([stats['average_similarity'][var] 
                               for var in climate_vars 
                               if var in stats['average_similarity']])
                strategy_names.append(strategy_name)
                similarity_scores.append(1 / (1 + score))
    
    if strategy_names and similarity_scores:
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))
        bars = axes[1, 1].bar(strategy_names, similarity_scores, color=colors)
        axes[1, 1].set_title('종합 Test 유사도 점수\n(높을수록 좋음)')
        axes[1, 1].set_ylabel('유사도 점수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 가장 좋은 전략 표시
        best_idx = np.argmax(similarity_scores)
        bars[best_idx].set_color('red')
        axes[1, 1].text(best_idx, similarity_scores[best_idx] + 0.01, 
                        'BEST', ha='center', fontweight='bold', color='red')
    
    # 4. Test 분포와의 온도 비교 (히스토그램)
    if '기온(°C)' in train.columns and '기온(°C)' in test.columns:
        axes[1, 2].hist(test['기온(°C)'].dropna(), bins=30, alpha=0.7, 
                       label='Test', density=True, color='red')
        
        # 각 전략의 첫 번째 fold validation 온도 분포
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (strategy_name, stats) in enumerate(list(all_stats.items())[:5]):
            if strategy_name == 'time_based':
                # 시간순 분할의 validation 부분
                if 'datetime' in train.columns:
                    train_sorted = train.sort_values('datetime')
                else:
                    train_copy = train.copy()
                    train_copy['datetime'] = pd.to_datetime(train_copy['일시'], format='%Y%m%d %H', errors='coerce')
                    train_sorted = train_copy.sort_values('datetime')
                split_idx = int(len(train_sorted) * 0.8)
                val_temp = train_sorted.iloc[split_idx:]['기온(°C)'].dropna()
            else:
                # 첫 번째 fold의 validation
                if 'fold_stats' in stats and len(stats['fold_stats']) > 0:
                    # CV 전략에서 실제 인덱스를 사용해 validation 온도 추출
                    # 이를 위해 실제 split을 다시 수행
                    cv_instance = create_cv_instance(strategy_name, test)
                    first_split = next(cv_instance.split(train))
                    val_indices = first_split[1]
                    val_temp = train.iloc[val_indices]['기온(°C)'].dropna()
                else:
                    continue
            
            if len(val_temp) > 0:
                axes[1, 2].hist(val_temp, bins=30, alpha=0.5, 
                               label=f'{strategy_name} Val', density=True, color=colors[i])
        
        axes[1, 2].set_title('온도 분포 비교')
        axes[1, 2].set_xlabel('기온(°C)')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cv_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_cv_instance(strategy_name: str, test_ref: pd.DataFrame):
    """전략 이름으로 CV 인스턴스 생성"""
    if strategy_name == 'test_similarity':
        cv = TestSimilarityCV(n_splits=5)
        cv.set_test_reference(test_ref)
        return cv
    elif strategy_name in ['temperature_based', 'season_based', 'clustering_based']:
        cv = ClimateBasedCV(n_splits=5, climate_strategy=strategy_name)
        cv.set_test_reference(test_ref)
        return cv
    else:
        return None

def generate_cv_report(all_stats: Dict, save_path: str):
    """CV 전략 분석 리포트 생성"""
    
    report = ["=" * 80]
    report.append("CV 전략 분석 리포트")
    report.append("=" * 80)
    report.append("")
    
    # 전략별 요약
    report.append("전략별 Test 유사도 분석:")
    report.append("-" * 40)
    
    climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
    
    for strategy_name, stats in all_stats.items():
        report.append(f"\n{strategy_name.upper()}:")
        
        if strategy_name == 'time_based':
            if 'climate_distribution' in stats:
                for var in climate_vars:
                    if var in stats['climate_distribution']:
                        diff = stats['climate_distribution'][var]['val_test_diff']
                        report.append(f"  - {var} 평균 차이: {diff:.3f}")
        else:
            if 'average_similarity' in stats:
                for var in climate_vars:
                    if var in stats['average_similarity']:
                        diff = stats['average_similarity'][var]
                        report.append(f"  - {var} 평균 차이: {diff:.3f}")
    
    # 권장사항
    report.append("\n\n권장사항:")
    report.append("-" * 40)
    
    # 가장 좋은 전략 찾기
    best_strategy = None
    best_score = float('inf')
    
    for strategy_name, stats in all_stats.items():
        if strategy_name == 'time_based':
            if 'climate_distribution' in stats:
                score = np.mean([stats['climate_distribution'][var]['val_test_diff'] 
                               for var in climate_vars 
                               if var in stats['climate_distribution']])
        else:
            if 'average_similarity' in stats:
                score = np.mean([stats['average_similarity'][var] 
                               for var in climate_vars 
                               if var in stats['average_similarity']])
            else:
                continue
        
        if score < best_score:
            best_score = score
            best_strategy = strategy_name
    
    if best_strategy:
        report.append(f"1. 가장 Test와 유사한 CV 전략: {best_strategy}")
        report.append(f"   평균 기후 변수 차이: {best_score:.3f}")
    
    report.append("2. Train-Test gap 최소화를 위해 권장되는 전략:")
    report.append("   - test_similarity: Test 분포와 가장 유사한 validation 구성")
    report.append("   - temperature_based: 온도 기준 유사도 매칭")
    report.append("   - clustering_based: 다차원 기후 조건 클러스터링")
    
    report.append("3. 기존 time_based 전략 대비 예상 개선:")
    if 'time_based' in all_stats and best_strategy != 'time_based':
        time_score = np.mean([all_stats['time_based']['climate_distribution'][var]['val_test_diff'] 
                             for var in climate_vars 
                             if var in all_stats['time_based']['climate_distribution']])
        improvement = (time_score - best_score) / time_score * 100
        report.append(f"   - 기후 변수 유사도 {improvement:.1f}% 개선 예상")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """메인 실행 함수"""
    
    # 데이터 로드
    train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    test = pd.read_csv("../method_07/test_building_merged.csv", encoding='utf-8-sig')
    
    print(f"데이터 로드 완료 - Train: {len(train):,}행, Test: {len(test):,}행")
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    # CV 전략 분석
    cv_analysis = analyze_cv_strategies(train, test, save_dir=eda_dir)
    
    # 최적 CV 전략 추천
    print("\n🎯 CV 전략 추천:")
    
    climate_vars = ['기온(°C)', '습도(%)', '풍속(m/s)']
    strategy_scores = {}
    
    for strategy_name, stats in cv_analysis.items():
        if strategy_name == 'time_based':
            if 'climate_distribution' in stats:
                score = np.mean([stats['climate_distribution'][var]['val_test_diff'] 
                               for var in climate_vars 
                               if var in stats['climate_distribution']])
                strategy_scores[strategy_name] = score
        else:
            if 'average_similarity' in stats:
                score = np.mean([stats['average_similarity'][var] 
                               for var in climate_vars 
                               if var in stats['average_similarity']])
                strategy_scores[strategy_name] = score
    
    # 점수 순으로 정렬 (낮을수록 좋음)
    sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1])
    
    for i, (strategy, score) in enumerate(sorted_strategies):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        print(f"   {rank} {strategy}: 평균 차이 {score:.3f}")
    
    best_strategy = sorted_strategies[0][0] if sorted_strategies else 'test_similarity'
    print(f"\n✅ 권장 CV 전략: {best_strategy}")
    print(f"   결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    
    return cv_analysis, best_strategy

if __name__ == "__main__":
    cv_analysis, best_strategy = main()