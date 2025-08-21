"""
Train Filtering: Test 분포 기준 Train 데이터 필터링

Test에 존재하지 않는 분포의 Train 샘플들을 제거하여
Distribution shift를 줄이고 Test 성능을 개선합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class TrainFilter:
    """Test 분포 기반 Train 데이터 필터링"""
    
    def __init__(self, filter_strategy: str = 'statistical', 
                 filter_strength: str = 'moderate'):
        """
        Args:
            filter_strategy: 'statistical', 'distance', 'clustering', 'combined'
            filter_strength: 'conservative', 'moderate', 'aggressive'
        """
        self.filter_strategy = filter_strategy
        self.filter_strength = filter_strength
        self.strength_params = {
            'conservative': {'percentile': 0.95, 'sigma': 2.5, 'distance_threshold': 0.8},
            'moderate': {'percentile': 0.90, 'sigma': 2.0, 'distance_threshold': 0.7},
            'aggressive': {'percentile': 0.80, 'sigma': 1.5, 'distance_threshold': 0.6}
        }
        
    def get_filter_params(self) -> Dict:
        """필터링 강도에 따른 파라미터 반환"""
        return self.strength_params[self.filter_strength]
    
    def statistical_filter(self, train: pd.DataFrame, test: pd.DataFrame,
                          feature_cols: Optional[List[str]] = None) -> pd.Index:
        """통계적 기준 필터링"""
        
        if feature_cols is None:
            feature_cols = ['기온(°C)', '습도(%)', '풍속(m/s)']
            if '강수량(mm)' in train.columns and '강수량(mm)' in test.columns:
                feature_cols.append('강수량(mm)')
        
        params = self.get_filter_params()
        percentile = params['percentile']
        
        filter_mask = pd.Series(True, index=train.index)
        filter_reasons = []
        
        for col in feature_cols:
            if col not in train.columns or col not in test.columns:
                continue
                
            train_values = train[col].dropna()
            test_values = test[col].dropna()
            
            # Test 분포의 percentile 범위
            test_lower = test_values.quantile((1 - percentile) / 2)
            test_upper = test_values.quantile(percentile + (1 - percentile) / 2)
            
            # 범위 밖 샘플 제거
            col_filter = (train[col] >= test_lower) & (train[col] <= test_upper)
            filter_mask &= col_filter
            
            removed_count = len(train) - col_filter.sum()
            if removed_count > 0:
                filter_reasons.append(f"{col}: {removed_count}개 샘플 제거 (범위: {test_lower:.2f}~{test_upper:.2f})")
        
        return train.index[filter_mask], filter_reasons
    
    def seasonal_filter(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.Index, List[str]]:
        """계절적 특성 기준 필터링"""
        
        # 시간 정보 추출
        train = train.copy()
        test = test.copy()
        
        train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H', errors='coerce')
        test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
        
        train['month'] = train['datetime'].dt.month
        train['hour'] = train['datetime'].dt.hour
        train['weekday'] = train['datetime'].dt.weekday
        
        test['month'] = test['datetime'].dt.month
        test['hour'] = test['datetime'].dt.hour  
        test['weekday'] = test['datetime'].dt.weekday
        
        # Test에 존재하는 시간 범위
        test_months = set(test['month'].dropna())
        test_hours = set(test['hour'].dropna())
        test_weekdays = set(test['weekday'].dropna())
        
        # 필터링
        filter_mask = (
            train['month'].isin(test_months) &
            train['hour'].isin(test_hours) &
            train['weekday'].isin(test_weekdays)
        )
        
        filter_reasons = [
            f"월 필터링: {len(train) - train['month'].isin(test_months).sum()}개 제거",
            f"시간 필터링: {len(train) - train['hour'].isin(test_hours).sum()}개 제거", 
            f"요일 필터링: {len(train) - train['weekday'].isin(test_weekdays).sum()}개 제거"
        ]
        
        return train.index[filter_mask], filter_reasons
    
    def consumption_outlier_filter(self, train: pd.DataFrame, 
                                 consumption_col: str = '전력소비량(kWh)') -> Tuple[pd.Index, List[str]]:
        """전력소비량 이상치 필터링"""
        
        if consumption_col not in train.columns:
            return train.index, ["전력소비량 컬럼 없음"]
        
        consumption = train[consumption_col].dropna()
        params = self.get_filter_params()
        
        # 방법 1: Z-score 기반
        z_scores = np.abs((consumption - consumption.mean()) / consumption.std())
        z_filter = z_scores <= params['sigma']
        
        # 방법 2: IQR 기반  
        q1 = consumption.quantile(0.25)
        q3 = consumption.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_filter = (consumption >= iqr_lower) & (consumption <= iqr_upper)
        
        # 방법 3: Percentile 기반
        percentile = params['percentile']
        perc_lower = consumption.quantile((1 - percentile) / 2)
        perc_upper = consumption.quantile(percentile + (1 - percentile) / 2)
        perc_filter = (consumption >= perc_lower) & (consumption <= perc_upper)
        
        # 조합 (모든 조건을 만족하는 샘플만 유지)
        if self.filter_strength == 'conservative':
            combined_filter = perc_filter
        elif self.filter_strength == 'moderate':
            combined_filter = iqr_filter & perc_filter
        else:  # aggressive
            combined_filter = z_filter & iqr_filter & perc_filter
        
        filter_reasons = [
            f"Z-score 필터링: {len(consumption) - z_filter.sum()}개 제거 (σ>{params['sigma']})",
            f"IQR 필터링: {len(consumption) - iqr_filter.sum()}개 제거",
            f"Percentile 필터링: {len(consumption) - perc_filter.sum()}개 제거 ({percentile*100}%)",
            f"최종 조합: {len(consumption) - combined_filter.sum()}개 제거"
        ]
        
        # 인덱스 정렬
        valid_indices = consumption[combined_filter].index
        
        return valid_indices, filter_reasons
    
    def building_specific_filter(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.Index, List[str]]:
        """건물별 특성 기반 필터링"""
        
        filter_mask = pd.Series(True, index=train.index)
        filter_reasons = []
        
        # 1. 건물유형 필터링
        if '건물유형' in train.columns and '건물유형' in test.columns:
            test_building_types = set(test['건물유형'].dropna())
            building_filter = train['건물유형'].isin(test_building_types)
            filter_mask &= building_filter
            
            removed_types = set(train['건물유형'].dropna()) - test_building_types
            if removed_types:
                filter_reasons.append(f"건물유형 제거: {removed_types}")
        
        # 2. 면적 범위 필터링
        area_cols = ['연면적(m2)', '냉방면적(m2)']
        for col in area_cols:
            if col not in train.columns or col not in test.columns:
                continue
                
            test_values = test[col].dropna()
            if len(test_values) == 0:
                continue
                
            # Test 면적 범위의 95% 범위
            test_lower = test_values.quantile(0.025)
            test_upper = test_values.quantile(0.975)
            
            col_filter = (train[col] >= test_lower) & (train[col] <= test_upper)
            filter_mask &= col_filter
            
            removed_count = len(train) - col_filter.sum()
            if removed_count > 0:
                filter_reasons.append(f"{col} 범위 제거: {removed_count}개 (범위: {test_lower:.0f}~{test_upper:.0f})")
        
        # 3. 설비 보유 여부 필터링
        equipment_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
        for col in equipment_cols:
            if col not in train.columns or col not in test.columns:
                continue
                
            # Test에서 해당 설비를 가진 건물의 비율
            test_has_equipment = (pd.to_numeric(test[col], errors='coerce').fillna(0) > 0).mean()
            train_has_equipment = (pd.to_numeric(train[col], errors='coerce').fillna(0) > 0)
            
            # Test 분포와 너무 다른 Train 샘플 제거
            if test_has_equipment < 0.1:  # Test에서 거의 없으면
                # Train에서도 해당 설비가 많은 샘플들 제거
                equipment_filter = ~train_has_equipment | (train_has_equipment & (np.random.random(len(train)) < 0.3))
            elif test_has_equipment > 0.9:  # Test에서 거의 모든 건물이 보유
                # Train에서 해당 설비가 없는 샘플들 제거
                equipment_filter = train_has_equipment | (np.random.random(len(train)) < 0.3)
            else:
                equipment_filter = pd.Series(True, index=train.index)
            
            filter_mask &= equipment_filter
        
        return train.index[filter_mask], filter_reasons
    
    def distance_based_filter(self, train: pd.DataFrame, test: pd.DataFrame,
                            feature_cols: Optional[List[str]] = None) -> Tuple[pd.Index, List[str]]:
        """거리 기반 필터링"""
        
        if feature_cols is None:
            feature_cols = ['기온(°C)', '습도(%)', '풍속(m/s)']
            feature_cols = [col for col in feature_cols 
                          if col in train.columns and col in test.columns]
        
        if not feature_cols:
            return train.index, ["필터링할 피처가 없음"]
        
        # 특징 정규화
        scaler = StandardScaler()
        
        train_features = train[feature_cols].dropna()
        test_features = test[feature_cols].dropna()
        
        # 결합하여 스케일러 학습
        all_features = pd.concat([train_features, test_features])
        scaler.fit(all_features)
        
        train_scaled = scaler.transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        # 각 Train 샘플에서 가장 가까운 Test 샘플까지의 거리
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs.fit(test_scaled)
        
        distances, _ = nbrs.kneighbors(train_scaled)
        distances = distances.flatten()
        
        # 거리 임계값 설정
        params = self.get_filter_params()
        distance_threshold = np.percentile(distances, params['distance_threshold'] * 100)
        
        # 임계값보다 가까운 샘플들만 유지
        close_samples = distances <= distance_threshold
        valid_indices = train_features.index[close_samples]
        
        filter_reasons = [
            f"거리 기반 필터링: {len(train_features) - close_samples.sum()}개 제거",
            f"거리 임계값: {distance_threshold:.3f}",
            f"평균 거리: {distances.mean():.3f}"
        ]
        
        return valid_indices, filter_reasons

def apply_combined_filter(train: pd.DataFrame, test: pd.DataFrame,
                         filter_strategy: str = 'moderate',
                         save_analysis: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """통합 필터링 적용"""
    
    print(f"🔍 Train 필터링 시작 (전략: {filter_strategy})")
    print(f"원본 Train 데이터: {len(train):,}행")
    
    filter_results = {
        'original_count': len(train),
        'filters_applied': [],
        'removal_stats': {}
    }
    
    current_train = train.copy()
    
    # 1. 통계적 필터링
    print("\n1. 통계적 필터링...")
    filterer = TrainFilter(filter_strategy='statistical', filter_strength=filter_strategy)
    stat_indices, stat_reasons = filterer.statistical_filter(current_train, test)
    
    current_train = current_train.loc[stat_indices]
    removed_stat = len(train) - len(current_train)
    
    filter_results['filters_applied'].append('statistical')
    filter_results['removal_stats']['statistical'] = {
        'removed_count': removed_stat,
        'remaining_count': len(current_train),
        'reasons': stat_reasons
    }
    
    print(f"   제거: {removed_stat:,}개, 남은 데이터: {len(current_train):,}행")
    
    # 2. 계절적 필터링
    print("\n2. 계절적 필터링...")
    seasonal_indices, seasonal_reasons = filterer.seasonal_filter(current_train, test)
    
    prev_count = len(current_train)
    current_train = current_train.loc[seasonal_indices]
    removed_seasonal = prev_count - len(current_train)
    
    filter_results['filters_applied'].append('seasonal')
    filter_results['removal_stats']['seasonal'] = {
        'removed_count': removed_seasonal,
        'remaining_count': len(current_train),
        'reasons': seasonal_reasons
    }
    
    print(f"   제거: {removed_seasonal:,}개, 남은 데이터: {len(current_train):,}행")
    
    # 3. 전력소비량 이상치 필터링
    print("\n3. 전력소비량 이상치 필터링...")
    consumption_indices, consumption_reasons = filterer.consumption_outlier_filter(current_train)
    
    prev_count = len(current_train)
    current_train = current_train.loc[consumption_indices]
    removed_consumption = prev_count - len(current_train)
    
    filter_results['filters_applied'].append('consumption_outlier')
    filter_results['removal_stats']['consumption_outlier'] = {
        'removed_count': removed_consumption,
        'remaining_count': len(current_train),
        'reasons': consumption_reasons
    }
    
    print(f"   제거: {removed_consumption:,}개, 남은 데이터: {len(current_train):,}행")
    
    # 4. 건물 특성 필터링
    print("\n4. 건물 특성 필터링...")
    building_indices, building_reasons = filterer.building_specific_filter(current_train, test)
    
    prev_count = len(current_train)
    current_train = current_train.loc[building_indices]
    removed_building = prev_count - len(current_train)
    
    filter_results['filters_applied'].append('building_specific')
    filter_results['removal_stats']['building_specific'] = {
        'removed_count': removed_building,
        'remaining_count': len(current_train),
        'reasons': building_reasons
    }
    
    print(f"   제거: {removed_building:,}개, 남은 데이터: {len(current_train):,}행")
    
    # 최종 통계
    total_removed = len(train) - len(current_train)
    removal_rate = total_removed / len(train) * 100
    
    filter_results['final_count'] = len(current_train)
    filter_results['total_removed'] = total_removed
    filter_results['removal_rate'] = removal_rate
    
    print(f"\n✅ 필터링 완료!")
    print(f"   전체 제거: {total_removed:,}개 ({removal_rate:.1f}%)")
    print(f"   최종 데이터: {len(current_train):,}행")
    
    # 분석 저장
    if save_analysis:
        save_filter_analysis(train, current_train, test, filter_results)
    
    return current_train, filter_results

def save_filter_analysis(original_train: pd.DataFrame, filtered_train: pd.DataFrame,
                        test: pd.DataFrame, filter_results: Dict, 
                        save_dir: str = "eda"):
    """필터링 분석 결과 저장"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. 필터링 전후 분포 비교 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    weather_cols = ['기온(°C)', '습도(%)', '풍속(m/s)']
    weather_cols = [col for col in weather_cols 
                    if col in original_train.columns and col in test.columns]
    
    for i, col in enumerate(weather_cols[:3]):
        if i >= 3:
            break
            
        # 원본 vs 필터링 후 vs Test 분포
        axes[0, i].hist(original_train[col].dropna(), bins=50, alpha=0.5, 
                       label='Original Train', density=True, color='blue')
        axes[0, i].hist(filtered_train[col].dropna(), bins=50, alpha=0.7, 
                       label='Filtered Train', density=True, color='green')
        axes[0, i].hist(test[col].dropna(), bins=50, alpha=0.7, 
                       label='Test', density=True, color='red')
        axes[0, i].set_title(f'{col} 분포 비교')
        axes[0, i].legend()
        
        # 시간별 분포 (월별)
        if 'datetime' not in original_train.columns:
            original_train['datetime'] = pd.to_datetime(original_train['일시'], format='%Y%m%d %H', errors='coerce')
            filtered_train['datetime'] = pd.to_datetime(filtered_train['일시'], format='%Y%m%d %H', errors='coerce')
            test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
        
        original_train['month'] = original_train['datetime'].dt.month
        filtered_train['month'] = filtered_train['datetime'].dt.month
        test['month'] = test['datetime'].dt.month
        
        if i == 0:  # 첫 번째 컬럼에서만 월별 분포 표시
            orig_monthly = original_train.groupby('month')[col].mean()
            filt_monthly = filtered_train.groupby('month')[col].mean()
            test_monthly = test.groupby('month')[col].mean()
            
            axes[1, 0].plot(orig_monthly.index, orig_monthly.values, 'o-', label='Original Train')
            axes[1, 0].plot(filt_monthly.index, filt_monthly.values, 's-', label='Filtered Train')
            axes[1, 0].plot(test_monthly.index, test_monthly.values, '^-', label='Test')
            axes[1, 0].set_title(f'월별 평균 {col}')
            axes[1, 0].set_xlabel('월')
            axes[1, 0].legend()
    
    # 제거율 막대그래프
    filter_names = list(filter_results['removal_stats'].keys())
    removal_counts = [filter_results['removal_stats'][f]['removed_count'] for f in filter_names]
    
    axes[1, 1].bar(range(len(filter_names)), removal_counts)
    axes[1, 1].set_title('필터별 제거 샘플 수')
    axes[1, 1].set_xticks(range(len(filter_names)))
    axes[1, 1].set_xticklabels(filter_names, rotation=45)
    axes[1, 1].set_ylabel('제거된 샘플 수')
    
    # 전력소비량 분포 비교 (있는 경우)
    if '전력소비량(kWh)' in original_train.columns:
        axes[1, 2].hist(original_train['전력소비량(kWh)'].dropna(), bins=50, alpha=0.5, 
                       label='Original', density=True, color='blue')
        axes[1, 2].hist(filtered_train['전력소비량(kWh)'].dropna(), bins=50, alpha=0.7, 
                       label='Filtered', density=True, color='green')
        axes[1, 2].set_title('전력소비량 분포 비교')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_filtering_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 필터링 리포트 저장
    report = ["=" * 80]
    report.append("Train 데이터 필터링 분석 리포트")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"원본 Train 데이터: {filter_results['original_count']:,}행")
    report.append(f"최종 Train 데이터: {filter_results['final_count']:,}행")
    report.append(f"전체 제거율: {filter_results['removal_rate']:.1f}%")
    report.append("")
    
    report.append("필터별 제거 통계:")
    report.append("-" * 40)
    
    for filter_name, stats in filter_results['removal_stats'].items():
        report.append(f"\n{filter_name.upper()}:")
        report.append(f"  - 제거: {stats['removed_count']:,}개")
        report.append(f"  - 남은 데이터: {stats['remaining_count']:,}개")
        if stats['reasons']:
            report.append("  - 세부 사유:")
            for reason in stats['reasons']:
                report.append(f"    * {reason}")
    
    with open(f"{save_dir}/train_filtering_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """메인 실행 함수"""
    
    # 데이터 로드
    train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    test = pd.read_csv("../method_07/test_building_merged.csv", encoding='utf-8-sig')
    
    print(f"원본 데이터 - Train: {len(train):,}행, Test: {len(test):,}행")
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    # 다양한 강도로 필터링 테스트
    strategies = ['conservative', 'moderate', 'aggressive']
    
    results = {}
    
    for strategy in strategies:
        print(f"\n" + "="*60)
        print(f"필터링 전략: {strategy.upper()}")
        print("="*60)
        
        filtered_train, filter_results = apply_combined_filter(
            train, test, filter_strategy=strategy, save_analysis=True
        )
        
        # 결과 저장
        filtered_train.to_csv(f"{eda_dir}/train_filtered_{strategy}.csv", 
                            index=False, encoding='utf-8-sig')
        
        results[strategy] = {
            'filtered_data': filtered_train,
            'filter_results': filter_results
        }
        
        # 개별 전략별 분석 저장
        save_filter_analysis(train, filtered_train, test, filter_results, 
                           f"{eda_dir}/{strategy}")
    
    # 전략별 비교 요약
    print("\n" + "="*60)
    print("전략별 필터링 결과 요약")
    print("="*60)
    
    for strategy, result in results.items():
        stats = result['filter_results']
        print(f"{strategy:>12}: {stats['final_count']:>8,}행 "
              f"({stats['removal_rate']:>5.1f}% 제거)")
    
    print(f"\n✅ 모든 필터링 전략 완료! 결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    
    return results

if __name__ == "__main__":
    results = main()