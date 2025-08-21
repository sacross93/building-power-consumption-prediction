"""
Test Adaptive Feature Engineering: Test 분포에 특화된 피처 생성

Test 데이터의 특성(8월, 특정 기후 조건)에 맞는 새로운 피처들을 생성하여
Train-Test gap을 줄이고 성능을 개선합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class TestAdaptiveFeatureEngineer:
    """Test 적응형 피처 엔지니어링"""
    
    def __init__(self, reference_test: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_test: 참조할 test 데이터 (분포 분석용)
        """
        self.reference_test = reference_test
        self.fitted_stats = {}
        
    def set_reference_test(self, test: pd.DataFrame):
        """참조 test 데이터 설정"""
        self.reference_test = test
        
    def fit_test_statistics(self, test: pd.DataFrame):
        """Test 데이터 통계 학습"""
        
        # 시간 정보 추출
        test = test.copy()
        test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
        test['month'] = test['datetime'].dt.month
        test['hour'] = test['datetime'].dt.hour
        test['weekday'] = test['datetime'].dt.weekday
        
        # Test 분포 통계 저장
        self.fitted_stats = {
            'temperature_stats': {
                'mean': test['기온(°C)'].mean(),
                'std': test['기온(°C)'].std(),
                'q25': test['기온(°C)'].quantile(0.25),
                'q75': test['기온(°C)'].quantile(0.75),
                'range': (test['기온(°C)'].min(), test['기온(°C)'].max())
            },
            'humidity_stats': {
                'mean': test['습도(%)'].mean(),
                'std': test['습도(%)'].std(),
                'q25': test['습도(%)'].quantile(0.25),
                'q75': test['습도(%)'].quantile(0.75),
                'range': (test['습도(%)'].min(), test['습도(%)'].max())
            },
            'time_stats': {
                'months': set(test['month'].dropna()),
                'hours': set(test['hour'].dropna()),
                'weekdays': set(test['weekday'].dropna()),
                'hour_distribution': test['hour'].value_counts(normalize=True).to_dict(),
                'weekday_distribution': test['weekday'].value_counts(normalize=True).to_dict()
            },
            'building_stats': {
                'types': set(test['건물유형'].dropna()),
                'type_distribution': test['건물유형'].value_counts(normalize=True).to_dict()
            }
        }
        
        if '연면적(m2)' in test.columns:
            self.fitted_stats['area_stats'] = {
                'mean': test['연면적(m2)'].mean(),
                'std': test['연면적(m2)'].std(),
                'q25': test['연면적(m2)'].quantile(0.25),
                'q75': test['연면적(m2)'].quantile(0.75)
            }
        
        return self
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """계절적 특성 피처 생성 (8월 특화)"""
        
        df = df.copy()
        
        # 시간 정보 추출
        df['datetime'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
        df['month'] = df['datetime'].dt.month
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['day'] = df['datetime'].dt.day
        
        # 8월 특화 피처
        df['is_august'] = (df['month'] == 8).astype(int)
        df['august_day_progress'] = df['day'] / 31  # 8월 진행률
        
        # 여름철 피크 시간대 (8월 기준)
        summer_peak_hours = [12, 13, 14, 15, 16]  # 여름철 전력 피크
        df['is_summer_peak'] = df['hour'].isin(summer_peak_hours).astype(int)
        
        # 여름철 냉방 활성 기간
        df['summer_cooling_intensity'] = 0
        df.loc[df['hour'].between(9, 18), 'summer_cooling_intensity'] = 1  # 주간
        df.loc[df['hour'].between(12, 16), 'summer_cooling_intensity'] = 2  # 피크
        
        # 주말 vs 평일 (여름철)
        df['is_summer_weekend'] = ((df['weekday'] >= 5) & (df['month'].isin([6, 7, 8]))).astype(int)
        
        # 8월 중순/말 구분 (휴가철 고려)
        df['august_period'] = 0
        df.loc[(df['month'] == 8) & (df['day'] <= 15), 'august_period'] = 1  # 8월 초중순
        df.loc[(df['month'] == 8) & (df['day'] > 15), 'august_period'] = 2   # 8월 말
        
        return df
    
    def create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """온도 기반 피처 생성 (Test 분포 적응형)"""
        
        df = df.copy()
        
        if '기온(°C)' not in df.columns:
            return df
        
        temp = df['기온(°C)']
        
        # Test 분포 기준 온도 구간
        if self.fitted_stats and 'temperature_stats' in self.fitted_stats:
            temp_stats = self.fitted_stats['temperature_stats']
            
            # Test 평균 기준 차이
            df['temp_vs_test_mean'] = temp - temp_stats['mean']
            df['temp_vs_test_mean_abs'] = np.abs(df['temp_vs_test_mean'])
            
            # Test 분포 내 위치 (0~1)
            temp_range = temp_stats['range'][1] - temp_stats['range'][0]
            df['temp_position_in_test'] = (temp - temp_stats['range'][0]) / temp_range
            df['temp_position_in_test'] = np.clip(df['temp_position_in_test'], 0, 1)
            
            # Test 중앙 범위 여부
            df['in_test_temp_core'] = (
                (temp >= temp_stats['q25']) & (temp <= temp_stats['q75'])
            ).astype(int)
            
        # 고온 스트레스 지수 (여름철 특화)
        df['high_temp_stress'] = np.maximum(0, temp - 28)  # 28도 이상 스트레스
        df['extreme_heat'] = (temp >= 33).astype(int)       # 폭염 여부
        
        # 온도 변화율 (시간별)
        df = df.sort_values(['건물번호', 'datetime'])
        df['temp_change_1h'] = df.groupby('건물번호')['기온(°C)'].diff()
        df['temp_change_3h'] = df.groupby('건물번호')['기온(°C)'].diff(periods=3)
        
        # 일일 온도 범위 (같은 날 내)
        df['date'] = df['datetime'].dt.date
        daily_temp = df.groupby(['건물번호', 'date'])['기온(°C)'].agg(['min', 'max', 'mean'])
        daily_temp['daily_temp_range'] = daily_temp['max'] - daily_temp['min']
        daily_temp = daily_temp.reset_index()
        
        df = df.merge(daily_temp[['건물번호', 'date', 'daily_temp_range']], 
                     on=['건물번호', 'date'], how='left')
        
        return df
    
    def create_humidity_cooling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """습도 기반 냉방 피처 생성"""
        
        df = df.copy()
        
        if '습도(%)' not in df.columns:
            return df
        
        humidity = df['습도(%)']
        temperature = df.get('기온(°C)', 25)  # 기본값
        
        # 불쾌지수 (여름철 중요)
        df['discomfort_index'] = 1.8 * temperature + 32 + 0.55 * (1 - humidity/100) * (1.8 * temperature - 26)
        
        # 냉방 필요도 지수
        df['cooling_need_index'] = 0
        df.loc[df['discomfort_index'] >= 68, 'cooling_need_index'] = 1  # 약간 불쾌
        df.loc[df['discomfort_index'] >= 75, 'cooling_need_index'] = 2  # 불쾌
        df.loc[df['discomfort_index'] >= 80, 'cooling_need_index'] = 3  # 매우 불쾌
        
        # 고온 고습 상황 (에어컨 과부하)
        df['high_temp_high_humidity'] = (
            (temperature >= 30) & (humidity >= 70)
        ).astype(int)
        
        # 습도 적정 범위 여부 (냉방 효율성)
        df['optimal_humidity'] = (
            (humidity >= 40) & (humidity <= 60)
        ).astype(int)
        
        # Test 습도 분포 기준 피처
        if self.fitted_stats and 'humidity_stats' in self.fitted_stats:
            humidity_stats = self.fitted_stats['humidity_stats']
            
            df['humidity_vs_test_mean'] = humidity - humidity_stats['mean']
            df['in_test_humidity_core'] = (
                (humidity >= humidity_stats['q25']) & (humidity <= humidity_stats['q75'])
            ).astype(int)
        
        return df
    
    def create_building_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """건물 특성 적응형 피처 생성"""
        
        df = df.copy()
        
        # Test에 존재하는 건물유형 여부
        if self.fitted_stats and 'building_stats' in self.fitted_stats:
            test_building_types = self.fitted_stats['building_stats']['types']
            df['is_test_building_type'] = df['건물유형'].isin(test_building_types).astype(int)
            
            # Test에서의 건물유형 비율
            type_dist = self.fitted_stats['building_stats']['type_distribution']
            df['building_type_test_frequency'] = df['건물유형'].map(type_dist).fillna(0)
        
        # 면적 기반 피처 (Test 분포 적응)
        if '연면적(m2)' in df.columns and '냉방면적(m2)' in df.columns:
            df['cooling_area_ratio'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
            
            # Test 면적 분포 기준 구간
            if self.fitted_stats and 'area_stats' in self.fitted_stats:
                area_stats = self.fitted_stats['area_stats']
                
                # Test 평균 면적 대비 비율
                df['area_vs_test_mean'] = df['연면적(m2)'] / area_stats['mean']
                
                # Test 중앙 면적 범위 여부
                df['in_test_area_core'] = (
                    (df['연면적(m2)'] >= area_stats['q25']) & 
                    (df['연면적(m2)'] <= area_stats['q75'])
                ).astype(int)
        
        # 설비 특성 (Test 분포 적응)
        equipment_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
        
        for col in equipment_cols:
            if col in df.columns:
                df[f'{col}_has'] = (pd.to_numeric(df[col], errors='coerce').fillna(0) > 0).astype(int)
                
                # 면적당 용량 (정규화)
                if '연면적(m2)' in df.columns:
                    capacity = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    area = df['연면적(m2)'] + 1  # 0 방지
                    df[f'{col}_per_area'] = capacity / area
        
        return df
    
    def create_test_similarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test 데이터와의 유사도 기반 피처"""
        
        df = df.copy()
        
        if not self.fitted_stats:
            return df
        
        # 시간 패턴 유사도
        if 'time_stats' in self.fitted_stats:
            time_stats = self.fitted_stats['time_stats']
            
            df['datetime'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            
            # Test 시간 분포와의 일치도
            df['hour_test_frequency'] = df['hour'].map(
                time_stats['hour_distribution']
            ).fillna(0)
            
            df['weekday_test_frequency'] = df['weekday'].map(
                time_stats['weekday_distribution']
            ).fillna(0)
            
            # 종합 시간 유사도
            df['temporal_similarity'] = (
                df['hour_test_frequency'] * df['weekday_test_frequency']
            )
        
        # 기후 조건 유사도
        climate_similarity = 1.0
        
        if 'temperature_stats' in self.fitted_stats and '기온(°C)' in df.columns:
            temp_stats = self.fitted_stats['temperature_stats']
            temp_diff = np.abs(df['기온(°C)'] - temp_stats['mean'])
            temp_similarity = np.exp(-temp_diff / temp_stats['std'])
            climate_similarity *= temp_similarity
        
        if 'humidity_stats' in self.fitted_stats and '습도(%)' in df.columns:
            humidity_stats = self.fitted_stats['humidity_stats']
            humidity_diff = np.abs(df['습도(%)'] - humidity_stats['mean'])
            humidity_similarity = np.exp(-humidity_diff / humidity_stats['std'])
            climate_similarity *= humidity_similarity
        
        df['climate_similarity'] = climate_similarity
        
        # 종합 유사도 점수
        df['test_similarity_score'] = (
            df.get('temporal_similarity', 1.0) * 
            df.get('climate_similarity', 1.0)
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test 특성 기반 상호작용 피처"""
        
        df = df.copy()
        
        # 시간 × 온도 상호작용 (여름철 특화)
        if 'hour' in df.columns and '기온(°C)' in df.columns:
            # 한낮 고온 조합
            df['midday_high_temp'] = (
                (df['hour'].between(11, 15)) & (df['기온(°C)'] >= 30)
            ).astype(int)
            
            # 야간 온도 × 시간
            df['night_temp_interaction'] = (
                df['기온(°C)'] * (df['hour'] >= 18).astype(int)
            )
        
        # 건물유형 × 기후 조건
        if '건물유형' in df.columns and '기온(°C)' in df.columns:
            # 호텔 × 고온 (에어컨 집중 사용)
            df['hotel_high_temp'] = (
                (df['건물유형'] == '호텔') & (df['기온(°C)'] >= 28)
            ).astype(int)
            
            # 상가 × 주간 고온
            df['retail_day_heat'] = (
                (df['건물유형'].str.contains('상가|마트', na=False)) & 
                (df['hour'].between(9, 18)) & 
                (df['기온(°C)'] >= 30)
            ).astype(int)
        
        # 면적 × 냉방 부하
        if '연면적(m2)' in df.columns and 'cooling_need_index' in df.columns:
            df['large_building_cooling'] = (
                (df['연면적(m2)'] > df['연면적(m2)'].quantile(0.75)) * 
                df['cooling_need_index']
            )
        
        # 설비 보유 × 기후 조건
        if '태양광용량(kW)' in df.columns and '기온(°C)' in df.columns:
            has_solar = (pd.to_numeric(df['태양광용량(kW)'], errors='coerce').fillna(0) > 0)
            df['solar_high_temp'] = (has_solar & (df['기온(°C)'] >= 30)).astype(int)
        
        return df

def apply_test_adaptive_engineering(train: pd.DataFrame, test: pd.DataFrame,
                                  save_analysis: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Test 적응형 피처 엔지니어링 적용"""
    
    print("🔧 Test 적응형 피처 엔지니어링 시작...")
    
    # 피처 엔지니어 초기화
    engineer = TestAdaptiveFeatureEngineer()
    engineer.fit_test_statistics(test)
    
    # 원본 피처 수
    original_train_features = len(train.columns)
    original_test_features = len(test.columns)
    
    feature_stats = {
        'original_features': original_train_features,
        'added_features': [],
        'feature_groups': {}
    }
    
    print(f"원본 피처: Train {original_train_features}개, Test {original_test_features}개")
    
    # 1. 계절적 특성 피처
    print("1. 계절적 특성 피처 생성...")
    train_enhanced = engineer.create_seasonal_features(train)
    test_enhanced = engineer.create_seasonal_features(test)
    
    seasonal_features = [col for col in train_enhanced.columns 
                        if col not in train.columns]
    feature_stats['feature_groups']['seasonal'] = seasonal_features
    print(f"   추가된 피처: {len(seasonal_features)}개")
    
    # 2. 온도 기반 피처
    print("2. 온도 기반 피처 생성...")
    train_enhanced = engineer.create_temperature_features(train_enhanced)
    test_enhanced = engineer.create_temperature_features(test_enhanced)
    
    temp_features = [col for col in train_enhanced.columns 
                    if col not in train.columns and col not in seasonal_features]
    feature_stats['feature_groups']['temperature'] = temp_features
    print(f"   추가된 피처: {len(temp_features)}개")
    
    # 3. 습도/냉방 피처
    print("3. 습도/냉방 피처 생성...")
    train_enhanced = engineer.create_humidity_cooling_features(train_enhanced)
    test_enhanced = engineer.create_humidity_cooling_features(test_enhanced)
    
    humidity_features = [col for col in train_enhanced.columns 
                        if col not in train.columns and 
                        col not in seasonal_features and 
                        col not in temp_features]
    feature_stats['feature_groups']['humidity_cooling'] = humidity_features
    print(f"   추가된 피처: {len(humidity_features)}개")
    
    # 4. 건물 적응형 피처
    print("4. 건물 적응형 피처 생성...")
    train_enhanced = engineer.create_building_adaptive_features(train_enhanced)
    test_enhanced = engineer.create_building_adaptive_features(test_enhanced)
    
    building_features = [col for col in train_enhanced.columns 
                        if col not in train.columns and 
                        col not in seasonal_features and 
                        col not in temp_features and
                        col not in humidity_features]
    feature_stats['feature_groups']['building_adaptive'] = building_features
    print(f"   추가된 피처: {len(building_features)}개")
    
    # 5. Test 유사도 피처
    print("5. Test 유사도 피처 생성...")
    train_enhanced = engineer.create_test_similarity_features(train_enhanced)
    test_enhanced = engineer.create_test_similarity_features(test_enhanced)
    
    similarity_features = [col for col in train_enhanced.columns 
                          if col not in train.columns and 
                          col not in seasonal_features and 
                          col not in temp_features and
                          col not in humidity_features and
                          col not in building_features]
    feature_stats['feature_groups']['test_similarity'] = similarity_features
    print(f"   추가된 피처: {len(similarity_features)}개")
    
    # 6. 상호작용 피처
    print("6. 상호작용 피처 생성...")
    train_enhanced = engineer.create_interaction_features(train_enhanced)
    test_enhanced = engineer.create_interaction_features(test_enhanced)
    
    interaction_features = [col for col in train_enhanced.columns 
                           if col not in train.columns and 
                           col not in seasonal_features and 
                           col not in temp_features and
                           col not in humidity_features and
                           col not in building_features and
                           col not in similarity_features]
    feature_stats['feature_groups']['interaction'] = interaction_features
    print(f"   추가된 피처: {len(interaction_features)}개")
    
    # 전체 통계 업데이트
    all_new_features = (seasonal_features + temp_features + humidity_features + 
                       building_features + similarity_features + interaction_features)
    feature_stats['added_features'] = all_new_features
    feature_stats['total_new_features'] = len(all_new_features)
    feature_stats['final_features'] = len(train_enhanced.columns)
    
    print(f"\n✅ 피처 엔지니어링 완료!")
    print(f"   전체 추가 피처: {len(all_new_features)}개")
    print(f"   최종 피처: {len(train_enhanced.columns)}개")
    
    # 분석 저장
    if save_analysis:
        save_feature_analysis(train, train_enhanced, test_enhanced, feature_stats)
    
    return train_enhanced, test_enhanced, feature_stats

def save_feature_analysis(original_train: pd.DataFrame, enhanced_train: pd.DataFrame,
                         enhanced_test: pd.DataFrame, feature_stats: Dict,
                         save_dir: str = "eda"):
    """피처 엔지니어링 분석 저장"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. 피처 그룹별 분포 시각화
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    plot_idx = 0
    
    for group_name, features in feature_stats['feature_groups'].items():
        if plot_idx >= len(axes) or not features:
            continue
            
        # 첫 번째 피처의 분포 시각화
        feature = features[0] if features else None
        if feature and feature in enhanced_train.columns:
            
            if enhanced_train[feature].dtype in ['int64', 'float64']:
                # 연속형 피처
                axes[plot_idx].hist(enhanced_train[feature].dropna(), bins=50, 
                                  alpha=0.7, label='Train', density=True)
                if feature in enhanced_test.columns:
                    axes[plot_idx].hist(enhanced_test[feature].dropna(), bins=50, 
                                      alpha=0.7, label='Test', density=True)
                axes[plot_idx].legend()
            else:
                # 범주형 피처
                train_counts = enhanced_train[feature].value_counts()
                axes[plot_idx].bar(range(len(train_counts)), train_counts.values)
                axes[plot_idx].set_xticks(range(len(train_counts)))
                axes[plot_idx].set_xticklabels(train_counts.index, rotation=45)
            
            axes[plot_idx].set_title(f'{group_name}: {feature}')
            plot_idx += 1
    
    # 빈 subplot 제거
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_adaptive_features_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 피처 상관관계 분석
    if '전력소비량(kWh)' in enhanced_train.columns:
        target_col = '전력소비량(kWh)'
        
        # 새로 추가된 피처들과 타겟의 상관관계
        new_features = feature_stats['added_features']
        numeric_new_features = []
        
        for feature in new_features:
            if (feature in enhanced_train.columns and 
                enhanced_train[feature].dtype in ['int64', 'float64']):
                numeric_new_features.append(feature)
        
        if numeric_new_features:
            correlations = enhanced_train[numeric_new_features + [target_col]].corr()[target_col]
            correlations = correlations.drop(target_col).sort_values(key=abs, ascending=False)
            
            # 상관관계 상위 20개 시각화
            top_correlations = correlations.head(20)
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_correlations.values]
            plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
            plt.yticks(range(len(top_correlations)), top_correlations.index)
            plt.xlabel('타겟과의 상관관계')
            plt.title('새로 추가된 피처들의 타겟 상관관계 (상위 20개)')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/new_features_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 상관관계 CSV 저장
            correlations.to_csv(f"{save_dir}/new_features_correlation.csv", 
                              header=['correlation'], encoding='utf-8-sig')
    
    # 3. 피처 리포트 저장
    report = ["=" * 80]
    report.append("Test 적응형 피처 엔지니어링 리포트")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"원본 피처 수: {feature_stats['original_features']}개")
    report.append(f"추가된 피처 수: {feature_stats['total_new_features']}개")
    report.append(f"최종 피처 수: {feature_stats['final_features']}개")
    report.append("")
    
    report.append("피처 그룹별 상세:")
    report.append("-" * 40)
    
    for group_name, features in feature_stats['feature_groups'].items():
        report.append(f"\n{group_name.upper()} ({len(features)}개):")
        for feature in features:
            report.append(f"  - {feature}")
    
    # Test 적응형 피처 설명
    report.append("\n\nTest 적응형 피처 설명:")
    report.append("-" * 40)
    report.append("1. SEASONAL: 8월 특화 시간/계절 피처")
    report.append("   - is_august, summer_peak_hours, cooling_intensity 등")
    report.append("2. TEMPERATURE: Test 온도 분포 기준 피처")
    report.append("   - temp_vs_test_mean, in_test_temp_core, high_temp_stress 등")
    report.append("3. HUMIDITY_COOLING: 냉방 관련 습도/불쾌지수 피처")
    report.append("   - discomfort_index, cooling_need_index 등")
    report.append("4. BUILDING_ADAPTIVE: Test 건물 분포 적응 피처")
    report.append("   - is_test_building_type, in_test_area_core 등")
    report.append("5. TEST_SIMILARITY: Test 데이터와의 유사도 피처")
    report.append("   - temporal_similarity, climate_similarity 등")
    report.append("6. INTERACTION: Test 특성 기반 상호작용 피처")
    report.append("   - midday_high_temp, hotel_high_temp 등")
    
    with open(f"{save_dir}/test_adaptive_features_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """메인 실행 함수"""
    
    # 데이터 로드
    train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    test = pd.read_csv("../method_07/test_building_merged.csv", encoding='utf-8-sig')
    
    print(f"원본 데이터 - Train: {len(train):,}행 {len(train.columns)}개 피처")
    print(f"               Test: {len(test):,}행 {len(test.columns)}개 피처")
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    # Test 적응형 피처 엔지니어링 적용
    enhanced_train, enhanced_test, feature_stats = apply_test_adaptive_engineering(
        train, test, save_analysis=True
    )
    
    # 결과 저장
    enhanced_train.to_csv(f"{eda_dir}/train_test_adaptive_features.csv", 
                         index=False, encoding='utf-8-sig')
    enhanced_test.to_csv(f"{eda_dir}/test_test_adaptive_features.csv", 
                        index=False, encoding='utf-8-sig')
    
    # 피처 목록 저장
    pd.DataFrame({
        'feature_name': feature_stats['added_features'],
        'group': sum([[group] * len(features) 
                     for group, features in feature_stats['feature_groups'].items()], [])
    }).to_csv(f"{eda_dir}/new_features_list.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Test 적응형 피처 엔지니어링 완료!")
    print(f"   결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    print(f"   Train: {len(enhanced_train.columns)}개 피처 (+{feature_stats['total_new_features']})")
    print(f"   Test: {len(enhanced_test.columns)}개 피처 (+{feature_stats['total_new_features']})")
    
    return enhanced_train, enhanced_test, feature_stats

if __name__ == "__main__":
    enhanced_train, enhanced_test, feature_stats = main()