"""
Advanced Feature Engineering
============================

건물 클러스터링, 시간 세분화, 고급 상호작용 피처
추가 성능 향상을 위한 정교한 피처 엔지니어링
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features
from improved_preprocessing import ImprovedPreprocessor


class AdvancedFeatureEngineer:
    """고급 피처 엔지니어링 클래스."""
    
    def __init__(self):
        self.building_clusters = {}
        self.time_patterns = {}
        self.interaction_features = []
        
    def create_building_clusters(self, train_df):
        """건물 클러스터링 기반 피처."""
        print("🏢 Creating building clusters...")
        
        # 건물별 기본 통계
        building_stats = train_df.groupby('건물번호').agg({
            '전력소비량(kWh)': ['mean', 'std', 'min', 'max'],
            'total_area': 'first',
            'cooling_area': 'first',
            'pv_capacity': 'first',
            'building_type': 'first'
        }).round(2)
        
        # 컬럼명 정리
        building_stats.columns = ['power_mean', 'power_std', 'power_min', 'power_max',
                                'total_area', 'cooling_area', 'pv_capacity', 'building_type']
        
        # 수치형 피처만 클러스터링에 사용
        cluster_features = ['power_mean', 'power_std', 'total_area', 'cooling_area', 'pv_capacity']
        
        # 결측값 처리
        for col in cluster_features:
            building_stats[col] = building_stats[col].fillna(building_stats[col].median())
        
        # 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(building_stats[cluster_features])
        
        # 클러스터링 (건물을 5개 그룹으로)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        building_stats['cluster'] = kmeans.fit_predict(features_scaled)
        
        # 클러스터별 특성 분석
        print("  Cluster characteristics:")
        for cluster in range(5):
            cluster_buildings = building_stats[building_stats['cluster'] == cluster]
            avg_power = cluster_buildings['power_mean'].mean()
            dominant_type = cluster_buildings['building_type'].mode().iloc[0]
            count = len(cluster_buildings)
            
            print(f"    Cluster {cluster}: {count} buildings, avg power: {avg_power:.1f}, "
                  f"dominant type: {dominant_type}")
        
        self.building_clusters = building_stats['cluster'].to_dict()
        return building_stats
    
    def create_advanced_time_features(self, df):
        """고급 시간 기반 피처."""
        print("⏰ Creating advanced time features...")
        df = df.copy()
        
        # 15분, 30분 단위 패턴
        df['quarter_hour'] = (df['datetime'].dt.minute // 15).astype(int)
        df['half_hour'] = (df['datetime'].dt.minute // 30).astype(int)
        
        # 주차별 패턴 (월 내)
        df['week_of_month'] = ((df['datetime'].dt.day - 1) // 7 + 1).astype(int)
        
        # 계절 세분화
        df['season_fine'] = pd.cut(df['datetime'].dt.dayofyear, 
                                  bins=[0, 79, 171, 263, 354, 366],
                                  labels=['late_winter', 'spring', 'summer', 'autumn', 'early_winter'])
        
        # 비즈니스 캘린더
        df['is_month_start'] = (df['datetime'].dt.day <= 3).astype(int)
        df['is_month_end'] = (df['datetime'].dt.day >= 28).astype(int)
        df['is_week_start'] = (df['datetime'].dt.weekday == 0).astype(int)
        df['is_week_end'] = (df['datetime'].dt.weekday == 4).astype(int)
        
        # 전력 소비 패턴별 시간대
        df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['weekday'] < 5)).astype(int)
        df['lunch_peak'] = ((df['hour'] >= 11) & (df['hour'] <= 14) & (df['weekday'] < 5)).astype(int)
        df['evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (df['weekday'] < 5)).astype(int)
        df['night_minimum'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        df['weekend_leisure'] = ((df['hour'] >= 10) & (df['hour'] <= 16) & (df['weekday'] >= 5)).astype(int)
        
        # 계절-시간 상호작용
        df['summer_cooling_peak'] = ((df['month'].isin([7, 8])) & 
                                    (df['hour'].between(13, 16))).astype(int)
        df['winter_heating_peak'] = ((df['month'].isin([12, 1, 2])) & 
                                    (df['hour'].between(7, 9))).astype(int)
        
        return df
    
    def create_weather_advanced_features(self, df):
        """고급 날씨 피처."""
        print("🌤️  Creating advanced weather features...")
        df = df.copy()
        
        if 'temp' in df.columns and 'humidity' in df.columns:
            # 체감온도 (더 정확한 공식)
            df['apparent_temp'] = df['temp'] + 0.33 * (df['humidity'] / 100 * 6.105 * 
                                 np.exp(17.27 * df['temp'] / (237.7 + df['temp']))) - 0.7 * df.get('wind_speed', 0) - 4.0
            
            # 불쾌지수 개선
            df['discomfort_enhanced'] = 1.8 * df['temp'] - 0.55 * (1 - df['humidity'] / 100) * \
                                       (1.8 * df['temp'] - 26) + 32
            
            # 냉방도일/난방도일 (시간별)
            df['hourly_cdd'] = np.maximum(df['temp'] - 18, 0)
            df['hourly_hdd'] = np.maximum(18 - df['temp'], 0)
            
            # 온도 변화율 (전 시간 대비)
            df['temp_change'] = df.groupby('건물번호')['temp'].diff().fillna(0)
            
            # 온도 변동성 (rolling std) - 인덱스 문제 해결
            temp_volatility = df.groupby('건물번호')['temp'].rolling(24, min_periods=1).std()
            temp_volatility.index = temp_volatility.index.droplevel(0)  # 멀티인덱스 제거
            df['temp_volatility'] = temp_volatility.reindex(df.index).fillna(0)
            
            # 습도 범주
            df['humidity_comfort'] = pd.cut(df['humidity'], 
                                          bins=[0, 30, 40, 60, 70, 100],
                                          labels=['very_dry', 'dry', 'comfortable', 'humid', 'very_humid'])
            
            # 온습도 조합 효과
            df['temp_humidity_zone'] = pd.cut(df['temp'], bins=[0, 18, 23, 28, 100], 
                                            labels=['cold', 'cool', 'warm', 'hot']).astype(str) + '_' + \
                                      df['humidity_comfort'].astype(str)
        
        return df
    
    def create_building_specific_features(self, df, building_clusters):
        """건물별 특화 피처."""
        print("🏗️  Creating building-specific features...")
        df = df.copy()
        
        # 클러스터 정보 추가
        df['building_cluster'] = df['건물번호'].map(building_clusters)
        df['building_cluster'] = df['building_cluster'].fillna(0).astype(int)
        
        # 클러스터별 특화 피처
        for cluster in range(5):
            cluster_mask = (df['building_cluster'] == cluster)
            df[f'is_cluster_{cluster}'] = cluster_mask.astype(int)
            
            # 클러스터별 시간 패턴
            if cluster_mask.sum() > 0:
                df[f'cluster_{cluster}_hour_factor'] = df[f'is_cluster_{cluster}'] * df['hour'] / 24
                df[f'cluster_{cluster}_weekend_factor'] = df[f'is_cluster_{cluster}'] * df['is_weekend']
        
        # 건물 효율성 지표
        if 'total_area' in df.columns and 'pv_capacity' in df.columns:
            # power_mean 계산 (rolling operation safe handling)
            if '전력소비량(kWh)' in df.columns:
                power_mean = df.groupby('건물번호')['전력소비량(kWh)'].rolling(24*7, min_periods=1).mean()
                power_mean.index = power_mean.index.droplevel(0)
                power_mean = power_mean.reindex(df.index).fillna(df.get('전력소비량(kWh)', 0))
            else:
                power_mean = df.get('power_mean', 0)
            
            df['building_efficiency_score'] = (
                df['pv_capacity'] / np.maximum(df['total_area'], 1) * 1000 +
                df.get('area_ratio', 0) * 100 -
                power_mean / 1000
            )
            
            df['green_building_score'] = (df['pv_capacity'] > 0).astype(int) * 2 + \
                                        (df.get('area_ratio', 0) > 0.8).astype(int)
        
        # 건물 타입별 특화 시간 패턴
        building_type_time_patterns = {
            'IDC(전화국)': {'peak_hours': [0, 1, 2, 22, 23], 'factor': 1.2},
            '백화점': {'peak_hours': [10, 11, 14, 15, 18, 19], 'factor': 1.5},
            '병원': {'peak_hours': [8, 9, 14, 15, 20, 21], 'factor': 1.1},
            '학교': {'peak_hours': [9, 10, 13, 14, 15, 16], 'factor': 1.3},
            '호텔': {'peak_hours': [7, 8, 19, 20, 21, 22], 'factor': 1.2}
        }
        
        for building_type, pattern in building_type_time_patterns.items():
            type_mask = (df.get('building_type', '') == building_type)
            peak_hours_mask = df['hour'].isin(pattern['peak_hours'])
            
            df[f'{building_type.lower()}_peak_time'] = (type_mask & peak_hours_mask).astype(int)
            df[f'{building_type.lower()}_efficiency'] = type_mask.astype(int) * pattern['factor']
        
        return df
    
    def create_interaction_features(self, df):
        """고급 상호작용 피처."""
        print("🔗 Creating interaction features...")
        df = df.copy()
        
        # 시간-날씨 상호작용
        if 'temp' in df.columns:
            df['morning_temp'] = (df['hour'] <= 10).astype(int) * df['temp']
            df['afternoon_temp'] = (df['hour'].between(12, 16)).astype(int) * df['temp']
            df['evening_temp'] = (df['hour'] >= 18).astype(int) * df['temp']
            
            # 주말-온도 패턴
            df['weekend_temp'] = df['is_weekend'] * df['temp']
            
            # 계절-온도 편차
            seasonal_temp_mean = df.groupby('month')['temp'].transform('mean')
            df['temp_seasonal_deviation'] = df['temp'] - seasonal_temp_mean
        
        # 건물-시간 고차 상호작용
        if 'total_area' in df.columns:
            df['area_hour_interaction'] = df['total_area'] * df['hour'] / (24 * 1000)
            df['area_weekend_interaction'] = df['total_area'] * df['is_weekend'] / 1000
            
            # 건물 크기별 시간 민감도
            df['large_building_peak'] = (df['total_area'] > 50000).astype(int) * df['hour_peak_flag']
            df['small_building_efficiency'] = (df['total_area'] < 10000).astype(int) * \
                                            (1 - df['hour_peak_flag'])
        
        # PV-날씨-시간 상호작용
        if 'pv_capacity' in df.columns and 'temp' in df.columns:
            # 태양광 발전 예상 시간대 (낮 시간)
            solar_hours = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
            df['pv_solar_potential'] = df['pv_capacity'] * solar_hours * (df['temp'] + 10) / 100
            
            # 태양광 절약 효과
            df['pv_cooling_offset'] = df['pv_capacity'] * df.get('temp_cooling_need', 0) / 1000
        
        # 3차 상호작용 (선별적)
        if all(col in df.columns for col in ['temp', 'humidity', 'hour']):
            df['temp_humidity_hour'] = df['temp'] * df['humidity'] * df['hour'] / 10000
        
        # 건물클러스터-시간-날씨 상호작용
        if 'building_cluster' in df.columns and 'temp' in df.columns:
            for cluster in range(5):
                cluster_mask = (df['building_cluster'] == cluster)
                df[f'cluster_{cluster}_temp_hour'] = cluster_mask.astype(int) * df['temp'] * df['hour'] / 100
        
        return df
    
    def feature_importance_analysis(self, df, target_col='전력소비량(kWh)'):
        """피처 중요도 분석."""
        print("📊 Analyzing feature importance...")
        
        if target_col not in df.columns:
            print("  Target column not found, skipping importance analysis")
            return None
        
        # 수치형 피처만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if len(numeric_cols) == 0:
            print("  No numeric features found")
            return None
        
        # 상관관계 계산
        correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.abs().sort_values(ascending=False)
        
        print(f"  Top 10 correlated features:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            print(f"    {i+1:2d}. {feature}: {corr:.4f}")
        
        return correlations
    
    def apply_advanced_feature_engineering(self, train_df, test_df):
        """전체 고급 피처 엔지니어링 적용."""
        print("=" * 80)
        print("ADVANCED FEATURE ENGINEERING")
        print("=" * 80)
        
        # 1. 기본 피처 엔지니어링 먼저 적용
        print("1. Applying basic feature engineering...")
        train_fe, test_fe = engineer_features(train_df.copy(), test_df.copy())
        
        print(f"   After basic FE - Train: {train_fe.shape}, Test: {test_fe.shape}")
        
        # 2. 건물 클러스터링
        building_stats = self.create_building_clusters(train_fe)
        
        # 3. 고급 시간 피처
        train_fe = self.create_advanced_time_features(train_fe)
        test_fe = self.create_advanced_time_features(test_fe)
        
        # 4. 고급 날씨 피처
        train_fe = self.create_weather_advanced_features(train_fe)
        test_fe = self.create_weather_advanced_features(test_fe)
        
        # 5. 건물별 특화 피처
        train_fe = self.create_building_specific_features(train_fe, self.building_clusters)
        test_fe = self.create_building_specific_features(test_fe, self.building_clusters)
        
        # 6. 상호작용 피처
        train_fe = self.create_interaction_features(train_fe)
        test_fe = self.create_interaction_features(test_fe)
        
        # 7. 피처 중요도 분석
        correlations = self.feature_importance_analysis(train_fe)
        
        print(f"\n✅ Advanced feature engineering completed!")
        print(f"   Final shapes - Train: {train_fe.shape}, Test: {test_fe.shape}")
        print(f"   Added {train_fe.shape[1] - train_df.shape[1]} new features")
        
        return train_fe, test_fe, correlations
    
    def create_feature_engineering_report(self, correlations, output_dir='./visualizations/'):
        """피처 엔지니어링 리포트."""
        report_path = Path(output_dir) / 'advanced_feature_engineering_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Advanced Feature Engineering Report\n\n")
            
            f.write("## Feature Categories Created\n\n")
            f.write("### 1. Building Clustering Features\n")
            f.write("- K-means clustering (5 clusters) based on power patterns\n")
            f.write("- Cluster-specific time factors\n")
            f.write("- Building efficiency scores\n\n")
            
            f.write("### 2. Advanced Time Features\n")
            f.write("- 15-minute and 30-minute patterns\n")
            f.write("- Business calendar features\n")
            f.write("- Season-time interactions\n")
            f.write("- Building type specific time patterns\n\n")
            
            f.write("### 3. Weather Enhancement\n")
            f.write("- Apparent temperature with wind chill\n")
            f.write("- Enhanced discomfort index\n")
            f.write("- Temperature volatility and change rates\n")
            f.write("- Humidity comfort zones\n\n")
            
            f.write("### 4. Building-Specific Features\n")
            f.write("- Green building scores\n")
            f.write("- Building type time patterns\n")
            f.write("- PV solar potential calculations\n\n")
            
            f.write("### 5. Interaction Features\n")
            f.write("- Time-weather interactions\n")
            f.write("- Building-time-weather 3-way interactions\n")
            f.write("- Cluster-specific patterns\n\n")
            
            if correlations is not None:
                f.write("## Top Correlations with Target\n\n")
                f.write("| Rank | Feature | Correlation |\n")
                f.write("|------|---------|-------------|\n")
                for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
                    f.write(f"| {i} | {feature} | {corr:.4f} |\n")
            
            f.write(f"\n## Cluster Analysis\n\n")
            f.write("Buildings were clustered into 5 groups based on:\n")
            f.write("- Average power consumption\n")
            f.write("- Power consumption variability\n")
            f.write("- Building physical characteristics\n\n")
            
            f.write("## Expected Impact\n\n")
            f.write("- **Improved temporal modeling**: Finer time granularity\n")
            f.write("- **Better building segmentation**: Cluster-based features\n")
            f.write("- **Enhanced weather sensitivity**: Advanced weather indices\n")
            f.write("- **Richer interactions**: Multi-way feature combinations\n")
        
        print(f"📄 Feature engineering report saved: {report_path}")


def test_advanced_feature_engineering():
    """고급 피처 엔지니어링 테스트."""
    print("Testing advanced feature engineering...")
    
    # 데이터 로드
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    
    # 고급 피처 엔지니어링 적용
    engineer = AdvancedFeatureEngineer()
    train_advanced, test_advanced, correlations = engineer.apply_advanced_feature_engineering(
        train_df, test_df
    )
    
    # 리포트 생성
    engineer.create_feature_engineering_report(correlations)
    
    return train_advanced, test_advanced, engineer


if __name__ == "__main__":
    train_advanced, test_advanced, engineer = test_advanced_feature_engineering()
    print(f"\n🎯 Advanced feature engineering test completed!")
    print(f"📊 Ready for model training with {train_advanced.shape[1]} features!")