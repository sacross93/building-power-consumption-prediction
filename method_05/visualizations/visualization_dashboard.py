"""
Data Preprocessing Visualization Dashboard
=========================================

현재 solution.py의 데이터 전처리 과정을 분석하고
전처리 후 데이터 분포와 변수간 상관관계를 종합적으로 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# Import preprocessing functions
import sys
sys.path.append('..')
from solution import load_data, engineer_features


class DataPreprocessingVisualizer:
    """데이터 전처리 과정 시각화 클래스."""
    
    def __init__(self, data_dir='../../data', output_dir='./'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_raw = None
        self.test_raw = None
        self.train_processed = None
        self.test_processed = None
        
    def load_and_process_data(self):
        """데이터 로드 및 전처리."""
        print("=" * 60)
        print("DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        
        # 원본 데이터 로드
        train_path = self.data_dir / 'train.csv'
        test_path = self.data_dir / 'test.csv'
        building_path = self.data_dir / 'building_info.csv'
        
        print("Loading raw data...")
        self.train_raw, self.test_raw = load_data(train_path, test_path, building_path)
        print(f"Raw train shape: {self.train_raw.shape}")
        print(f"Raw test shape: {self.test_raw.shape}")
        
        # 전처리 적용
        print("Applying feature engineering...")
        self.train_processed, self.test_processed = engineer_features(
            self.train_raw.copy(), self.test_raw.copy()
        )
        print(f"Processed train shape: {self.train_processed.shape}")
        print(f"Processed test shape: {self.test_processed.shape}")
        
        # 전처리 요약 정보
        raw_cols = set(self.train_raw.columns)
        processed_cols = set(self.train_processed.columns)
        new_features = processed_cols - raw_cols
        
        print(f"\nFeature Engineering Summary:")
        print(f"Original features: {len(raw_cols)}")
        print(f"New features added: {len(new_features)}")
        print(f"Total features: {len(processed_cols)}")
        
        return new_features
    
    def create_preprocessing_flowchart(self):
        """전처리 과정 플로우차트 생성."""
        print("\nCreating preprocessing flowchart...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 전처리 단계별 박스
        steps = [
            {"pos": (1, 9), "text": "1. Raw Data\nLoading", "color": "lightblue"},
            {"pos": (3, 9), "text": "2. Column\nRenaming", "color": "lightgreen"},
            {"pos": (5, 9), "text": "3. Building Info\nMerging", "color": "lightcoral"},
            {"pos": (7, 9), "text": "4. DateTime\nParsing", "color": "lightyellow"},
            {"pos": (9, 9), "text": "5. Temporal\nFeatures", "color": "lightpink"},
            
            {"pos": (1, 7), "text": "6. Missing Value\nImputation", "color": "lightsteelblue"},
            {"pos": (3, 7), "text": "7. Weather\nApproximation", "color": "lightseagreen"},
            {"pos": (5, 7), "text": "8. Building\nStatistics", "color": "lightsalmon"},
            {"pos": (7, 7), "text": "9. Peak Hour\nAnalysis", "color": "lightgoldenrodyellow"},
            {"pos": (9, 7), "text": "10. Enhanced\nFeatures", "color": "plum"},
            
            {"pos": (2, 5), "text": "11. Temperature\nInteractions", "color": "lightcyan"},
            {"pos": (4, 5), "text": "12. Building\nEfficiency", "color": "lightgray"},
            {"pos": (6, 5), "text": "13. Cyclical\nEncoding", "color": "lightblue"},
            {"pos": (8, 5), "text": "14. Final\nProcessed Data", "color": "gold"},
        ]
        
        # 박스 그리기
        for step in steps:
            x, y = step["pos"]
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                               facecolor=step["color"], 
                               edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, step["text"], ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # 화살표 연결
        arrows = [
            ((1.4, 9), (2.6, 9)),  # 1->2
            ((3.4, 9), (4.6, 9)),  # 2->3
            ((5.4, 9), (6.6, 9)),  # 3->4
            ((7.4, 9), (8.6, 9)),  # 4->5
            ((1, 8.7), (1, 7.3)),  # 5->6
            ((3, 8.7), (3, 7.3)),  # 6->7
            ((5, 8.7), (5, 7.3)),  # 7->8
            ((7, 8.7), (7, 7.3)),  # 8->9
            ((9, 8.7), (9, 7.3)),  # 9->10
            ((1.4, 7), (1.6, 5.3)),  # branching to bottom
            ((3.4, 7), (3.6, 5.3)),
            ((5.4, 7), (5.6, 5.3)),
            ((7.4, 7), (7.6, 5.3)),
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
        
        ax.set_title('Data Preprocessing Pipeline Flowchart', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessing_analysis/preprocessing_flowchart.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_data_distributions(self):
        """데이터 분포 분석."""
        print("\nAnalyzing data distributions...")
        
        # 1. 원본 vs 전처리 후 타겟 변수 분포
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 전력소비량 분포 (원본)
        target_raw = self.train_raw['전력소비량(kWh)']
        ax1.hist(target_raw, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Raw Power Consumption Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Power Consumption (kWh)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 전력소비량 분포 (로그 스케일)
        ax2.hist(np.log1p(target_raw), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Log-scaled Power Consumption Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Log(Power Consumption + 1)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 건물별 평균 전력소비량
        building_means = self.train_raw.groupby('건물번호')['전력소비량(kWh)'].mean()
        ax3.bar(range(len(building_means)), sorted(building_means.values), 
               color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_title('Average Power Consumption by Building', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Building ID (sorted)')
        ax3.set_ylabel('Average Power Consumption (kWh)')
        ax3.grid(True, alpha=0.3)
        
        # 건물 유형별 분포
        building_type_stats = self.train_raw.groupby('building_type')['전력소비량(kWh)'].agg(['mean', 'std'])
        ax4.bar(building_type_stats.index, building_type_stats['mean'], 
               yerr=building_type_stats['std'], color='gold', alpha=0.7, 
               edgecolor='black', capsize=5)
        ax4.set_title('Power Consumption by Building Type', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Building Type')
        ax4.set_ylabel('Average Power Consumption (kWh)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_distribution/raw_data_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 시간별 패턴 분석
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 시간별 평균 전력소비량
        hourly_pattern = self.train_processed.groupby('hour')['전력소비량(kWh)'].mean()
        ax1.plot(hourly_pattern.index, hourly_pattern.values, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Hourly Power Consumption Pattern', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Power Consumption (kWh)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # 요일별 패턴
        weekday_pattern = self.train_processed.groupby('weekday')['전력소비량(kWh)'].mean()
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(weekday_names, weekday_pattern.values, color=['lightblue' if i < 5 else 'lightcoral' for i in range(7)])
        ax2.set_title('Weekday Power Consumption Pattern', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Power Consumption (kWh)')
        ax2.grid(True, alpha=0.3)
        
        # 월별 패턴
        monthly_pattern = self.train_processed.groupby('month')['전력소비량(kWh)'].mean()
        ax3.plot(monthly_pattern.index, monthly_pattern.values, marker='s', linewidth=2, markersize=8, color='green')
        ax3.set_title('Monthly Power Consumption Pattern', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Power Consumption (kWh)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks([6, 7, 8])
        
        # 주말 vs 평일
        weekend_pattern = self.train_processed.groupby('is_weekend')['전력소비량(kWh)'].mean()
        ax4.bar(['Weekday', 'Weekend'], weekend_pattern.values, 
               color=['lightsteelblue', 'lightsalmon'])
        ax4.set_title('Weekday vs Weekend Consumption', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Power Consumption (kWh)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_distribution/temporal_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 날씨 변수 분포
        weather_cols = ['temp', 'humidity', 'wind_speed', 'rainfall']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(weather_cols):
            if col in self.train_processed.columns:
                data = self.train_processed[col].dropna()
                axes[i].hist(data, bins=30, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
                axes[i].set_title(f'{col.title()} Distribution', fontsize=14, fontweight='bold')
                axes[i].set_xlabel(col.title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # 기본 통계
                axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
                axes[i].axvline(data.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_distribution/weather_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_engineering_effects(self):
        """피처 엔지니어링 효과 분석."""
        print("\nAnalyzing feature engineering effects...")
        
        # 새로 생성된 피처들 분석
        new_features = [
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
            'bld_hour_mean', 'bld_wd_mean', 'bld_month_mean',
            'temp_squared', 'temp_cooling_need', 'temp_heating_need',
            'area_ratio', 'pv_per_area', 'humidity_temp'
        ]
        
        # 존재하는 피처만 필터링
        existing_features = [f for f in new_features if f in self.train_processed.columns]
        
        if len(existing_features) >= 6:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(existing_features[:6]):
                data = self.train_processed[feature].dropna()
                axes[i].hist(data, bins=30, alpha=0.7, color=plt.cm.tab10(i), edgecolor='black')
                axes[i].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # 통계 정보
                axes[i].text(0.7, 0.8, f'Mean: {data.mean():.3f}\nStd: {data.std():.3f}', 
                           transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            
            plt.suptitle('Engineered Features Distribution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'data_distribution/feature_distributions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_correlation_analysis(self):
        """상관관계 분석."""
        print("\nCreating correlation analysis...")
        
        # 수치형 변수만 선택
        numeric_cols = self.train_processed.select_dtypes(include=[np.number]).columns
        numeric_data = self.train_processed[numeric_cols].copy()
        
        # 타겟 변수와의 상관관계
        target_corr = numeric_data.corr()['전력소비량(kWh)'].drop('전력소비량(kWh)').sort_values(key=abs, ascending=False)
        
        # 1. 타겟과의 상관관계 Top 20
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        top_corr = target_corr.head(20)
        colors = ['red' if x > 0 else 'blue' for x in top_corr.values]
        bars = ax1.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_corr)))
        ax1.set_yticklabels(top_corr.index, fontsize=10)
        ax1.set_xlabel('Correlation with Power Consumption')
        ax1.set_title('Top 20 Features Correlated with Power Consumption', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color='black', linewidth=1)
        
        # 수치 라벨 추가
        for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
            ax1.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        # 2. 전체 상관관계 히트맵 (주요 변수만)
        important_features = target_corr.head(15).index.tolist() + ['전력소비량(kWh)']
        corr_matrix = numeric_data[important_features].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Correlation Matrix (Top Features)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlations/feature_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 건물 유형별 상관관계
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        building_types = self.train_processed['building_type'].value_counts().head(4).index
        key_features = ['temp', 'humidity', 'hour', 'weekday']
        
        for i, building_type in enumerate(building_types):
            building_data = self.train_processed[self.train_processed['building_type'] == building_type]
            
            if len(key_features) <= len(building_data.columns):
                available_features = [f for f in key_features if f in building_data.columns]
                available_features.append('전력소비량(kWh)')
                
                if len(available_features) > 1:
                    corr_data = building_data[available_features].corr()
                    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                               center=0, ax=axes[i], cbar_kws={'shrink': 0.8})
                    axes[i].set_title(f'{building_type} Correlations', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlations/building_type_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return target_corr
    
    def create_preprocessing_summary(self, new_features, target_corr):
        """전처리 요약 보고서 생성."""
        print("\nCreating preprocessing summary report...")
        
        summary_path = self.output_dir / 'preprocessing_analysis/preprocessing_summary.md'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Data Preprocessing Analysis Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Raw training data**: {self.train_raw.shape[0]:,} rows × {self.train_raw.shape[1]} columns\n")
            f.write(f"- **Processed training data**: {self.train_processed.shape[0]:,} rows × {self.train_processed.shape[1]} columns\n")
            f.write(f"- **New features created**: {len(new_features)}\n")
            f.write(f"- **Target variable**: 전력소비량(kWh)\n\n")
            
            f.write("## Preprocessing Pipeline\n\n")
            f.write("### 1. Data Loading & Merging\n")
            f.write("- Load train, test, and building metadata\n")
            f.write("- Rename Korean columns to English\n")
            f.write("- Merge building information\n")
            f.write("- Handle missing values in building fields\n\n")
            
            f.write("### 2. DateTime Processing\n")
            f.write("- Parse datetime column (YYYYMMDD HH format)\n")
            f.write("- Extract temporal features: year, month, day, hour, weekday\n")
            f.write("- Create weekend indicator\n")
            f.write("- Generate cyclical encodings (sin/cos) for temporal features\n\n")
            
            f.write("### 3. Missing Value Imputation\n")
            f.write("- Use training set medians for building metadata\n")
            f.write("- Approximate missing weather data in test set using August averages\n\n")
            
            f.write("### 4. Feature Engineering\n")
            f.write("- **Building Statistics**: Hour/weekday/month-specific consumption patterns\n")
            f.write("- **Peak Hour Analysis**: Individual building peak identification\n")
            f.write("- **Temperature Features**: Cooling/heating needs, squared terms\n")
            f.write("- **Building Efficiency**: Area ratios, PV capacity per area\n")
            f.write("- **Weather Interactions**: Temperature-humidity, rainfall-wind combinations\n\n")
            
            f.write("## Key Statistics\n\n")
            f.write("### Target Variable (전력소비량)\n")
            target_stats = self.train_processed['전력소비량(kWh)'].describe()
            f.write(f"- **Mean**: {target_stats['mean']:.2f} kWh\n")
            f.write(f"- **Median**: {target_stats['50%']:.2f} kWh\n")
            f.write(f"- **Std**: {target_stats['std']:.2f} kWh\n")
            f.write(f"- **Range**: {target_stats['min']:.2f} - {target_stats['max']:.2f} kWh\n\n")
            
            f.write("### Building Types\n")
            building_type_counts = self.train_processed['building_type'].value_counts()
            for building_type, count in building_type_counts.items():
                f.write(f"- **{building_type}**: {count:,} records\n")
            f.write("\n")
            
            f.write("### Top Correlations with Target\n")
            f.write("| Feature | Correlation |\n")
            f.write("|---------|-------------|\n")
            for feature, corr in target_corr.head(10).items():
                f.write(f"| {feature} | {corr:.4f} |\n")
            f.write("\n")
            
            f.write("## New Features Created\n\n")
            if new_features:
                f.write("### Temporal Features\n")
                temporal_features = [f for f in new_features if any(x in f for x in ['hour', 'weekday', 'month', 'sin', 'cos', 'peak'])]
                for feature in temporal_features:
                    f.write(f"- `{feature}`\n")
                
                f.write("\n### Building-specific Features\n")
                building_features = [f for f in new_features if any(x in f for x in ['bld_', 'building_', 'area_', 'pv_'])]
                for feature in building_features:
                    f.write(f"- `{feature}`\n")
                
                f.write("\n### Weather/Temperature Features\n")
                weather_features = [f for f in new_features if any(x in f for x in ['temp_', 'humidity_', 'rain_', 'wind_'])]
                for feature in weather_features:
                    f.write(f"- `{feature}`\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("### Strengths\n")
            f.write("- Comprehensive temporal feature engineering\n")
            f.write("- Building-specific statistical features\n")
            f.write("- Proper handling of missing values\n")
            f.write("- Cyclical encoding for temporal patterns\n\n")
            
            f.write("### Potential Improvements\n")
            f.write("- Consider lag features (with proper time series validation)\n")
            f.write("- Add more weather interaction terms\n")
            f.write("- Implement building-type specific feature engineering\n")
            f.write("- Consider external data sources (holidays, events)\n\n")
            
            f.write("## Visualization Files Generated\n\n")
            f.write("### Data Distribution\n")
            f.write("- `raw_data_distributions.png`: Original data distributions\n")
            f.write("- `temporal_patterns.png`: Time-based consumption patterns\n")
            f.write("- `weather_distributions.png`: Weather variable distributions\n")
            f.write("- `feature_distributions.png`: Engineered feature distributions\n\n")
            
            f.write("### Correlations\n")
            f.write("- `feature_correlations.png`: Feature correlations with target\n")
            f.write("- `building_type_correlations.png`: Building-type specific correlations\n\n")
            
            f.write("### Process Analysis\n")
            f.write("- `preprocessing_flowchart.png`: Complete preprocessing pipeline\n")
            f.write("- `preprocessing_summary.md`: This comprehensive report\n")
    
    def run_complete_analysis(self):
        """전체 분석 실행."""
        print("🚀 Starting comprehensive data preprocessing analysis...")
        
        # 1. 데이터 로드 및 전처리
        new_features = self.load_and_process_data()
        
        # 2. 전처리 플로우차트 생성
        self.create_preprocessing_flowchart()
        
        # 3. 데이터 분포 분석
        self.analyze_data_distributions()
        
        # 4. 피처 엔지니어링 효과 분석
        self.analyze_feature_engineering_effects()
        
        # 5. 상관관계 분석
        target_corr = self.create_correlation_analysis()
        
        # 6. 요약 보고서 생성
        self.create_preprocessing_summary(new_features, target_corr)
        
        print("\n✅ Complete preprocessing analysis finished!")
        print(f"📊 Visualizations saved to: {self.output_dir}")
        print(f"📄 Summary report: preprocessing_analysis/preprocessing_summary.md")
        
        return {
            'new_features': new_features,
            'target_correlations': target_corr,
            'train_shape': self.train_processed.shape,
            'test_shape': self.test_processed.shape
        }


if __name__ == "__main__":
    # 시각화 분석 실행
    visualizer = DataPreprocessingVisualizer()
    results = visualizer.run_complete_analysis()
    
    print(f"\n🎯 Analysis Results Summary:")
    print(f"- New features created: {len(results['new_features'])}")
    print(f"- Training data shape: {results['train_shape']}")
    print(f"- Test data shape: {results['test_shape']}")
    print(f"- Top correlation: {results['target_correlations'].iloc[0]:.4f}")