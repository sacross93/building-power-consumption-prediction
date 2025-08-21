"""
Distribution Analysis: Train vs Test 분포 차이 분석 및 시각화

Train(6~7월) vs Test(8월) 데이터의 분포 차이를 정량화하고,
Test 분포에 맞는 Train 필터링을 위한 기준을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(
    train_path: str = "../method_07/train_building_merged.csv",
    test_path: str = "../method_07/test_building_merged.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test 데이터 로드"""
    train = pd.read_csv(train_path, encoding='utf-8-sig')
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    
    # 시간 파싱
    train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H', errors='coerce')
    test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H', errors='coerce')
    
    # 월 추출
    train['month'] = train['datetime'].dt.month
    test['month'] = test['datetime'].dt.month
    
    return train, test

def analyze_temporal_distribution(train: pd.DataFrame, test: pd.DataFrame, save_dir: str = "eda"):
    """시간별 분포 분석"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 월별 분포
    train_months = train['month'].value_counts().sort_index()
    test_months = test['month'].value_counts().sort_index()
    
    axes[0,0].bar(train_months.index, train_months.values, alpha=0.7, label='Train', color='blue')
    axes[0,0].bar(test_months.index, test_months.values, alpha=0.7, label='Test', color='red')
    axes[0,0].set_title('월별 데이터 분포')
    axes[0,0].set_xlabel('월')
    axes[0,0].set_ylabel('샘플 수')
    axes[0,0].legend()
    
    # 시간대별 분포
    train['hour'] = train['datetime'].dt.hour
    test['hour'] = test['datetime'].dt.hour
    
    train_hours = train['hour'].value_counts().sort_index()
    test_hours = test['hour'].value_counts().sort_index()
    
    axes[0,1].plot(train_hours.index, train_hours.values, label='Train', marker='o')
    axes[0,1].plot(test_hours.index, test_hours.values, label='Test', marker='s')
    axes[0,1].set_title('시간대별 데이터 분포')
    axes[0,1].set_xlabel('시간')
    axes[0,1].set_ylabel('샘플 수')
    axes[0,1].legend()
    
    # 요일별 분포
    train['weekday'] = train['datetime'].dt.weekday
    test['weekday'] = test['datetime'].dt.weekday
    
    train_weekdays = train['weekday'].value_counts().sort_index()
    test_weekdays = test['weekday'].value_counts().sort_index()
    
    axes[1,0].bar(train_weekdays.index, train_weekdays.values, alpha=0.7, label='Train', color='blue')
    axes[1,0].bar(test_weekdays.index, test_weekdays.values, alpha=0.7, label='Test', color='red')
    axes[1,0].set_title('요일별 데이터 분포')
    axes[1,0].set_xlabel('요일 (0=월요일)')
    axes[1,0].set_ylabel('샘플 수')
    axes[1,0].legend()
    
    # 건물유형별 분포
    train_types = train['건물유형'].value_counts()
    test_types = test['건물유형'].value_counts()
    
    all_types = list(set(train_types.index) | set(test_types.index))
    train_type_counts = [train_types.get(t, 0) for t in all_types]
    test_type_counts = [test_types.get(t, 0) for t in all_types]
    
    x = np.arange(len(all_types))
    width = 0.35
    
    axes[1,1].bar(x - width/2, train_type_counts, width, label='Train', alpha=0.7, color='blue')
    axes[1,1].bar(x + width/2, test_type_counts, width, label='Test', alpha=0.7, color='red')
    axes[1,1].set_title('건물유형별 데이터 분포')
    axes[1,1].set_xlabel('건물유형')
    axes[1,1].set_ylabel('샘플 수')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(all_types, rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/temporal_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'train_months': train_months,
        'test_months': test_months,
        'train_hours': train_hours,
        'test_hours': test_hours
    }

def analyze_weather_distribution(train: pd.DataFrame, test: pd.DataFrame, save_dir: str = "eda"):
    """기후 변수 분포 분석"""
    weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    
    # Test에만 있는 컬럼 제외
    weather_cols = [col for col in weather_cols if col in train.columns and col in test.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    distribution_stats = {}
    
    for i, col in enumerate(weather_cols):
        if i >= len(axes):
            break
            
        train_values = train[col].dropna()
        test_values = test[col].dropna()
        
        # 히스토그램
        axes[i].hist(train_values, bins=50, alpha=0.7, label='Train', density=True, color='blue')
        axes[i].hist(test_values, bins=50, alpha=0.7, label='Test', density=True, color='red')
        axes[i].set_title(f'{col} 분포 비교')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('밀도')
        axes[i].legend()
        
        # 통계 저장
        distribution_stats[col] = {
            'train_mean': train_values.mean(),
            'train_std': train_values.std(),
            'train_q25': train_values.quantile(0.25),
            'train_q75': train_values.quantile(0.75),
            'test_mean': test_values.mean(),
            'test_std': test_values.std(),
            'test_q25': test_values.quantile(0.25),
            'test_q75': test_values.quantile(0.75),
        }
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/weather_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return distribution_stats

def analyze_consumption_distribution(train: pd.DataFrame, save_dir: str = "eda"):
    """전력소비량 분포 분석 (Train only)"""
    if '전력소비량(kWh)' not in train.columns:
        return {}
    
    consumption = train['전력소비량(kWh)'].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 원본 분포
    axes[0,0].hist(consumption, bins=100, alpha=0.7, color='blue')
    axes[0,0].set_title('전력소비량 원본 분포')
    axes[0,0].set_xlabel('전력소비량(kWh)')
    axes[0,0].set_ylabel('빈도')
    
    # 로그 분포
    log_consumption = np.log1p(consumption)
    axes[0,1].hist(log_consumption, bins=100, alpha=0.7, color='green')
    axes[0,1].set_title('전력소비량 로그 분포')
    axes[0,1].set_xlabel('log1p(전력소비량)')
    axes[0,1].set_ylabel('빈도')
    
    # 건물유형별 분포
    for i, building_type in enumerate(train['건물유형'].unique()):
        if i >= 10:  # 최대 10개만
            break
        type_consumption = train[train['건물유형'] == building_type]['전력소비량(kWh)'].dropna()
        axes[1,0].hist(type_consumption, bins=30, alpha=0.5, label=building_type)
    
    axes[1,0].set_title('건물유형별 전력소비량 분포')
    axes[1,0].set_xlabel('전력소비량(kWh)')
    axes[1,0].set_ylabel('빈도')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 월별 평균 소비량
    monthly_consumption = train.groupby('month')['전력소비량(kWh)'].agg(['mean', 'std']).reset_index()
    axes[1,1].bar(monthly_consumption['month'], monthly_consumption['mean'], yerr=monthly_consumption['std'], alpha=0.7)
    axes[1,1].set_title('월별 평균 전력소비량')
    axes[1,1].set_xlabel('월')
    axes[1,1].set_ylabel('평균 전력소비량(kWh)')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/consumption_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'consumption_stats': {
            'mean': consumption.mean(),
            'std': consumption.std(),
            'q25': consumption.quantile(0.25),
            'q50': consumption.quantile(0.50),
            'q75': consumption.quantile(0.75),
            'q95': consumption.quantile(0.95),
            'q99': consumption.quantile(0.99)
        },
        'monthly_stats': monthly_consumption.to_dict('records')
    }

def calculate_distribution_shift_metrics(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """분포 차이 정량화 메트릭 계산"""
    from scipy import stats
    
    metrics = {}
    common_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    common_cols = [col for col in common_cols if col in train.columns and col in test.columns]
    
    for col in common_cols:
        train_values = train[col].dropna()
        test_values = test[col].dropna()
        
        # KS test (분포 차이 검정)
        ks_stat, ks_p = stats.ks_2samp(train_values, test_values)
        
        # Mean shift
        mean_shift = test_values.mean() - train_values.mean()
        
        # Std ratio
        std_ratio = test_values.std() / train_values.std() if train_values.std() > 0 else 1.0
        
        metrics[col] = {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mean_shift': mean_shift,
            'std_ratio': std_ratio,
            'train_range': (train_values.min(), train_values.max()),
            'test_range': (test_values.min(), test_values.max())
        }
    
    return metrics

def identify_outlier_samples(train: pd.DataFrame, test: pd.DataFrame, 
                           method: str = 'percentile', percentile: float = 0.95) -> pd.Index:
    """Test 분포 기준으로 Train 이상치 샘플 식별"""
    
    weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    weather_cols = [col for col in weather_cols if col in train.columns and col in test.columns]
    
    outlier_mask = pd.Series(False, index=train.index)
    
    for col in weather_cols:
        test_values = test[col].dropna()
        train_values = train[col]
        
        if method == 'percentile':
            # Test 분포의 percentile 범위 밖 샘플들
            lower_bound = test_values.quantile((1 - percentile) / 2)
            upper_bound = test_values.quantile(percentile + (1 - percentile) / 2)
            
            col_outliers = (train_values < lower_bound) | (train_values > upper_bound)
            outlier_mask |= col_outliers
            
        elif method == 'iqr':
            # Test IQR 기준
            q25 = test_values.quantile(0.25)
            q75 = test_values.quantile(0.75)
            iqr = q75 - q25
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            col_outliers = (train_values < lower_bound) | (train_values > upper_bound)
            outlier_mask |= col_outliers
    
    return train.index[outlier_mask]

def generate_distribution_report(train: pd.DataFrame, test: pd.DataFrame, 
                               save_path: str = "eda/distribution_analysis_report.txt"):
    """분포 분석 리포트 생성"""
    
    # 기본 통계
    report = ["=" * 80]
    report.append("Train vs Test 분포 분석 리포트")
    report.append("=" * 80)
    report.append("")
    
    # 데이터 기본 정보
    report.append(f"Train 데이터: {len(train):,}행")
    report.append(f"Test 데이터: {len(test):,}행")
    report.append(f"Train 기간: {train['datetime'].min()} ~ {train['datetime'].max()}")
    report.append(f"Test 기간: {test['datetime'].min()} ~ {test['datetime'].max()}")
    report.append("")
    
    # 분포 차이 메트릭
    shift_metrics = calculate_distribution_shift_metrics(train, test)
    report.append("분포 차이 메트릭:")
    report.append("-" * 40)
    
    for col, metrics in shift_metrics.items():
        report.append(f"\n{col}:")
        report.append(f"  - KS 통계량: {metrics['ks_statistic']:.4f} (p={metrics['ks_p_value']:.4f})")
        report.append(f"  - 평균 차이: {metrics['mean_shift']:.4f}")
        report.append(f"  - 표준편차 비율: {metrics['std_ratio']:.4f}")
        report.append(f"  - Train 범위: {metrics['train_range']}")
        report.append(f"  - Test 범위: {metrics['test_range']}")
    
    # 이상치 분석
    outliers_95 = identify_outlier_samples(train, test, 'percentile', 0.95)
    outliers_90 = identify_outlier_samples(train, test, 'percentile', 0.90)
    
    report.append("\n이상치 분석:")
    report.append("-" * 40)
    report.append(f"Test 95% 범위 밖 Train 샘플: {len(outliers_95):,}개 ({len(outliers_95)/len(train)*100:.2f}%)")
    report.append(f"Test 90% 범위 밖 Train 샘플: {len(outliers_90):,}개 ({len(outliers_90)/len(train)*100:.2f}%)")
    
    # 권장사항
    report.append("\n권장사항:")
    report.append("-" * 40)
    
    for col, metrics in shift_metrics.items():
        if metrics['ks_p_value'] < 0.05:
            report.append(f"- {col}: 분포 차이 유의함 (p<0.05) → 필터링/리웨이팅 필요")
        
        if abs(metrics['mean_shift']) > metrics.get('train_std', 1) * 0.5:
            report.append(f"- {col}: 평균 차이 큼 → 중요도 재조정 필요")
    
    # 파일 저장
    Path(save_path).parent.mkdir(exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    """메인 실행 함수"""
    # 데이터 로드
    train, test = load_data()
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    print("🔍 Train vs Test 분포 분석 시작...")
    
    # 시간별 분포 분석
    print("1. 시간별 분포 분석...")
    temporal_stats = analyze_temporal_distribution(train, test, eda_dir)
    
    # 기후 변수 분포 분석
    print("2. 기후 변수 분포 분석...")
    weather_stats = analyze_weather_distribution(train, test, eda_dir)
    
    # 전력소비량 분포 분석
    print("3. 전력소비량 분포 분석...")
    consumption_stats = analyze_consumption_distribution(train, eda_dir)
    
    # 분포 차이 정량화
    print("4. 분포 차이 메트릭 계산...")
    shift_metrics = calculate_distribution_shift_metrics(train, test)
    
    # 이상치 식별
    print("5. Test 기준 이상치 식별...")
    outliers_95 = identify_outlier_samples(train, test, 'percentile', 0.95)
    outliers_90 = identify_outlier_samples(train, test, 'percentile', 0.90)
    
    # 리포트 생성
    print("6. 분석 리포트 생성...")
    report = generate_distribution_report(train, test, f"{eda_dir}/distribution_analysis_report.txt")
    
    print(f"\n✅ 분석 완료! 결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    print(f"📊 주요 결과:")
    print(f"   - Train: {len(train):,}행, Test: {len(test):,}행")
    print(f"   - Test 95% 범위 밖 Train 샘플: {len(outliers_95):,}개 ({len(outliers_95)/len(train)*100:.1f}%)")
    print(f"   - Test 90% 범위 밖 Train 샘플: {len(outliers_90):,}개 ({len(outliers_90)/len(train)*100:.1f}%)")
    
    return {
        'temporal_stats': temporal_stats,
        'weather_stats': weather_stats,
        'consumption_stats': consumption_stats,
        'shift_metrics': shift_metrics,
        'outliers_95': outliers_95,
        'outliers_90': outliers_90
    }

if __name__ == "__main__":
    results = main()