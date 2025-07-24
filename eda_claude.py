#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건물 전력사용량 예측 AI 경진대회 - EDA (Exploratory Data Analysis)
Author: Claude
Date: 2025-07-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import matplotlib.font_manager as fm

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 결과 저장 폴더 확인
os.makedirs('visualizations', exist_ok=True)
os.makedirs('eda_results', exist_ok=True)

def save_text_result(content, filename):
    """텍스트 결과를 파일로 저장"""
    with open(f'eda_results/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"결과 저장: eda_results/{filename}")

def save_plot(filename):
    """그래프를 파일로 저장"""
    plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    print(f"시각화 저장: visualizations/{filename}")
    plt.close()

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")
    train = pd.read_csv('data/train.csv')
    building_info = pd.read_csv('data/building_info.csv')
    test = pd.read_csv('data/test.csv')
    
    # 일시 컬럼을 datetime으로 변환
    train['일시'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
    test['일시'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')
    
    return train, building_info, test

def basic_info_analysis(train, building_info, test):
    """기본 정보 분석"""
    print("\n=== 데이터 기본 정보 분석 ===")
    
    info_text = []
    info_text.append("=== 건물 전력사용량 예측 EDA 결과 ===\n")
    info_text.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Train 데이터 정보
    info_text.append(f"Train 데이터 크기: {train.shape}")
    info_text.append(f"Train 컬럼: {list(train.columns)}")
    info_text.append(f"Train 기간: {train['일시'].min()} ~ {train['일시'].max()}")
    info_text.append(f"건물 수: {train['건물번호'].nunique()}개")
    info_text.append("")
    
    # Building info 데이터 정보
    info_text.append(f"Building Info 데이터 크기: {building_info.shape}")
    info_text.append(f"Building Info 컬럼: {list(building_info.columns)}")
    info_text.append(f"건물 유형: {building_info['건물유형'].unique()}")
    info_text.append("")
    
    # Test 데이터 정보
    info_text.append(f"Test 데이터 크기: {test.shape}")
    info_text.append(f"Test 기간: {test['일시'].min()} ~ {test['일시'].max()}")
    info_text.append("")
    
    # 데이터 타입 정보
    info_text.append("=== 데이터 타입 정보 ===")
    info_text.append("Train 데이터 타입:")
    for col, dtype in train.dtypes.items():
        info_text.append(f"  {col}: {dtype}")
    info_text.append("")
    
    # 기본 통계량
    info_text.append("=== 전력소비량 기본 통계량 ===")
    power_stats = train['전력소비량(kWh)'].describe()
    for stat, value in power_stats.items():
        info_text.append(f"  {stat}: {value:.2f}")
    info_text.append("")
    
    result_text = "\n".join(info_text)
    print(result_text)
    save_text_result(result_text, 'basic_info.txt')
    
    return result_text

def missing_value_analysis(train, building_info):
    """결측치 분석"""
    print("\n=== 결측치 분석 ===")
    
    missing_text = []
    missing_text.append("=== 결측치 분석 결과 ===\n")
    
    # Train 데이터 결측치
    missing_text.append("Train 데이터 결측치:")
    train_missing = train.isnull().sum()
    for col, count in train_missing.items():
        missing_text.append(f"  {col}: {count}개 ({count/len(train)*100:.2f}%)")
    missing_text.append("")
    
    # Building info 결측치
    missing_text.append("Building Info 데이터 결측치:")
    building_missing = building_info.isnull().sum()
    for col, count in building_missing.items():
        missing_text.append(f"  {col}: {count}개 ({count/len(building_info)*100:.2f}%)")
    missing_text.append("")
    
    # '-' 값 확인 (building_info에서)
    missing_text.append("Building Info에서 '-' 값 확인:")
    for col in building_info.columns:
        if building_info[col].dtype == 'object':
            dash_count = (building_info[col] == '-').sum()
            if dash_count > 0:
                missing_text.append(f"  {col}: {dash_count}개")
    
    result_text = "\n".join(missing_text)
    print(result_text)
    save_text_result(result_text, 'missing_values.txt')

def power_consumption_analysis(train, building_info):
    """전력소비량 분석"""
    print("\n=== 전력소비량 분석 ===")
    
    # 전력소비량 분포 히스토그램
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(train['전력소비량(kWh)'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Power Consumption Distribution')
    plt.xlabel('Power Consumption (kWh)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(train['전력소비량(kWh)']), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Log Power Consumption Distribution')
    plt.xlabel('Log(Power Consumption + 1)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(train['전력소비량(kWh)'])
    plt.title('Power Consumption Box Plot')
    plt.ylabel('Power Consumption (kWh)')
    
    plt.tight_layout()
    save_plot('power_distribution.png')
    
    # 건물별 전력소비량
    building_power = train.groupby('건물번호')['전력소비량(kWh)'].agg(['mean', 'std', 'min', 'max']).reset_index()
    building_power = building_power.merge(building_info[['건물번호', '건물유형']], on='건물번호')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    building_power_sorted = building_power.sort_values('mean', ascending=True)
    plt.barh(range(len(building_power_sorted)), building_power_sorted['mean'])
    plt.yticks(range(len(building_power_sorted)), [f'Building {x}' for x in building_power_sorted['건물번호']])
    plt.title('Average Power Consumption by Building')
    plt.xlabel('Average Power Consumption (kWh)')
    
    plt.subplot(2, 2, 2)
    type_power = building_power.groupby('건물유형')['mean'].mean().sort_values(ascending=True)
    plt.barh(range(len(type_power)), type_power.values)
    plt.yticks(range(len(type_power)), type_power.index)
    plt.title('Average Power Consumption by Building Type')
    plt.xlabel('Average Power Consumption (kWh)')
    
    plt.subplot(2, 2, 3)
    plt.scatter(building_info['연면적(m2)'], building_power['mean'], alpha=0.6)
    plt.title('Power Consumption vs Floor Area')
    plt.xlabel('Floor Area (m2)')
    plt.ylabel('Average Power Consumption (kWh)')
    
    plt.subplot(2, 2, 4)
    plt.scatter(building_info['냉방면적(m2)'], building_power['mean'], alpha=0.6)
    plt.title('Power Consumption vs Cooling Area')
    plt.xlabel('Cooling Area (m2)')
    plt.ylabel('Average Power Consumption (kWh)')
    
    plt.tight_layout()
    save_plot('power_by_building.png')
    
    # 이상치 분석
    Q1 = train['전력소비량(kWh)'].quantile(0.25)
    Q3 = train['전력소비량(kWh)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train['전력소비량(kWh)'] < lower_bound) | 
                     (train['전력소비량(kWh)'] > upper_bound)]
    
    outlier_text = f"""=== 이상치 분석 결과 ===

전력소비량 사분위수:
  Q1: {Q1:.2f}
  Q3: {Q3:.2f}
  IQR: {IQR:.2f}
  하한: {lower_bound:.2f}
  상한: {upper_bound:.2f}

이상치 개수: {len(outliers)}개 ({len(outliers)/len(train)*100:.2f}%)
이상치 범위: {outliers['전력소비량(kWh)'].min():.2f} ~ {outliers['전력소비량(kWh)'].max():.2f}
"""
    
    print(outlier_text)
    save_text_result(outlier_text, 'outliers.txt')

def time_pattern_analysis(train):
    """시간 패턴 분석"""
    print("\n=== 시간 패턴 분석 ===")
    
    # 시간 관련 변수 생성
    train['hour'] = train['일시'].dt.hour
    train['day'] = train['일시'].dt.day
    train['month'] = train['일시'].dt.month
    train['weekday'] = train['일시'].dt.dayofweek
    
    plt.figure(figsize=(20, 12))
    
    # 시간대별 패턴
    plt.subplot(2, 3, 1)
    hourly_power = train.groupby('hour')['전력소비량(kWh)'].mean()
    plt.plot(hourly_power.index, hourly_power.values, marker='o')
    plt.title('Average Power Consumption by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Power Consumption (kWh)')
    plt.grid(True)
    
    # 요일별 패턴
    plt.subplot(2, 3, 2)
    weekday_power = train.groupby('weekday')['전력소비량(kWh)'].mean()
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.bar(weekdays, weekday_power.values)
    plt.title('Average Power Consumption by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Power Consumption (kWh)')
    
    # 월별 패턴
    plt.subplot(2, 3, 3)
    monthly_power = train.groupby('month')['전력소비량(kWh)'].mean()
    plt.plot(monthly_power.index, monthly_power.values, marker='o')
    plt.title('Average Power Consumption by Month')
    plt.xlabel('Month')
    plt.ylabel('Power Consumption (kWh)')
    plt.grid(True)
    
    # 히트맵: 시간 vs 요일
    plt.subplot(2, 3, 4)
    pivot_hour_weekday = train.pivot_table(values='전력소비량(kWh)', 
                                           index='hour', columns='weekday', aggfunc='mean')
    sns.heatmap(pivot_hour_weekday, cmap='YlOrRd', cbar_kws={'label': 'Power Consumption (kWh)'})
    plt.title('Power Consumption Heatmap (Hour vs Weekday)')
    plt.xlabel('Weekday')
    plt.ylabel('Hour')
    
    # 시계열 전체 트렌드 (샘플링)
    plt.subplot(2, 1, 2)
    sample_data = train.groupby('일시')['전력소비량(kWh)'].mean().reset_index()
    plt.plot(sample_data['일시'], sample_data['전력소비량(kWh)'], alpha=0.7)
    plt.title('Power Consumption Time Series')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (kWh)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    save_plot('time_patterns.png')

def weather_analysis(train):
    """기상 데이터 분석"""
    print("\n=== 기상 데이터 분석 ===")
    
    weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)']
    
    plt.figure(figsize=(20, 15))
    
    # 기상 변수별 전력소비량과의 상관관계
    for i, col in enumerate(weather_cols, 1):
        plt.subplot(3, 3, i)
        plt.scatter(train[col], train['전력소비량(kWh)'], alpha=0.1)
        plt.xlabel(col)
        plt.ylabel('Power Consumption (kWh)')
        plt.title(f'Power vs {col}')
        
        # 상관계수 계산
        corr = train[col].corr(train['전력소비량(kWh)'])
        plt.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 상관관계 히트맵
    plt.subplot(3, 3, 7)
    corr_matrix = train[weather_cols + ['전력소비량(kWh)']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                cbar_kws={'label': 'Correlation'})
    plt.title('Weather Variables Correlation Matrix')
    
    # 기온별 전력소비량 분포
    plt.subplot(3, 3, 8)
    temp_bins = pd.cut(train['기온(°C)'], bins=10)
    temp_power = train.groupby(temp_bins)['전력소비량(kWh)'].mean()
    temp_power.plot(kind='bar', rot=45)
    plt.title('Power Consumption by Temperature Range')
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Average Power Consumption (kWh)')
    
    # 습도별 전력소비량 분포
    plt.subplot(3, 3, 9)
    humidity_bins = pd.cut(train['습도(%)'], bins=10)
    humidity_power = train.groupby(humidity_bins)['전력소비량(kWh)'].mean()
    humidity_power.plot(kind='bar', rot=45)
    plt.title('Power Consumption by Humidity Range')
    plt.xlabel('Humidity Range (%)')
    plt.ylabel('Average Power Consumption (kWh)')
    
    plt.tight_layout()
    save_plot('weather_analysis.png')
    
    # 상관관계 분석 결과 저장
    weather_text = []
    weather_text.append("=== 기상 데이터 분석 결과 ===\n")
    weather_text.append("전력소비량과 기상 변수 간 상관관계:")
    for col in weather_cols:
        corr = train[col].corr(train['전력소비량(kWh)'])
        weather_text.append(f"  {col}: {corr:.4f}")
    
    weather_text.append("\n기상 변수 기본 통계량:")
    for col in weather_cols:
        stats = train[col].describe()
        weather_text.append(f"\n{col}:")
        weather_text.append(f"  평균: {stats['mean']:.2f}")
        weather_text.append(f"  표준편차: {stats['std']:.2f}")
        weather_text.append(f"  최소값: {stats['min']:.2f}")
        weather_text.append(f"  최대값: {stats['max']:.2f}")
    
    result_text = "\n".join(weather_text)
    print(result_text)
    save_text_result(result_text, 'weather_analysis.txt')

def building_analysis(train, building_info):
    """건물 정보 분석"""
    print("\n=== 건물 정보 분석 ===")
    
    # 건물별 평균 전력소비량 계산
    building_power = train.groupby('건물번호')['전력소비량(kWh)'].agg(['mean', 'std']).reset_index()
    building_merged = building_power.merge(building_info, on='건물번호')
    
    plt.figure(figsize=(20, 15))
    
    # 건물 유형별 전력소비량
    plt.subplot(3, 3, 1)
    type_power = building_merged.groupby('건물유형')['mean'].mean().sort_values(ascending=True)
    plt.barh(range(len(type_power)), type_power.values)
    plt.yticks(range(len(type_power)), type_power.index)
    plt.title('Average Power Consumption by Building Type')
    plt.xlabel('Average Power Consumption (kWh)')
    
    # 연면적과 전력소비량
    plt.subplot(3, 3, 2)
    plt.scatter(building_merged['연면적(m2)'], building_merged['mean'], alpha=0.6)
    plt.xlabel('Floor Area (m2)')
    plt.ylabel('Average Power Consumption (kWh)')
    plt.title('Power Consumption vs Floor Area')
    
    # 냉방면적과 전력소비량
    plt.subplot(3, 3, 3)
    plt.scatter(building_merged['냉방면적(m2)'], building_merged['mean'], alpha=0.6)
    plt.xlabel('Cooling Area (m2)')
    plt.ylabel('Average Power Consumption (kWh)')
    plt.title('Power Consumption vs Cooling Area')
    
    # 태양광 설비 영향
    plt.subplot(3, 3, 4)
    building_merged['has_solar'] = building_merged['태양광용량(kW)'] != '-'
    solar_power = building_merged.groupby('has_solar')['mean'].mean()
    plt.bar(['No Solar', 'Has Solar'], solar_power.values)
    plt.title('Power Consumption: Solar vs No Solar')
    plt.ylabel('Average Power Consumption (kWh)')
    
    # ESS 설비 영향
    plt.subplot(3, 3, 5)
    building_merged['has_ess'] = building_merged['ESS저장용량(kWh)'] != '-'
    ess_power = building_merged.groupby('has_ess')['mean'].mean()
    plt.bar(['No ESS', 'Has ESS'], ess_power.values)
    plt.title('Power Consumption: ESS vs No ESS')
    plt.ylabel('Average Power Consumption (kWh)')
    
    # 건물 유형별 분포
    plt.subplot(3, 3, 6)
    type_counts = building_info['건물유형'].value_counts()
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    plt.title('Building Type Distribution')
    
    # 연면적 분포
    plt.subplot(3, 3, 7)
    plt.hist(building_info['연면적(m2)'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Floor Area (m2)')
    plt.ylabel('Frequency')
    plt.title('Floor Area Distribution')
    
    # 건물 유형별 연면적 박스플롯
    plt.subplot(3, 3, 8)
    building_info.boxplot(column='연면적(m2)', by='건물유형', ax=plt.gca())
    plt.xticks(rotation=45)
    plt.title('Floor Area by Building Type')
    plt.suptitle('')
    
    # 태양광 용량 분포 (있는 경우만)
    plt.subplot(3, 3, 9)
    solar_data = building_info[building_info['태양광용량(kW)'] != '-']['태양광용량(kW)'].astype(float)
    if len(solar_data) > 0:
        plt.hist(solar_data, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Solar Capacity (kW)')
        plt.ylabel('Frequency')
        plt.title('Solar Capacity Distribution')
    
    plt.tight_layout()
    save_plot('building_analysis.png')
    
    # 분석 결과 텍스트
    building_text = []
    building_text.append("=== 건물 정보 분석 결과 ===\n")
    
    building_text.append("건물 유형별 평균 전력소비량:")
    for building_type, power in type_power.items():
        building_text.append(f"  {building_type}: {power:.2f} kWh")
    
    building_text.append(f"\n연면적과 전력소비량 상관계수: {building_merged['연면적(m2)'].corr(building_merged['mean']):.4f}")
    building_text.append(f"냉방면적과 전력소비량 상관계수: {building_merged['냉방면적(m2)'].corr(building_merged['mean']):.4f}")
    
    solar_count = (building_info['태양광용량(kW)'] != '-').sum()
    ess_count = (building_info['ESS저장용량(kWh)'] != '-').sum()
    building_text.append(f"\n태양광 설비 보유 건물: {solar_count}개 ({solar_count/len(building_info)*100:.1f}%)")
    building_text.append(f"ESS 설비 보유 건물: {ess_count}개 ({ess_count/len(building_info)*100:.1f}%)")
    
    result_text = "\n".join(building_text)
    print(result_text)
    save_text_result(result_text, 'building_analysis.txt')

def generate_insights(train, building_info):
    """종합 인사이트 생성"""
    print("\n=== 종합 인사이트 생성 ===")
    
    insights = []
    insights.append("=== 건물 전력사용량 예측 EDA 종합 인사이트 ===\n")
    
    insights.append("주요 발견사항:")
    insights.append("1. 전력소비량은 건물 유형별로 큰 차이를 보임")
    insights.append("2. 시간대별 패턴이 뚜렷함 (일반적으로 낮 시간대에 높음)")
    insights.append("3. 기온과 전력소비량 간에 상관관계 존재 (냉난방 영향)")
    insights.append("4. 건물 규모(연면적)와 전력소비량 간 양의 상관관계")
    insights.append("5. 태양광/ESS 설비가 있는 건물의 전력 사용 패턴 차이")
    
    insights.append("\n모델링 시 고려사항:")
    insights.append("1. 건물별 특성을 반영한 개별 모델 또는 건물 유형별 모델 고려")
    insights.append("2. 시간적 특성(시간, 요일, 계절) 피처 엔지니어링")
    insights.append("3. 기상 데이터의 상호작용 효과 고려")
    insights.append("4. 건물 규모 정규화 (연면적, 냉방면적 대비 전력소비량)")
    insights.append("5. 태양광/ESS 설비 보유 여부를 이진 변수로 활용")
    insights.append("6. 이상치 처리 방안 검토 필요")
    insights.append("7. SMAPE 평가지표 특성상 예측값이 0에 가까울 때 주의")
    
    insights.append("\n추가 분석 권장사항:")
    insights.append("1. 건물별 시계열 패턴의 안정성 분석")
    insights.append("2. 계절별 전력 사용 패턴 상세 분석")
    insights.append("3. 기상 변수 간 다중공선성 검토")
    insights.append("4. 휴일/평일 구분에 따른 전력 사용 패턴 분석")
    
    result_text = "\n".join(insights)
    print(result_text)
    save_text_result(result_text, 'insights.txt')

def main():
    """메인 실행 함수"""
    print("건물 전력사용량 예측 EDA 시작")
    print("=" * 50)
    
    # 데이터 로드
    train, building_info, test = load_data()
    
    # 분석 실행
    basic_info_analysis(train, building_info, test)
    missing_value_analysis(train, building_info)
    power_consumption_analysis(train, building_info)
    time_pattern_analysis(train)
    weather_analysis(train)
    building_analysis(train, building_info)
    generate_insights(train, building_info)
    
    print("\n" + "=" * 50)
    print("EDA 완료!")
    print("결과 파일:")
    print("- 시각화: visualizations/ 폴더")
    print("- 텍스트 분석: eda_results/ 폴더")

if __name__ == "__main__":
    main()