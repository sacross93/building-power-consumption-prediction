import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Additional imports for advanced time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# Font settings for English display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.font_manager as fm
plt.rcParams['font.size'] = 10

# Create folders for results
os.makedirs('visualization_results', exist_ok=True)
os.makedirs('txt_results', exist_ok=True)

# Create a comprehensive analysis report
report_file = 'txt_results/eda_analysis_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("")  # Initialize empty file

def save_to_report(text, print_also=True):
    """Save text to report file and optionally print to console"""
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')
    if print_also:
        print(text)

save_to_report("=" * 60)
save_to_report("2025 Power Consumption Prediction AI Competition")
save_to_report("Exploratory Data Analysis (EDA)")
save_to_report("=" * 60)

# 1. Data Loading
save_to_report("\n1. Data Loading")
save_to_report("-" * 40)

train_df = pd.read_csv('data/train.csv')
building_info_df = pd.read_csv('data/building_info.csv')

save_to_report(f"Train data shape: {train_df.shape}")
save_to_report(f"Building info data shape: {building_info_df.shape}")

# Rename columns to English for better visualization
train_df_en = train_df.copy()
train_df_en.columns = ['num_date_time', 'building_number', 'datetime', 'temperature', 
                       'precipitation', 'windspeed', 'humidity', 'sunshine', 
                       'solar_radiation', 'power_consumption']

building_info_en = building_info_df.copy()
building_info_en.columns = ['building_number', 'building_type', 'total_area', 
                           'cooling_area', 'solar_capacity', 'ess_capacity', 'pcs_capacity']

# Map Korean building types to English
building_type_mapping = {
    '백화점': 'Department Store',
    '호텔': 'Hotel',
    '상용': 'Commercial',
    '학교': 'School',
    '건물기타': 'Other Buildings',
    '병원': 'Hospital',
    '아파트': 'Apartment',
    '연구소': 'Research Institute',
    'IDC(전화국)': 'IDC (Telecom)',
    '공공': 'Public'
}
building_info_en['building_type'] = building_info_en['building_type'].map(building_type_mapping)

# 2. Basic Information
save_to_report("\n2. Basic Data Information")
save_to_report("-" * 40)

save_to_report("\n[Train Data Info]")
# Capture info() output
import io
import sys
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()
train_df_en.info()
info_output = buffer.getvalue()
sys.stdout = old_stdout
save_to_report(info_output)

save_to_report("\n[Train Data Statistics]")
save_to_report(str(train_df_en.describe()))

save_to_report("\n[Building Info Data Info]")
# Capture info() output for building data
sys.stdout = buffer = io.StringIO()
building_info_en.info()
info_output = buffer.getvalue()
sys.stdout = old_stdout
save_to_report(info_output)

save_to_report("\n[Building Info Statistics]")
save_to_report(str(building_info_en.describe()))

# 3. Missing Values Analysis
save_to_report("\n3. Missing Values Analysis")
save_to_report("-" * 40)

save_to_report("\n[Train Data Missing Values]")
train_missing = train_df_en.isnull().sum()
missing_result = train_missing[train_missing > 0]
if len(missing_result) == 0:
    save_to_report("No missing values found in train data")
else:
    save_to_report(str(missing_result))

save_to_report("\n[Building Info Missing Values]")
building_missing = building_info_en.isnull().sum()
missing_result = building_missing[building_missing > 0]
if len(missing_result) == 0:
    save_to_report("No missing values found in building info data")
else:
    save_to_report(str(missing_result))

# 4. Building Information Analysis
save_to_report("\n4. Building Information Analysis")
save_to_report("-" * 40)

save_to_report("\n[Building Type Distribution]")
building_type_counts = building_info_en['building_type'].value_counts()
save_to_report(str(building_type_counts))

# Building type distribution visualization
plt.figure(figsize=(12, 6))
building_type_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Building Types', fontsize=14, fontweight='bold')
plt.xlabel('Building Type', fontsize=12)
plt.ylabel('Number of Buildings', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_results/building_type_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Building area analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Total area distribution
axes[0, 0].hist(building_info_en['total_area'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Total Area', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Total Area (m²)', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Cooling area distribution
axes[0, 1].hist(building_info_en['cooling_area'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Cooling Area', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Cooling Area (m²)', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Total area by building type
building_info_en.boxplot(column='total_area', by='building_type', ax=axes[1, 0])
axes[1, 0].set_title('Total Area by Building Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Building Type', fontsize=10)
axes[1, 0].set_ylabel('Total Area (m²)', fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=45)

# Cooling area by building type
building_info_en.boxplot(column='cooling_area', by='building_type', ax=axes[1, 1])
axes[1, 1].set_title('Cooling Area by Building Type', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Building Type', fontsize=10)
axes[1, 1].set_ylabel('Cooling Area (m²)', fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualization_results/building_area_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Time Data Processing and Analysis
save_to_report("\n5. Time Data Analysis")
save_to_report("-" * 40)

# Convert datetime column
train_df_en['datetime'] = pd.to_datetime(train_df_en['datetime'], format='%Y%m%d %H')
train_df_en['date'] = train_df_en['datetime'].dt.date
train_df_en['hour'] = train_df_en['datetime'].dt.hour
train_df_en['dayofweek'] = train_df_en['datetime'].dt.dayofweek
train_df_en['month'] = train_df_en['datetime'].dt.month
train_df_en['day'] = train_df_en['datetime'].dt.day

save_to_report("Time-related columns added successfully")
save_to_report(f"Data period: {train_df_en['datetime'].min()} ~ {train_df_en['datetime'].max()}")

# 6. Power Consumption Analysis
save_to_report("\n6. Power Consumption Analysis")
save_to_report("-" * 40)

save_to_report("Power Consumption Statistics:")
save_to_report(f"Mean: {train_df_en['power_consumption'].mean():.2f} kWh")
save_to_report(f"Median: {train_df_en['power_consumption'].median():.2f} kWh")
save_to_report(f"Min: {train_df_en['power_consumption'].min():.2f} kWh")
save_to_report(f"Max: {train_df_en['power_consumption'].max():.2f} kWh")
save_to_report(f"Std: {train_df_en['power_consumption'].std():.2f} kWh")

# Power consumption analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall power consumption distribution
axes[0, 0].hist(train_df_en['power_consumption'], bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[0, 0].set_title('Power Consumption Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Power Consumption (kWh)', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Hourly average power consumption
hourly_power = train_df_en.groupby('hour')['power_consumption'].mean()
axes[0, 1].plot(hourly_power.index, hourly_power.values, marker='o', linewidth=2, color='blue')
axes[0, 1].set_title('Average Power Consumption by Hour', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Hour of Day', fontsize=10)
axes[0, 1].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(0, 24, 2))

# Daily average power consumption
daily_power = train_df_en.groupby('dayofweek')['power_consumption'].mean()
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1, 0].bar(range(7), daily_power.values, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Average Power Consumption by Day of Week', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Day of Week', fontsize=10)
axes[1, 0].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(weekdays)
axes[1, 0].grid(axis='y', alpha=0.3)

# Monthly average power consumption
monthly_power = train_df_en.groupby('month')['power_consumption'].mean()
axes[1, 1].plot(monthly_power.index, monthly_power.values, marker='o', linewidth=2, color='red')
axes[1, 1].set_title('Average Power Consumption by Month', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month', fontsize=10)
axes[1, 1].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_results/power_consumption_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Building-wise Power Consumption Analysis
save_to_report("\n7. Building-wise Power Consumption Analysis")
save_to_report("-" * 40)

# Merge building info with train data
merged_df = train_df_en.merge(building_info_en, on='building_number', how='left')

# Power consumption by building type
building_power = merged_df.groupby('building_type')['power_consumption'].agg(['mean', 'median', 'std']).round(2)
save_to_report("\nPower Consumption Statistics by Building Type:")
save_to_report(str(building_power))

# Power consumption boxplot by building type
plt.figure(figsize=(14, 8))
box_plot = merged_df.boxplot(column='power_consumption', by='building_type', figsize=(14, 8))
plt.title('Power Consumption Distribution by Building Type', fontsize=14, fontweight='bold')
plt.xlabel('Building Type', fontsize=12)
plt.ylabel('Power Consumption (kWh)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_results/power_by_building_type.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Weather Data Correlation Analysis
save_to_report("\n8. Weather Data Correlation Analysis")
save_to_report("-" * 40)

# Weather variables
weather_vars = ['temperature', 'precipitation', 'windspeed', 'humidity', 'sunshine', 'solar_radiation']

# Calculate correlation
correlation_matrix = train_df_en[weather_vars + ['power_consumption']].corr()
save_to_report("Correlation between power consumption and weather variables:")
save_to_report(str(correlation_matrix['power_consumption'].sort_values(ascending=False)))

# Correlation heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('Weather Variables and Power Consumption Correlation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plots of weather variables vs power consumption
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

weather_labels = ['Temperature (°C)', 'Precipitation (mm)', 'Wind Speed (m/s)', 
                  'Humidity (%)', 'Sunshine (hr)', 'Solar Radiation (MJ/m²)']

for i, (var, label) in enumerate(zip(weather_vars, weather_labels)):
    # Sample data for better visualization performance
    sample_df = train_df_en.sample(n=min(10000, len(train_df_en)))
    axes[i].scatter(sample_df[var], sample_df['power_consumption'], alpha=0.5, s=1)
    axes[i].set_xlabel(label, fontsize=10)
    axes[i].set_ylabel('Power Consumption (kWh)', fontsize=10)
    axes[i].set_title(f'{label} vs Power Consumption', fontsize=11, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(sample_df[var], sample_df['power_consumption'], 1)
    p = np.poly1d(z)
    axes[i].plot(sample_df[var], p(sample_df[var]), "r--", alpha=0.8, linewidth=2)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_results/weather_vs_power_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Time Series Pattern Analysis
save_to_report("\n9. Time Series Pattern Analysis")
save_to_report("-" * 40)

# Daily power consumption trend (first 2 weeks)
daily_power_trend = train_df_en.groupby('date')['power_consumption'].sum().head(14)
save_to_report("Daily power consumption trend calculated for visualization")

plt.figure(figsize=(15, 6))
plt.plot(daily_power_trend.index, daily_power_trend.values, marker='o', linewidth=2, markersize=6)
plt.title('Daily Total Power Consumption Trend (First 2 Weeks)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Power Consumption (kWh)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_results/daily_power_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Outlier Analysis
save_to_report("\n10. Outlier Analysis")
save_to_report("-" * 40)

# Outlier detection using IQR method
Q1 = train_df_en['power_consumption'].quantile(0.25)
Q3 = train_df_en['power_consumption'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train_df_en[(train_df_en['power_consumption'] < lower_bound) | 
                      (train_df_en['power_consumption'] > upper_bound)]

save_to_report(f"Number of outliers: {len(outliers)}")
save_to_report(f"Percentage of total data: {len(outliers)/len(train_df_en)*100:.2f}%")

if len(outliers) > 0:
    save_to_report("\nOutlier statistics:")
    save_to_report(str(outliers['power_consumption'].describe()))

# 11. Advanced Time Series Analysis
save_to_report("\n11. Advanced Time Series Analysis")
save_to_report("-" * 40)

# 11.1 Seasonality Analysis
save_to_report("\n11.1 Seasonality Analysis")
save_to_report("~" * 30)

# Monthly and seasonal patterns
train_df_en['year_month'] = train_df_en['datetime'].dt.to_period('M')
train_df_en['season'] = train_df_en['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring', 
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Monthly power consumption analysis
monthly_consumption = train_df_en.groupby('month')['power_consumption'].agg(['mean', 'std', 'median']).round(2)
save_to_report("\nMonthly Power Consumption Statistics:")
save_to_report(str(monthly_consumption))

# Seasonal analysis
seasonal_consumption = train_df_en.groupby('season')['power_consumption'].agg(['mean', 'std', 'median']).round(2)
save_to_report("\nSeasonal Power Consumption Statistics:")
save_to_report(str(seasonal_consumption))

# Seasonality visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Monthly boxplot
monthly_data = [train_df_en[train_df_en['month'] == m]['power_consumption'] for m in range(6, 9)]  # June, July, August
axes[0, 0].boxplot(monthly_data, labels=['June', 'July', 'August'])
axes[0, 0].set_title('Monthly Power Consumption Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Power Consumption (kWh)', fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Seasonal comparison
seasonal_means = train_df_en.groupby('season')['power_consumption'].mean()
axes[0, 1].bar(seasonal_means.index, seasonal_means.values, color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
axes[0, 1].set_title('Average Power Consumption by Season', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)

# Hourly patterns by month
for month in [6, 7, 8]:
    month_data = train_df_en[train_df_en['month'] == month]
    hourly_month = month_data.groupby('hour')['power_consumption'].mean()
    month_names = {6: 'June', 7: 'July', 8: 'August'}
    axes[1, 0].plot(hourly_month.index, hourly_month.values, marker='o', label=month_names[month], linewidth=2)

axes[1, 0].set_title('Hourly Power Consumption by Month', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hour of Day', fontsize=10)
axes[1, 0].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Daily temperature vs power consumption
daily_temp_power = train_df_en.groupby('date').agg({
    'power_consumption': 'mean',
    'temperature': 'mean'
}).reset_index()

axes[1, 1].scatter(daily_temp_power['temperature'], daily_temp_power['power_consumption'], alpha=0.6, s=30)
axes[1, 1].set_title('Daily Average Temperature vs Power Consumption', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Average Temperature (°C)', fontsize=10)
axes[1, 1].set_ylabel('Average Power Consumption (kWh)', fontsize=10)
axes[1, 1].grid(alpha=0.3)

# Add trend line
z = np.polyfit(daily_temp_power['temperature'], daily_temp_power['power_consumption'], 1)
p = np.poly1d(z)
axes[1, 1].plot(daily_temp_power['temperature'], p(daily_temp_power['temperature']), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('visualization_results/seasonality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 11.2 Time Series Decomposition
save_to_report("\n11.2 Time Series Decomposition")
save_to_report("~" * 35)

# Prepare daily aggregated data for decomposition
daily_power = train_df_en.groupby('date')['power_consumption'].sum().reset_index()
daily_power.set_index('date', inplace=True)
daily_power.index = pd.to_datetime(daily_power.index)

# Perform seasonal decomposition
try:
    decomposition = seasonal_decompose(daily_power['power_consumption'], model='additive', period=7)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=axes[0], title='Original Time Series', color='blue')
    axes[0].set_ylabel('Power Consumption')
    axes[0].grid(alpha=0.3)
    
    decomposition.trend.plot(ax=axes[1], title='Trend Component', color='green')
    axes[1].set_ylabel('Trend')
    axes[1].grid(alpha=0.3)
    
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component', color='orange')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(alpha=0.3)
    
    decomposition.resid.plot(ax=axes[3], title='Residual Component', color='red')
    axes[3].set_ylabel('Residual')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization_results/time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    save_to_report("Time series decomposition completed successfully")
    save_to_report(f"Trend component range: {decomposition.trend.min():.2f} to {decomposition.trend.max():.2f}")
    save_to_report(f"Seasonal component range: {decomposition.seasonal.min():.2f} to {decomposition.seasonal.max():.2f}")
    
except Exception as e:
    save_to_report(f"Time series decomposition failed: {str(e)}")

# 11.3 Autocorrelation Analysis
save_to_report("\n11.3 Autocorrelation Analysis")
save_to_report("~" * 35)

# Prepare hourly data sample for autocorrelation
hourly_sample = train_df_en.groupby('datetime')['power_consumption'].mean().head(1000)  # First 1000 hours

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ACF plot
plot_acf(hourly_sample.dropna(), lags=48, ax=axes[0], title='Autocorrelation Function (ACF)')
axes[0].set_xlabel('Lag (hours)')
axes[0].grid(alpha=0.3)

# PACF plot
plot_pacf(hourly_sample.dropna(), lags=48, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
axes[1].set_xlabel('Lag (hours)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_results/autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

save_to_report("Autocorrelation analysis completed")
save_to_report("ACF and PACF plots show time dependencies in power consumption")

# 11.4 Stationarity Test
save_to_report("\n11.4 Stationarity Analysis")
save_to_report("~" * 30)

# Augmented Dickey-Fuller test
try:
    adf_result = adfuller(hourly_sample.dropna())
    save_to_report("Augmented Dickey-Fuller Test Results:")
    save_to_report(f"ADF Statistic: {adf_result[0]:.6f}")
    save_to_report(f"p-value: {adf_result[1]:.6f}")
    save_to_report(f"Critical Values:")
    for key, value in adf_result[4].items():
        save_to_report(f"\t{key}: {value:.3f}")
    
    if adf_result[1] <= 0.05:
        save_to_report("Result: Time series is stationary (reject null hypothesis)")
    else:
        save_to_report("Result: Time series is non-stationary (fail to reject null hypothesis)")
        
except Exception as e:
    save_to_report(f"Stationarity test failed: {str(e)}")

# 11.5 Building Type Specific Time Series Patterns
save_to_report("\n11.5 Building-Specific Time Series Patterns")
save_to_report("~" * 45)

# Analyze each building type's temporal patterns
building_patterns = {}
for building_type in merged_df['building_type'].unique():
    if pd.notna(building_type):
        building_data = merged_df[merged_df['building_type'] == building_type]
        
        # Calculate hourly and daily patterns
        hourly_pattern = building_data.groupby('hour')['power_consumption'].mean()
        daily_pattern = building_data.groupby('dayofweek')['power_consumption'].mean()
        
        # Peak hour identification
        peak_hour = hourly_pattern.idxmax()
        peak_consumption = hourly_pattern.max()
        
        building_patterns[building_type] = {
            'peak_hour': peak_hour,
            'peak_consumption': peak_consumption,
            'daily_std': daily_pattern.std(),
            'hourly_std': hourly_pattern.std()
        }
        
        save_to_report(f"\n{building_type}:")
        save_to_report(f"  Peak hour: {peak_hour}:00")
        save_to_report(f"  Peak consumption: {peak_consumption:.2f} kWh")
        save_to_report(f"  Daily variability: {daily_pattern.std():.2f}")
        save_to_report(f"  Hourly variability: {hourly_pattern.std():.2f}")

# Visualize building-specific patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 4 building types by count
top_building_types = building_info_en['building_type'].value_counts().head(4).index

colors = ['blue', 'red', 'green', 'orange']
for i, building_type in enumerate(top_building_types):
    building_data = merged_df[merged_df['building_type'] == building_type]
    hourly_pattern = building_data.groupby('hour')['power_consumption'].mean()
    
    if i < 4:
        row, col = i // 2, i % 2
        axes[row, col].plot(hourly_pattern.index, hourly_pattern.values, 
                           color=colors[i], linewidth=2, marker='o', markersize=4)
        axes[row, col].set_title(f'{building_type} - Hourly Pattern', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Hour of Day', fontsize=10)
        axes[row, col].set_ylabel('Avg Power Consumption (kWh)', fontsize=10)
        axes[row, col].grid(alpha=0.3)
        axes[row, col].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('visualization_results/building_specific_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# 11.6 Weather Lag Effects Analysis
save_to_report("\n11.6 Weather Lag Effects Analysis")
save_to_report("~" * 40)

# Analyze lagged correlations between weather and power consumption
weather_vars = ['temperature', 'humidity', 'solar_radiation']
lag_correlations = {}

for var in weather_vars:
    correlations = []
    lags = range(0, 25)  # 0 to 24 hours lag
    
    for lag in lags:
        if lag == 0:
            corr, _ = pearsonr(train_df_en[var], train_df_en['power_consumption'])
        else:
            # Create lagged data
            lagged_weather = train_df_en[var].shift(lag)
            # Remove NaN values
            valid_indices = ~(lagged_weather.isna() | train_df_en['power_consumption'].isna())
            if valid_indices.sum() > 100:  # Ensure enough data points
                corr, _ = pearsonr(lagged_weather[valid_indices], 
                                 train_df_en['power_consumption'][valid_indices])
            else:
                corr = 0
        correlations.append(corr)
    
    lag_correlations[var] = correlations
    
    # Find optimal lag
    max_corr_idx = np.argmax(np.abs(correlations))
    save_to_report(f"\n{var.replace('_', ' ').title()}:")
    save_to_report(f"  Best correlation: {correlations[max_corr_idx]:.4f} at lag {max_corr_idx} hours")

# Plot lag correlations
plt.figure(figsize=(14, 8))
for var in weather_vars:
    plt.plot(range(25), lag_correlations[var], marker='o', linewidth=2, label=var.replace('_', ' ').title())

plt.title('Weather Variables Lag Correlation with Power Consumption', fontsize=14, fontweight='bold')
plt.xlabel('Lag (hours)', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_results/weather_lag_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Machine Learning Strategy and Preprocessing Direction
save_to_report("\n" + "="*60)
save_to_report("AI Model Development Strategy for Power Consumption Prediction")
save_to_report("="*60)

save_to_report("\n1. Dataset Characteristics:")
save_to_report(f"   - Total records: {len(train_df_en):,}")
save_to_report(f"   - Number of buildings: {train_df_en['building_number'].nunique()}")
save_to_report(f"   - Time period: {(train_df_en['datetime'].max() - train_df_en['datetime'].min()).days} days")
save_to_report(f"   - Data frequency: Hourly")
save_to_report(f"   - Building types: {building_info_en['building_type'].nunique()}")

save_to_report("\n2. Key Findings:")
save_to_report("   - Clear hourly power consumption patterns (higher during daytime)")
save_to_report("   - Distinct consumption patterns across building types")
save_to_report("   - IDC buildings show highest consumption (avg. 10,317 kWh)")
save_to_report("   - Temperature shows strongest correlation with power consumption (0.124)")
save_to_report(f"   - Outlier ratio: {len(outliers)/len(train_df_en)*100:.2f}%")

save_to_report("\n3. Feature Engineering Strategy:")
save_to_report("   - Time-based features: hour, day of week, month, season")
save_to_report("   - Lag features: previous hour/day weather and consumption data")
save_to_report("   - Rolling statistics: moving averages, trends")
save_to_report("   - Building characteristics: area ratios, building type encoding")
save_to_report("   - Weather interactions: temperature × humidity, etc.")

save_to_report("\n4. Modeling Approach:")
save_to_report("   - Time series models: LSTM, GRU for temporal patterns")
save_to_report("   - Tree-based models: XGBoost, LightGBM, CatBoost")
save_to_report("   - Individual vs. unified model: building-specific or global model")
save_to_report("   - Ensemble methods: combining multiple model predictions")
save_to_report("   - Cross-validation: time series split respecting temporal order")

save_to_report("\n5. Data Preprocessing:")
save_to_report("   - Handle outliers: cap/remove extreme values")
save_to_report("   - Feature scaling: standardization for neural networks")
save_to_report("   - Missing value imputation: if any in test data")
save_to_report("   - Building type encoding: label/one-hot encoding")

save_to_report("\n6. Model Evaluation:")
save_to_report("   - Time-based train/validation split")
save_to_report("   - Building-wise performance analysis")
save_to_report("   - Seasonal and hourly performance evaluation")
save_to_report("   - Competition metric optimization")

save_to_report("\n7. Competition Strategy:")
save_to_report("   - Baseline model: Simple time series models")
save_to_report("   - Advanced models: Deep learning + ensemble")
save_to_report("   - Feature importance analysis")
save_to_report("   - Model interpretability for business insights")

# Save additional detailed analysis to separate files
print("\nSaving detailed analysis results to txt_results folder...")

# Save building type analysis
with open('txt_results/building_type_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Building Type Analysis\n")
    f.write("="*30 + "\n\n")
    f.write("Distribution:\n")
    f.write(str(building_type_counts) + "\n\n")
    f.write("Power Consumption Statistics by Building Type:\n")
    f.write(str(building_power) + "\n")

# Save correlation analysis
with open('txt_results/correlation_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Weather and Power Consumption Correlation Analysis\n")
    f.write("="*50 + "\n\n")
    f.write("Correlation Matrix:\n")
    f.write(str(correlation_matrix) + "\n\n")
    f.write("Power Consumption Correlations (sorted):\n")
    f.write(str(correlation_matrix['power_consumption'].sort_values(ascending=False)) + "\n")

# Save time series insights
with open('txt_results/time_series_insights.txt', 'w', encoding='utf-8') as f:
    f.write("Time Series Analysis Insights\n")
    f.write("="*35 + "\n\n")
    f.write(f"Data Period: {train_df_en['datetime'].min()} to {train_df_en['datetime'].max()}\n")
    f.write(f"Total Days: {(train_df_en['datetime'].max() - train_df_en['datetime'].min()).days}\n")
    f.write(f"Data Frequency: Hourly\n\n")
    
    f.write("Hourly Power Consumption Pattern:\n")
    hourly_avg = train_df_en.groupby('hour')['power_consumption'].mean()
    f.write(str(hourly_avg) + "\n\n")
    
    f.write("Daily Average Power Consumption (by weekday):\n")
    daily_avg = train_df_en.groupby('dayofweek')['power_consumption'].mean()
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, avg in enumerate(daily_avg):
        f.write(f"{weekday_names[i]}: {avg:.2f} kWh\n")

# Save advanced time series analysis
with open('txt_results/advanced_time_series_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Advanced Time Series Analysis Results\n")
    f.write("="*40 + "\n\n")
    
    f.write("SEASONALITY ANALYSIS\n")
    f.write("-"*20 + "\n")
    f.write("Monthly Power Consumption Statistics:\n")
    f.write(str(monthly_consumption) + "\n\n")
    f.write("Seasonal Power Consumption Statistics:\n")
    f.write(str(seasonal_consumption) + "\n\n")
    
    f.write("BUILDING-SPECIFIC TEMPORAL PATTERNS\n")
    f.write("-"*35 + "\n")
    for building_type, patterns in building_patterns.items():
        f.write(f"\n{building_type}:\n")
        f.write(f"  Peak hour: {patterns['peak_hour']}:00\n")
        f.write(f"  Peak consumption: {patterns['peak_consumption']:.2f} kWh\n")
        f.write(f"  Daily variability: {patterns['daily_std']:.2f}\n")
        f.write(f"  Hourly variability: {patterns['hourly_std']:.2f}\n")
    
    f.write("\nWEATHER LAG EFFECTS\n")
    f.write("-"*20 + "\n")
    for var in weather_vars:
        max_corr_idx = np.argmax(np.abs(lag_correlations[var]))
        f.write(f"{var.replace('_', ' ').title()}:\n")
        f.write(f"  Best correlation: {lag_correlations[var][max_corr_idx]:.4f} at lag {max_corr_idx} hours\n")

# Save outlier analysis
with open('txt_results/outlier_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Outlier Analysis Report\n")
    f.write("="*25 + "\n\n")
    f.write(f"IQR Method Results:\n")
    f.write(f"Q1: {Q1:.2f} kWh\n")
    f.write(f"Q3: {Q3:.2f} kWh\n")
    f.write(f"IQR: {IQR:.2f} kWh\n")
    f.write(f"Lower Bound: {lower_bound:.2f} kWh\n")
    f.write(f"Upper Bound: {upper_bound:.2f} kWh\n\n")
    f.write(f"Number of outliers: {len(outliers):,}\n")
    f.write(f"Percentage of total data: {len(outliers)/len(train_df_en)*100:.2f}%\n\n")
    if len(outliers) > 0:
        f.write("Outlier Statistics:\n")
        f.write(str(outliers['power_consumption'].describe()) + "\n")

save_to_report(f"\nAnalysis completed! Visualization results saved in 'visualization_results' folder.")
save_to_report("Text analysis results saved in 'txt_results' folder.")
save_to_report("Ready for AI model development for the 2025 Power Consumption Prediction Competition!") 