# Advanced Feature Engineering Report

## Feature Categories Created

### 1. Building Clustering Features
- K-means clustering (5 clusters) based on power patterns
- Cluster-specific time factors
- Building efficiency scores

### 2. Advanced Time Features
- 15-minute and 30-minute patterns
- Business calendar features
- Season-time interactions
- Building type specific time patterns

### 3. Weather Enhancement
- Apparent temperature with wind chill
- Enhanced discomfort index
- Temperature volatility and change rates
- Humidity comfort zones

### 4. Building-Specific Features
- Green building scores
- Building type time patterns
- PV solar potential calculations

### 5. Interaction Features
- Time-weather interactions
- Building-time-weather 3-way interactions
- Cluster-specific patterns

## Top Correlations with Target

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | bld_hour_month_interaction | 0.9832 |
| 2 | bld_hour_mean | 0.9824 |
| 3 | bld_month_mean | 0.9553 |
| 4 | bld_wd_mean | 0.9516 |
| 5 | is_cluster_2 | 0.6745 |
| 6 | building_cluster | 0.6567 |
| 7 | is_cluster_0 | 0.6206 |
| 8 | idc(전화국)_efficiency | 0.5957 |
| 9 | is_idc | 0.5957 |
| 10 | cluster_2_hour_factor | 0.5716 |
| 11 | cluster_2_temp_hour | 0.5638 |
| 12 | idc_night_factor | 0.5392 |
| 13 | cluster_4_temp_hour | 0.3724 |
| 14 | cluster_0_hour_factor | 0.3659 |
| 15 | is_cluster_4 | 0.3626 |
| 16 | cluster_4_hour_factor | 0.3586 |
| 17 | cluster_2_weekend_factor | 0.3489 |
| 18 | cluster_0_temp_hour | 0.3445 |
| 19 | idc(전화국)_peak_time | 0.2681 |
| 20 | cluster_0_weekend_factor | 0.2117 |

## Cluster Analysis

Buildings were clustered into 5 groups based on:
- Average power consumption
- Power consumption variability
- Building physical characteristics

## Expected Impact

- **Improved temporal modeling**: Finer time granularity
- **Better building segmentation**: Cluster-based features
- **Enhanced weather sensitivity**: Advanced weather indices
- **Richer interactions**: Multi-way feature combinations
