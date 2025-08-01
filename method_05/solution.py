"""
Electricity Consumption Prediction for Buildings
-------------------------------------------------

This script prepares data, performs exploratory data analysis (EDA) and
pre‑processing, and trains a machine‑learning model to forecast hourly
electricity consumption for a set of buildings.  The target metric for
evaluation is the Symmetric Mean Absolute Percentage Error (SMAPE),
defined in the literature as the mean of the absolute differences
between the forecast and the actual values divided by half the sum of
their absolute values【179233723382245†L118-L130】.  This metric is
bounded between 0 and 200 % and is less sensitive to large errors than
the classical mean absolute percentage error【179233723382245†L155-L163】.

The workflow implemented in this module consists of the following steps:

1. **Data loading** – read the training, test and building meta‑data from
   the provided CSV files.
2. **Feature engineering** – merge building information, convert
   timestamp columns to `datetime` objects, and derive additional
   temporal features (month, day, hour, weekday, weekend flag).
   Building‑specific statistics (mean consumption per hour, weekday and
   month) are computed from the training set and merged back into both
   training and test sets.  Weather‑related fields that are missing in
   the test set (sunshine hours and solar radiation) are approximated
   using the average values observed in August for the corresponding
   hour of the day.  Additional engineered features such as the ratio
   of cooling area to total area, photovoltaic capacity per area and
   interaction terms (humidity × temperature, rainfall × wind speed)
   are also created.
3. **Model training** – build a pipeline consisting of a `OneHotEncoder`
   for categorical variables (building ID and building type) followed
   by an `XGBRegressor`.  The data are split chronologically, with the
   last week of the training period reserved for validation.  A
   moderate depth, a relatively large number of estimators and a low
   learning rate are used to balance bias and variance.
4. **Evaluation** – compute the SMAPE on the validation split to gauge
   expected performance.  The full model is subsequently fitted on the
   entire training set and used to generate predictions for the test
   period.  These predictions are written to `submission.csv` in the
   required format.

Although extensive efforts were made to reduce the SMAPE below the
requested threshold, the final validation score may still exceed 6 %.
This is likely due to the limited length of the training period and
missing weather attributes in the test set.  Nevertheless, the model
provides a reasonable baseline and can be further improved with
additional domain‑specific features or a longer observation period.

Usage
-----
Run this script from the command line.  It produces two files in
`./` relative to the script location:

* `submission.csv` – predictions for the test set in the competition
  format.
* `model_validation.txt` – text file containing the validation SMAPE.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is defined as:

        100 / n * sum(|F_t - A_t| / ((|A_t| + |F_t|) / 2))

    where ``F_t`` are the forecasts and ``A_t`` are the actual values【179233723382245†L118-L130】.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    float
        The SMAPE expressed as a percentage.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # avoid division by zero – if both true and pred are zero, contribution is zero
    mask = denominator != 0
    smape_values = np.zeros_like(numerator, dtype=float)
    smape_values[mask] = numerator[mask] / denominator[mask]
    return 100.0 * np.mean(smape_values)


def load_data(train_path: Path, test_path: Path, building_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train, test and building information from CSV files.

    Parameters
    ----------
    train_path : Path
        Path to the training data file.
    test_path : Path
        Path to the test data file.
    building_path : Path
        Path to the building metadata file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrames containing the training and test data merged with
        building information.
    """
    # Column renaming map to avoid special characters
    rename_map = {
        '기온(°C)': 'temp',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'wind_speed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine_hours',
        '일사(MJ/m2)': 'solar_radiation',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'pv_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity',
        '건물유형': 'building_type',
    }
    train = pd.read_csv(train_path).rename(columns=rename_map)
    test = pd.read_csv(test_path).rename(columns=rename_map)
    building_info = pd.read_csv(building_path).rename(columns=rename_map)
    # Cast numeric building fields, replacing '-' with NaN
    for col in ['total_area', 'cooling_area', 'pv_capacity', 'ess_capacity', 'pcs_capacity']:
        building_info[col] = building_info[col].replace('-', np.nan).astype(float)
    # Merge building info
    train = train.merge(building_info, on='건물번호', how='left')
    test = test.merge(building_info, on='건물번호', how='left')
    return train, test


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform feature engineering on the training and test data.

    This function adds temporal features, imputes missing weather
    information for the test set, computes building‑specific statistics
    from the training data, and derives several interaction terms.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with building information.
    test : pd.DataFrame
        Test data with building information.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The transformed training and test DataFrames.
    """
    # Parse datetime and derive basic calendar features
    train['datetime'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
    test['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')
    for df in (train, test):
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday  # Monday=0
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # Enhanced time features based on EDA insights
        df['hour_peak_flag'] = df['hour'].isin([10, 11, 14, 15]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Impute missing numeric building info with training medians
    for col in ['total_area', 'cooling_area', 'pv_capacity', 'ess_capacity', 'pcs_capacity']:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
    # Approximate sunshine and solar radiation for the test set
    # Use average values per hour observed in August in the training data
    train_august = train[train['month'] == 8]
    avg_sunshine = train_august.groupby('hour')['sunshine_hours'].mean()
    avg_solar = train_august.groupby('hour')['solar_radiation'].mean()
    train['sunshine_est'] = train['sunshine_hours']
    train['solar_est'] = train['solar_radiation']
    test['sunshine_est'] = test['hour'].map(avg_sunshine)
    test['solar_est'] = test['hour'].map(avg_solar)
    # Drop the original sunshine and solar columns because the test set lacks them
    train.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True)
    test.drop(columns=['sunshine_hours', 'solar_radiation'], inplace=True, errors='ignore')
    # Compute building‑level mean consumption (overall)
    building_mean = train.groupby('건물번호')['전력소비량(kWh)'].mean()
    # Mean consumption per building and hour
    bld_hour_mean = (
        train.groupby(['건물번호', 'hour'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_hour_mean'})
    )
    train = train.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test = test.merge(bld_hour_mean, on=['건물번호', 'hour'], how='left')
    test['bld_hour_mean'] = test['bld_hour_mean'].fillna(test['건물번호'].map(building_mean))
    # Mean consumption per building and weekday
    bld_weekday_mean = (
        train.groupby(['건물번호', 'weekday'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_wd_mean'})
    )
    train = train.merge(bld_weekday_mean, on=['건물번호', 'weekday'], how='left')
    test = test.merge(bld_weekday_mean, on=['건물번호', 'weekday'], how='left')
    test['bld_wd_mean'] = test['bld_wd_mean'].fillna(test['건물번호'].map(building_mean))
    # Mean consumption per building and month
    bld_month_mean = (
        train.groupby(['건물번호', 'month'])['전력소비량(kWh)']
        .mean()
        .reset_index()
        .rename(columns={'전력소비량(kWh)': 'bld_month_mean'})
    )
    train = train.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test = test.merge(bld_month_mean, on=['건물번호', 'month'], how='left')
    test['bld_month_mean'] = test['bld_month_mean'].fillna(test['건물번호'].map(building_mean))
    
    # Building-specific peak hour analysis  
    # Building-specific peak hour analysis - fixed for pandas compatibility
    building_peak_stats = []
    for building_id in train['건물번호'].unique():
        building_data = train[train['건물번호'] == building_id]
        peak_hour = building_data.groupby('hour')['전력소비량(kWh)'].mean().idxmax()
        building_peak_stats.append({'건물번호': building_id, 'peak_hour': peak_hour})
    building_peak_hours = pd.DataFrame(building_peak_stats)
    
    train = train.merge(building_peak_hours, on='건물번호', how='left')
    test = test.merge(building_peak_hours, on='건물번호', how='left')
    test['peak_hour'] = test['peak_hour'].fillna(test['hour'])  # fallback
    
    for df in (train, test):
        df['hour_deviation_from_peak'] = abs(df['hour'] - df['peak_hour'])
        df['is_building_peak_hour'] = (df['hour'] == df['peak_hour']).astype(int)
    
    # Enhanced building and weather features
    for df in (train, test):
        # Temperature features (U-shaped relationship with power consumption)
        df['temp_squared'] = df['temp'] ** 2
        df['temp_cooling_need'] = np.maximum(0, df['temp'] - 23)  # cooling threshold
        df['temp_heating_need'] = np.maximum(0, 20 - df['temp'])  # heating threshold
        
        # Building efficiency features
        df['area_ratio'] = df['cooling_area'] / df['total_area']
        df['pv_per_area'] = df['pv_capacity'] / df['total_area']
        
        # Enhanced weather interactions
        df['humidity_temp'] = df['humidity'] * df['temp']
        df['rain_wind'] = df['rainfall'] * df['wind_speed']
        df['temp_humidity_cooling'] = df['temp'] * df['humidity'] * df['cooling_area'] / 10000
        
        # Building type specific features
        df['is_idc'] = (df['building_type'] == 'IDC(전화국)').astype(int)
        df['is_department_store'] = (df['building_type'] == '백화점').astype(int)
        df['is_hospital'] = (df['building_type'] == '병원').astype(int)
        
        # Time-building type interactions
        df['idc_night_factor'] = df['is_idc'] * (1 - df['hour_peak_flag'])
        df['store_business_hours'] = df['is_department_store'] * df['hour_peak_flag']
        
        # High-importance feature interactions
        df['bld_hour_month_interaction'] = df['bld_hour_mean'] * df['month'] / 12
        df['cooling_temp_interaction'] = df['cooling_area'] * df['temp'] / 1000
    
    return train, test


def analyze_feature_importance(pipeline, feature_cols: list, categorical_cols: list, output_dir: Path) -> pd.DataFrame:
    """Analyze and visualize feature importance from the trained XGBoost model.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline containing preprocessor and XGBoost model
    feature_cols : list
        List of all feature column names
    categorical_cols : list
        List of categorical column names
    output_dir : Path
        Directory to save importance plots and analysis
        
    Returns
    -------
    pd.DataFrame
        Feature importance analysis results
    """
    # Get the XGBoost model from pipeline
    model = pipeline.named_steps['model']
    
    # Get feature importance
    importance_gain = model.feature_importances_
    
    # Get feature names after one-hot encoding
    preprocessor = pipeline.named_steps['preprocess']
    
    # Get encoded feature names
    encoded_feature_names = []
    
    # Categorical features (one-hot encoded)
    cat_transformer = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_transformer.get_feature_names_out(categorical_cols)
    encoded_feature_names.extend(cat_feature_names)
    
    # Numerical features (passthrough)
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    encoded_feature_names.extend(numerical_cols)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': encoded_feature_names,
        'importance': importance_gain
    }).sort_values('importance', ascending=False)
    
    # Save importance analysis
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze categorical vs numerical importance
    cat_importance = importance_df[importance_df['feature'].str.contains('|'.join(categorical_cols))]
    num_importance = importance_df[~importance_df['feature'].str.contains('|'.join(categorical_cols))]
    
    analysis = {
        'total_features': len(importance_df),
        'top_10_features': importance_df.head(10)['feature'].tolist(),
        'categorical_avg_importance': cat_importance['importance'].mean(),
        'numerical_avg_importance': num_importance['importance'].mean(),
        'top_categorical': cat_importance.head(5)['feature'].tolist(),
        'top_numerical': num_importance.head(5)['feature'].tolist()
    }
    
    # Save analysis summary
    with open(output_dir / 'feature_analysis_summary.txt', 'w') as f:
        f.write("=== Feature Importance Analysis ===\n\n")
        f.write(f"Total features: {analysis['total_features']}\n")
        f.write(f"Average categorical importance: {analysis['categorical_avg_importance']:.6f}\n")
        f.write(f"Average numerical importance: {analysis['numerical_avg_importance']:.6f}\n\n")
        f.write("Top 10 Most Important Features:\n")
        for i, feat in enumerate(analysis['top_10_features'], 1):
            importance_val = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
            f.write(f"{i:2d}. {feat}: {importance_val:.6f}\n")
        f.write(f"\nTop 5 Categorical Features:\n")
        for feat in analysis['top_categorical']:
            importance_val = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
            f.write(f"  - {feat}: {importance_val:.6f}\n")
        f.write(f"\nTop 5 Numerical Features:\n")
        for feat in analysis['top_numerical']:
            importance_val = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
            f.write(f"  - {feat}: {importance_val:.6f}\n")
    
    return importance_df


def build_and_train_model(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    """Train an XGBoost model and produce predictions for the test set.

    Parameters
    ----------
    train : pd.DataFrame
        The training DataFrame containing engineered features and the target.
    test : pd.DataFrame
        The test DataFrame containing engineered features.
    output_dir : Path
        Directory where the submission file and validation summary will be saved.
    """
    # Identify the feature columns available in both train and test
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]
    X = train[feature_cols].copy()
    y = train['전력소비량(kWh)']
    X_test = test[feature_cols].copy()
    # Convert categorical features to string type for one‑hot encoding
    X['건물번호'] = X['건물번호'].astype(str)
    X['building_type'] = X['building_type'].astype(str)
    X_test['건물번호'] = X_test['건물번호'].astype(str)
    X_test['building_type'] = X_test['building_type'].astype(str)
    categorical_cols = ['건물번호', 'building_type']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    # Chronological split: last 7 days of training period as validation
    cutoff = pd.Timestamp('2024-08-18')
    train_mask = train['datetime'] < cutoff
    val_mask = ~train_mask
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    # Balanced XGBoost regressor for good performance and speed
    model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=400,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=2.0,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42
    )
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    # Train on the chronological training subset
    pipeline.fit(X_train, y_train)
    # Evaluate on the validation subset
    val_pred = pipeline.predict(X_val)
    val_smape = smape(y_val.values, val_pred)
    
    # Analyze feature importance on validation model
    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(pipeline, feature_cols, categorical_cols, output_dir)
    
    # Save validation SMAPE to a text file
    validation_path = output_dir / 'model_validation.txt'
    with validation_path.open('w') as f:
        f.write(f'Validation SMAPE: {val_smape:.6f}%\n')
        f.write(f'Top 5 features: {importance_df.head(5)["feature"].tolist()}\n')
    
    print(f"Validation SMAPE: {val_smape:.6f}%")
    print(f"Top 5 important features: {importance_df.head(5)['feature'].tolist()}")
    
    # Retrain on the full training data
    pipeline.fit(X, y)
    
    # Analyze feature importance on final model
    final_importance_df = analyze_feature_importance(pipeline, feature_cols, categorical_cols, output_dir)
    final_importance_df.to_csv(output_dir / 'final_feature_importance.csv', index=False)
    # Generate predictions for the test set
    test_pred = pipeline.predict(X_test)
    # Create submission file
    submission = test[['num_date_time']].copy()
    submission['answer'] = test_pred
    submission.to_csv(output_dir / 'submission.csv', index=False)


def main() -> None:
    base_dir = Path('.')
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    # Load raw data and merge building info
    train_df, test_df = load_data(train_path, test_path, building_path)
    # Engineer features
    train_fe, test_fe = engineer_features(train_df, test_df)
    # Train model and write output files
    build_and_train_model(train_fe, test_fe, base_dir)


if __name__ == '__main__':
    main()