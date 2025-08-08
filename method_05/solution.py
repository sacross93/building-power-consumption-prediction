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
    # Additional engineered features
    for df in (train, test):
        df['area_ratio'] = df['cooling_area'] / df['total_area']
        df['pv_per_area'] = df['pv_capacity'] / df['total_area']
        df['humidity_temp'] = df['humidity'] * df['temp']
        df['rain_wind'] = df['rainfall'] * df['wind_speed']
    return train, test


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
    # Instantiate the XGBoost regressor
    model = xgb.XGBRegressor(
        max_depth=12,
        n_estimators=1000,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
    )
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    # Train on the chronological training subset
    pipeline.fit(X_train, y_train)
    # Evaluate on the validation subset
    val_pred = pipeline.predict(X_val)
    val_smape = smape(y_val.values, val_pred)
    # Save validation SMAPE to a text file
    validation_path = output_dir / 'model_validation.txt'
    with validation_path.open('w') as f:
        f.write(f'Validation SMAPE: {val_smape:.6f}%\n')
    # Retrain on the full training data
    pipeline.fit(X, y)
    # Generate predictions for the test set
    test_pred = pipeline.predict(X_test)
    # Create submission file
    submission = test[['num_date_time']].copy()
    submission['answer'] = test_pred
    submission.to_csv(output_dir / 'submission.csv', index=False)


def main() -> None:
    base_dir = Path('.')
    train_path = base_dir / 'train.csv'
    test_path = base_dir / 'test.csv'
    building_path = base_dir / 'building_info.csv'
    # Load raw data and merge building info
    train_df, test_df = load_data(train_path, test_path, building_path)
    # Engineer features
    train_fe, test_fe = engineer_features(train_df, test_df)
    # Train model and write output files
    build_and_train_model(train_fe, test_fe, base_dir)


if __name__ == '__main__':
    main()