import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import logging
import holidays
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_and_prepare_weather(input_path, start='2020-01-01', freq='15min'):
    """
    Load NetCDFs of temperature for baseline (ERA5: t2m) and RCP scenarios (tas),
    interpolate to 15-min, and return a dict of DataFrames.

    For RCP scenarios that start after 'start', this function backfills earlier periods with
    the baseline values to ensure a complete 2020–end horizon.

    input_path: root folder containing 'Climate Data' subfolder with the .nc files.
    """
    # define file paths
    nc_paths = {
        'baseline': os.path.join(input_path, 'Climate Data', 'era5_portugal_baseline.nc'),
        'rcp26':    os.path.join(input_path, 'Climate Data', 'copernicus_rcp_2_6_portugal_future.nc'),
        'rcp45':    os.path.join(input_path, 'Climate Data', 'copernicus_rcp_4_5_portugal_future.nc'),
        'rcp85':    os.path.join(input_path, 'Climate Data', 'copernicus_rcp_8_5_portugal_future.nc')
    }
    # gather all time arrays to define full horizon
    all_times = []
    for f in nc_paths.values():
        ds = xr.open_dataset(f, chunks={'time': 365})
        all_times.append(ds['time'].values)
        ds.close()
    combined = np.concatenate(all_times)
    full_start = pd.to_datetime(combined.min())
    full_end   = pd.to_datetime(combined.max())
    new_index  = pd.date_range(start=max(full_start, pd.to_datetime(start)),
                                end=full_end, freq=freq)

    # first load baseline to use for backfilling
    weather = {}
    name = 'baseline'
    fpath = nc_paths[name]
    logger.info(f"Loading climate scenario '{name}' from {fpath}")
    ds = xr.open_dataset(fpath, chunks={'time': 365})
    ds = ds.mean(dim=[d for d in ds.dims if d != 'time'])
    temp_data = ds['t2m'] - 273.15
    dfw = temp_data.to_dataframe().reset_index()[['time', 't2m']]
    ds.close()
    dfw = dfw[dfw['time'] >= start].set_index('time')
    vals = interpolate_variable_series(
        dfw.index.values, dfw['t2m'].values,
        new_index.values, method='cosine'
    )
    w_baseline = pd.DataFrame({'avg_t2m': vals}, index=new_index)
    w_baseline.index = pd.to_datetime(w_baseline.index).tz_localize('UTC')
    w_baseline.index = w_baseline.index.tz_convert('Europe/Lisbon').tz_localize(None)
    w_baseline.index.name = 'LocalTime'
    # drop any duplicate local times (DST repeats)
    w_baseline = w_baseline[~w_baseline.index.duplicated(keep='first')]
    weather['baseline'] = w_baseline

    # load RCP scenarios, backfill with baseline for early period
    for name in ['rcp26', 'rcp45', 'rcp85']:
        fpath = nc_paths[name]
        logger.info(f"Loading climate scenario '{name}' from {fpath}")
        ds = xr.open_dataset(fpath, chunks={'time': 365})
        ds = ds.mean(dim=[d for d in ds.dims if d != 'time'])
        temp_var = 't2m' if 't2m' in ds else 'tas'
        temp_data = ds[temp_var] - 273.15
        dfw = temp_data.to_dataframe().reset_index()[['time', temp_var]].rename(columns={temp_var: 't2m'})
        ds.close()
        dfw = dfw[dfw['time'] >= start].set_index('time')
        # scenario start
        scen_start = dfw.index.min()
        vals = interpolate_variable_series(
            dfw.index.values, dfw['t2m'].values,
            new_index.values, method='cosine'
        )
        wdf = pd.DataFrame({'avg_t2m': vals}, index=new_index)
        # backfill before scen_start using baseline via aligned array
        pre_mask = wdf.index < scen_start
        # get baseline values reindexed
        baseline_vals = w_baseline['avg_t2m'].reindex(wdf.index).values
        # replace early period
        wdf['avg_t2m'] = np.where(pre_mask, baseline_vals, wdf['avg_t2m'].values)
        # convert timezone
        wdf.index = pd.to_datetime(wdf.index).tz_localize('UTC')
        wdf.index = wdf.index.tz_convert('Europe/Lisbon').tz_localize(None)
        wdf.index.name = 'LocalTime'
        # drop any duplicate local times (DST repeats)
        wdf = wdf[~wdf.index.duplicated(keep='first')]
        weather[name] = wdf

    return weather

def interpolate_variable_series(orig_times, orig_values, new_times, method='linear'):
    """
    Interpolates a 1D array of values given the original timestamps and the new target timestamps.
    Parameters:
      orig_times: array-like of original timestamps (assumed to be sorted, dtype=datetime64[ns])
      orig_values: array-like of original values (numeric)
      new_times: array-like of new timestamps (dtype=datetime64[ns])
      method: either 'cosine' (for a smooth, sinusoidal transition) or 'linear'
    Returns:
      A numpy array of interpolated values for new_times.
    """
    # Sort the original times (if not already) and convert to numeric seconds.
    orig_times = np.array(orig_times)
    sort_idx = np.argsort(orig_times)
    orig_times = orig_times[sort_idx]
    orig_values = np.array(orig_values)[sort_idx]

    # ✂ drop any missing values:
    valid = ~np.isnan(orig_values)
    orig_times = orig_times[valid]
    orig_values = orig_values[valid]

    orig_t_numeric = orig_times.astype('int64') // 10**9
    new_t_numeric = new_times.astype('int64') // 10**9

    # Initialize result array
    res = np.empty(new_t_numeric.shape, dtype=float)

    # Fill values for new times before the first measurement and after the last.
    res[new_t_numeric <= orig_t_numeric[0]] = orig_values[0]
    res[new_t_numeric >= orig_t_numeric[-1]] = orig_values[-1]

    # For times in between, do interpolation.
    mask = (new_t_numeric > orig_t_numeric[0]) & (new_t_numeric < orig_t_numeric[-1])
    if np.any(mask):
        new_valid = new_t_numeric[mask]
        # For each new_valid time, find the index in orig_t_numeric
        idx_upper = np.searchsorted(orig_t_numeric, new_valid, side='right')
        idx_lower = idx_upper - 1

        lower_t = orig_t_numeric[idx_lower]
        upper_t = orig_t_numeric[idx_upper]
        lower_values = orig_values[idx_lower]
        upper_values = orig_values[idx_upper]

        # Compute the time difference (in seconds) between lower and upper timestamps.
        delta = (upper_t - lower_t).astype(float)
        # Compute the fractional difference for each new time, take care to avoid division by zero.
        frac = (new_valid - lower_t) / np.where(delta == 0, 1, delta)

        if method == 'cosine':
            # Use cosine-based weighting for smooth diurnal variables.
            weight = (1 - np.cos(np.pi * frac)) / 2
            interp_val = lower_values + (upper_values - lower_values) * weight
        else:  # linear
            interp_val = lower_values + (upper_values - lower_values) * frac

        res[mask] = interp_val
    return res


def load_and_preprocess_data(filepath, target_column,
                             upper_clip = 0.995, lower_clip = 0.005):
    """
    Load data from Excel, parse dates, convert the target column to numeric,
    and return both actual and cleaned versions.
    """
    df = pd.read_excel(filepath)

    # Ensure 'Date/Time' is datetime
    if not np.issubdtype(df['LocalTime'].dtype, np.datetime64):
        df['LocalTime'] = pd.to_datetime(df['Date/Time'])
    if target_column != 'Consumption (kWh)':
        df = df[['LocalTime', target_column, 'Consumption (kWh)']]
        df['Consumption (kWh)'] = pd.to_numeric(df['Consumption (kWh)'], errors='coerce')
    else:
        df = df[['LocalTime', target_column]]

    df = df[df['LocalTime'] < pd.Timestamp("2025-04-01")]
    df.set_index('LocalTime', inplace=True)

    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Drop duplicate timestamps (keep first)
    df_actual = df[~df.index.duplicated(keep='first')].copy()

    # Cleaned version with outlier clipping
    df_cleaned = df_actual.copy()
    lower = df_cleaned[target_column].quantile(lower_clip)
    upper = df_cleaned[target_column].quantile(upper_clip)
    df_cleaned[target_column] = df_cleaned[target_column].clip(lower, upper)

    return df_actual, df_cleaned

def get_season(month):
    """Return season name given a month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def renormalize_profile(month_df):
    """Renormalize the 'fraction' column so that it sums to 1 for a given group."""
    total = month_df['fraction'].sum()
    if total > 0:
        month_df['fraction'] = month_df['fraction'] / total
    return month_df


def log_fig(fig, filename):
    """Log a given figure via MLflow and then close it."""
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def disaggregate_monthly_to_interval_mc(monthly_forecast, daily_profile, noise_daily, daily_factor_clip, interval,
                                        summer_noise_daily=None,
                                        summer_months=[6, 7, 8]):
    """
    Disaggregate a monthly forecast series into a finer interval (e.g. 15min) by:
      - Splitting the monthly total equally among all days,
      - Using a typical daily profile (filtered by month, weekday, and holiday) to distribute each day's total,
      - Applying a random noise factor (clamped by daily_factor_clip).

    Parameters:
      monthly_forecast (pd.Series): Monthly consumption totals.
      daily_profile (pd.DataFrame): Typical fractions for each day segment, must include columns:
            'month', 'weekday', 'holiday', 'time', 'fraction'
      noise_daily (float): Base range for random noise (e.g., 0.02 for ±2% noise).
      daily_factor_clip (float): Maximum offset allowed in the daily noise.
      interval (str): Target interval frequency (e.g., '15min').
      summer_noise_daily (float, optional): Noise range to apply for summer months. If not provided,
                                            the base noise_daily is used.
      summer_months (list, optional): List of month numbers considered to be summer (default: [6,7,8]).

    Returns:
      pd.Series: Forecast disaggregated at the specified interval.
    """
    forecast_interval_list = []
    # Determine the number of periods in one day based on the interval
    n_periods = int(pd.Timedelta('1 day') / pd.Timedelta(interval))

    # Loop over each monthly period in the forecast series
    for month_end, monthly_value in monthly_forecast.items():
        # Get the first day of the month
        month_start = month_end.replace(day=1)
        all_days = pd.date_range(start=month_start, end=month_end, freq='D')
        num_days = len(all_days)
        # Distribute the monthly total equally among days
        daily_total = monthly_value / num_days

        for day in all_days:
            day_date = pd.to_datetime(day)
            day_month = day_date.month
            day_weekday = day_date.day_name()
            # Determine holiday flag using the holidays package for Portugal
            pt_future = holidays.country_holidays('PT', years=[day_date.year])
            day_holiday = 1 if day_date.date() in pt_future else 0
            # Create the time index for the day using the specified interval
            times = pd.date_range(start=day_date.normalize(), periods=n_periods, freq=interval)

            # Choose the appropriate noise factor:
            # Use summer_noise_daily if provided and the day is in one of the summer months.
            effective_noise = summer_noise_daily if (
                        summer_noise_daily is not None and day_month in summer_months) else noise_daily

            # Fetch the typical profile for this combination
            profile_filter = (
                    (daily_profile['month'] == day_month) &
                    (daily_profile['weekday'] == day_weekday) &
                    (daily_profile['holiday'] == day_holiday)
            )
            profile_day = daily_profile[profile_filter].copy()
            # If no profile is found, fallback to filtering by month and weekday only
            if profile_day.empty:
                profile_filter = (
                        (daily_profile['month'] == day_month) &
                        (daily_profile['weekday'] == day_weekday)
                )
                profile_day = daily_profile[profile_filter].copy()
            # Reindex the profile to match the times in the day (based on the time component)
            profile_day = profile_day.set_index('time').reindex([t.time() for t in times])
            # Interpolate missing fractions linearly
            profile_day['fraction'] = profile_day['fraction'].interpolate(method='linear')
            # Normalize the fractions so they sum to 1
            profile_day['fraction'] = profile_day['fraction'] / profile_day['fraction'].sum()
            # Apply a random daily factor with noise, ensuring it stays within the clipping range
            raw_factor = np.random.uniform(-effective_noise, effective_noise)
            raw_factor = np.clip(raw_factor, -daily_factor_clip, daily_factor_clip)
            random_factor = 1 + raw_factor
            day_values = daily_total * random_factor * profile_day['fraction'].values
            s = pd.Series(day_values, index=times)
            forecast_interval_list.append(s)

    forecast_interval = pd.concat(forecast_interval_list)
    return forecast_interval


def get_extended_pt_holidays(start_year, end_year):
    """
    Get a set of extended Portuguese holiday dates between start_year and end_year.
    If end_year exceeds 2033, replicate the pattern from 2033 (note: this is naive for holidays
    that depend on the date of Easter or other moving dates).
    """
    # Gather official Portuguese holidays up to year 2033

    start_year = int(start_year)
    end_year = int(end_year)

    known_holidays = holidays.country_holidays('PT', years=range(start_year, min(end_year, 2033) + 1))
    all_holiday_dates = set(known_holidays.keys())

    # If end_year is greater than 2033, replicate the pattern from 2033
    if end_year > 2033:
        for year in range(2034, end_year + 1):
            for dt in known_holidays.keys():
                if dt.year == 2033:
                    new_dt = dt.replace(year=year)
                    all_holiday_dates.add(new_dt)

    return all_holiday_dates


def build_daily_profile(df, target_column, interval):
    """
    Given a DataFrame, resample the target column to the chosen interval and compute a typical daily profile.
    The function returns a DataFrame with columns for month, weekday, holiday (for Portugal),
    time and the normalized fraction of consumption (or production) that falls in that interval.
    """
    df_interval = df[target_column].resample(interval).ffill().to_frame('consumption')
    df_interval['date'] = df_interval.index.normalize()
    df_interval['time'] = df_interval.index.time
    df_interval['weekday'] = df_interval.index.day_name()
    df_interval['month'] = df_interval.index.month
    df_interval['season'] = df_interval['month'].map(get_season)
    # Get holidays for the observed range
    hist_years = df_interval.index.year.unique()
    extended_holidays = get_extended_pt_holidays(hist_years.min(), df_interval.index.year.max())
    df_interval['holiday'] = df_interval.index.map(lambda x: 1 if x.date() in extended_holidays else 0)
    daily_totals = df_interval.groupby('date')['consumption'].transform('sum')
    df_interval['fraction'] = df_interval['consumption'] / daily_totals
    daily_profile = (df_interval
                     .groupby(['month', 'weekday', 'holiday', 'time'])['fraction']
                     .mean()
                     .reset_index())
    daily_profile = daily_profile.groupby(['month', 'weekday', 'holiday'], group_keys=False).apply(renormalize_profile)
    return daily_profile

def create_time_features(df, holiday_years, weather_df=None):
    feats = pd.DataFrame(index=df.index)
    feats['month']     = df.index.month
    feats['dayofweek'] = df.index.dayofweek
    feats['hour']      = df.index.hour
    feats['minute']    = df.index.minute
    hols = get_extended_pt_holidays(holiday_years[0], holiday_years[-1])
    feats['holiday']   = df.index.map(lambda t: 1 if t.date() in hols else 0)
    feats['season']    = df.index.month.map(get_season)
    if weather_df is not None:
        w = weather_df.reindex(df.index).ffill()
        feats['avg_t2m'] = w['avg_t2m']
        # fill any remaining missing with historical mean
        mean_temp = feats['avg_t2m'].mean()
        feats['avg_t2m'] = feats['avg_t2m'].fillna(mean_temp)
    feats = pd.get_dummies(
        feats,
        columns=['month','dayofweek','hour','season'],
        drop_first=True
    )
    # ensure no NaNs remain
    feats.fillna(0, inplace=True)
    return feats


def train_point_and_quantile_models(
    df_train,
    years,
    model_dir,
    target_col,
    quantiles=None,
    weather_df=None,
    resid_split_ratio=0.2
):
    """
    Train point and quantile GradientBoostingRegressor models for a given target column.

    Args:
        df_train: DataFrame of training data
        years: list of years present in the full dataset
        model_dir: directory to save models and feature columns
        target_col: name of the target column in df_train
        quantiles: list of quantiles (floats) for quantile models
        weather_df: optional weather DataFrame aligned to df_train.index
        resid_split_ratio: fraction of df_train reserved for residual quantile modeling
    """
    logger.info(f'Training Model for: {target_col}')
    os.makedirs(model_dir, exist_ok=True)
    if quantiles is None:
        quantiles = [q/100 for q in range(5, 100, 10)]

    # split for point vs residual modeling
    df_pt, df_res = train_test_split(df_train, test_size=resid_split_ratio, shuffle=False)

    # point model
    X_pt = create_time_features(df_pt, years, weather_df)
    y_pt = df_pt[target_col]
    pm = GradientBoostingRegressor(
        loss='squared_error', n_estimators=200, max_depth=4, random_state=42
    )
    pm.fit(X_pt, y_pt)
    joblib.dump(pm, os.path.join(model_dir, 'point_model.pkl'))

    # save feature columns
    feat_cols = X_pt.columns.tolist()
    joblib.dump(feat_cols, os.path.join(model_dir, 'feature_columns.pkl'))
    logger.info('Point model and features saved in %s', model_dir)

    # quantile models
    X_res = create_time_features(df_res, years, weather_df)
    X_res = X_res.reindex(columns=feat_cols, fill_value=0)
    y_pred_res = pm.predict(X_res)
    resid = df_res[target_col] - y_pred_res

    Xq = X_res.copy()
    Xq['point_forecast'] = y_pred_res
    for q in quantiles:
        qr = GradientBoostingRegressor(
            loss='quantile', alpha=q, n_estimators=200, max_depth=4, random_state=42
        )
        qr.fit(Xq, resid)
        joblib.dump(qr, os.path.join(model_dir, f'quantile_{int(q*100):03d}.pkl'))
        logger.info('Quantile %d%% model saved.', int(q*100))


def add_exog_lags_and_roll(X, exog_series, lags=[1,4,96], windows=[4,16,96,672], prefix=""):
    """
    Add lagged and rolling‐mean features for an exogenous series.
    exog_series: pd.Series indexed like X
    """
    for lag in lags:
        X[f"{prefix}lag_{lag}"] = exog_series.shift(lag)
    for w in windows:
        X[f"{prefix}roll_mean_{w}"] = exog_series.rolling(w).mean()
    return X.fillna(0)


def convert_scen_to_filename(scen_key):
    mapping = {
        'rcp26': 'rcp_2_6',
        'rcp45': 'rcp_4_5',
        'rcp85': 'rcp_8_5',
    }
    return mapping.get(scen_key, scen_key)




