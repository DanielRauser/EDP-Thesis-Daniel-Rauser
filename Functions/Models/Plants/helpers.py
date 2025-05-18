import pandas as pd
import numpy as np
import pvlib
import holidays
import rasterio
import logging
from math import pi, sin

logger = logging.getLogger(__name__)


def get_weibull_params(lon, lat, scale_tif, shape_tif):
    with rasterio.open(scale_tif) as src_scale:
        row, col = src_scale.index(lon, lat)
        c_hr = src_scale.read(1)[row, col]
    with rasterio.open(shape_tif) as src_shape:
        row, col = src_shape.index(lon, lat)
        k_hr = src_shape.read(1)[row, col]
    return c_hr, k_hr

def add_solar_elevation(df):
    # Reset index if duplicates exist
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)

    if 'time' in df.columns:
        time_column = 'time'
    elif 'UTC' in df.columns:
        time_column = 'UTC'
    else:
        raise KeyError("The dataframe must contain 'time' or 'UTC' as a time column.")

    # Ensure unique time values
    if df[time_column].duplicated().any():
        df = df.drop_duplicates(subset=[time_column, 'latitude', 'longitude'])

    def compute_solar(df_group):
        """Compute solar position for each group."""
        solpos = pvlib.solarposition.get_solarposition(
            time=df_group[time_column],
            latitude=df_group['latitude'].iloc[0],
            longitude=df_group['longitude'].iloc[0]
        )
        df_group['solar_elevation'] = solpos['apparent_elevation'].values
        return df_group

    # Group by an identifier combining latitude and longitude
    if "identifier" not in df.columns:
        df["identifier"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    df = df.groupby("identifier", group_keys=False).apply(compute_solar)

    return df

def compute_clear_sky_index(data):
    # Extract the unique latitude and longitude for the group.
    lat = data['latitude'].iloc[0]
    lon = data['longitude'].iloc[0]
    # Create a Location object for this plant.
    location = pvlib.location.Location(latitude=lat, longitude=lon, tz='UTC')
    # Convert LocalTime to a DatetimeIndex (which pvlib expects)
    times = pd.DatetimeIndex(data['LocalTime'])
    cs = location.get_clearsky(times)

    # Extract GHI values
    ghi = cs['ghi'].values
    # Small threshold to avoid division by zero.
    epsilon = 1e-6
    # Get ssrd values (ensure alignment)
    ssrd_vals = data['ssrd'].values.astype(float)

    # Create an empty array for clear sky index.
    clear_sky_index = np.zeros_like(ghi, dtype=float)

    # For rows where GHI is above the threshold, compute the ratio.
    valid = ghi > epsilon
    clear_sky_index[valid] = ssrd_vals[valid] / ghi[valid]

    # Replace any inf or nan values with 0 (or another default if desired).
    clear_sky_index = np.nan_to_num(clear_sky_index, nan=-1.0, posinf=-2.0, neginf=-1.0)

    # Assign the cleaned clear sky index back to the group.
    data['clear_sky_index'] = clear_sky_index
    return data


def calculate_morning_ramp(solar_elevation, threshold=10, min_factor=0.2):
    """
    Computes a factor for the gradual build-up of power in the morning.

    For solar elevations between 0 and 'threshold' (in degrees),
    the factor ramps from 0 to 1 using a sine function.

    - solar_elevation : float
        The current solar elevation (in degrees).
    - threshold : float, optional
        The elevation (in degrees) at which full power is reached.

    Returns a factor between 0 and 1.
    """
    if solar_elevation <= 0:
        return 0
    elif solar_elevation < threshold:
        return max(min_factor, sin((pi / 2) * (solar_elevation / threshold)))
    else:
        return 1


def calculate_evening_ramp(solar_elevation, threshold=10):
    """
    Computes a factor for the gradual ramp-down of power in the evening.

    For solar elevations between 0 and 'threshold' (in degrees),
    the factor ramps from 1 down to 0 using a sine function.
    (The same sine function used for ramp-up produces the desired shape when solar_elevation
    goes from the threshold down to zero.)

    - solar_elevation : float
        The current solar elevation (in degrees).
    - threshold : float, optional
        The elevation (in degrees) below which the power starts to ramp down.

    Returns a factor between 0 and 1.
    """
    if solar_elevation <= 0:
        return 0
    elif solar_elevation < threshold:
        # When solar elevation is low, the same sine-based function applies:
        # at 0 => 0; at threshold => sin(pi/2)=1.
        # In usage, you would apply this factor in the evening (when solar elevation is falling)
        # to gradually bring the power output to zero.
        return sin((pi / 2) * (solar_elevation / threshold))
    else:
        return 1


def apply_morning_and_evening_ramps(df, threshold=10):
    """
    Applies the morning and evening ramp corrections to the predicted power based on solar elevation.

    For times earlier than noon, the morning ramp factor is applied; for times later than noon,
    the evening ramp factor is applied.

    Parameters:
      df : pandas.DataFrame
          DataFrame containing at least 'LocalTime', 'solar_elevation', and 'Power(MW)'.
      threshold : float, optional
          The solar elevation threshold (in degrees) over which full power can be expected.

    Returns:
      pandas.DataFrame with adjusted 'Power(MW)'.
    """

    def _apply_ramp(row):
        # Use a simple rule: before noon use ramp-up; after noon use ramp–down.
        # (Adjust this decision rule depending on your site characteristics.)
        hour = row['LocalTime'].hour + row['LocalTime'].minute / 60.0
        if hour < 12:
            ramp_factor = calculate_morning_ramp(row['solar_elevation'], threshold)
        else:
            ramp_factor = calculate_evening_ramp(row['solar_elevation'], threshold)
        return row['Power(MW)'] * ramp_factor

    df['Power(MW)'] = df.apply(_apply_ramp, axis=1)
    return df


def compute_minutes_sunrise_sunset_from_elevation(group):
    """
    Computes two features for each group (e.g., one plant’s data for a single day)
    based solely on the solar_elevation data:
      - minutes_since_sunrise: Minutes elapsed since the interpolated sunrise time.
      - minutes_until_sunset: Minutes remaining until the interpolated sunset time.

    The sunrise (or sunset) time is determined via linear interpolation between the
    two consecutive timestamps where solar_elevation crosses zero.

    Assumes that the group DataFrame contains:
      - 'LocalTime': datetime values (pd.Timestamp) sorted in ascending order.
      - 'solar_elevation': the computed solar elevation (in degrees).
    """
    # Ensure the group is sorted by LocalTime.
    group = group.sort_values('LocalTime').copy()
    times = group['LocalTime'].tolist()
    elevs = group['solar_elevation'].values.astype(float)

    sunrise_time = None
    sunset_time = None

    # --- Determine Sunrise Time ---
    # Look for the first crossing from <=0 to >0.
    for i in range(1, len(elevs)):
        if elevs[i - 1] <= 0 and elevs[i] > 0:
            t0 = times[i - 1]
            t1 = times[i]
            e0 = elevs[i - 1]
            e1 = elevs[i]
            # Avoid division by zero
            if (e1 - e0) != 0:
                ratio = (0 - e0) / (e1 - e0)
            else:
                ratio = 0
            sunrise_time = t0 + (t1 - t0) * ratio
            break
    # If no crossing is found, fallback: use the first time with positive elevation.
    if sunrise_time is None:
        pos_indices = np.where(elevs > 0)[0]
        if len(pos_indices) > 0:
            sunrise_time = times[pos_indices[0]]

    # --- Determine Sunset Time ---
    # Look for the last crossing from >0 to <=0.
    for i in range(len(elevs) - 1, 0, -1):
        if elevs[i - 1] > 0 and elevs[i] <= 0:
            t0 = times[i - 1]
            t1 = times[i]
            e0 = elevs[i - 1]
            e1 = elevs[i]
            if (e1 - e0) != 0:
                ratio = (0 - e0) / (e1 - e0)
            else:
                ratio = 0
            sunset_time = t0 + (t1 - t0) * ratio
            break
    # If no crossing is found, fallback: use the last time with positive elevation.
    if sunset_time is None:
        pos_indices = np.where(elevs > 0)[0]
        if len(pos_indices) > 0:
            sunset_time = times[pos_indices[-1]]

    # If still undefined, assign default values (first and last times).
    if sunrise_time is None:
        sunrise_time = times[0]
    if sunset_time is None:
        sunset_time = times[-1]

    # --- Compute the new features ---
    def calc_minutes_since_sunrise(t):
        # Compute minutes (as float) since sunrise, clip negative values.
        delta = (t - sunrise_time).total_seconds() / 60.0
        return max(delta, 0)

    def calc_minutes_until_sunset(t):
        # Compute minutes until sunset, clip negative values.
        delta = (sunset_time - t).total_seconds() / 60.0
        return max(delta, 0)

    group['minutes_since_sunrise'] = group['LocalTime'].apply(calc_minutes_since_sunrise)
    group['minutes_until_sunset'] = group['LocalTime'].apply(calc_minutes_until_sunset)

    return group

def interpolate_series(values, orig_times_numeric, new_times_numeric, method):
    """
    Interpolates a 1D array 'values' given the original timestamps (in seconds)
    and the new timestamps.

    If method == 'cosine', applies cosine-based interpolation to capture
    the smooth (sinusoidal) diurnal transition for variables like solar radiation.
    Otherwise (for 'linear'), performs a simple linear interpolation.

    Boundary points are handled by using the first or last available value.
    """
    # Initialize the result array with NaNs.
    interp = np.full(new_times_numeric.shape, np.nan, dtype=float)

    first_time = orig_times_numeric[0]
    last_time = orig_times_numeric[-1]

    # For new times earlier than or equal to the first original timestamp,
    # assign the first value.
    before_mask = new_times_numeric <= first_time
    interp[before_mask] = values[0]

    # For new times later than or equal to the last original timestamp,
    # assign the last value.
    after_mask = new_times_numeric >= last_time
    interp[after_mask] = values[-1]

    # For new times strictly within the original time bounds:
    in_range_mask = (new_times_numeric > first_time) & (new_times_numeric < last_time)
    if not np.any(in_range_mask):
        return interp

    valid_new_times = new_times_numeric[in_range_mask]
    # For each valid new time, find indices in the original array that bracket it.
    idx_upper = np.searchsorted(orig_times_numeric, valid_new_times, side='right')
    idx_lower = idx_upper - 1

    lower_vals = values[idx_lower]
    upper_vals = values[idx_upper]
    lower_times = orig_times_numeric[idx_lower]
    upper_times = orig_times_numeric[idx_upper]

    # Compute the fractional position between the two times.
    delta = (upper_times - lower_times).astype(float)
    frac = (valid_new_times - lower_times) / np.where(delta == 0, 1, delta)

    if method == 'cosine':
        weight = (1 - np.cos(np.pi * frac)) / 2
        interp[in_range_mask] = lower_vals + (upper_vals - lower_vals) * weight
    else:  # linear interpolation
        interp[in_range_mask] = lower_vals + (upper_vals - lower_vals) * frac

    return interp


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


def disaggregate_monthly_to_interval_mc(monthly_forecast, daily_profile, noise_daily, daily_factor_clip, interval):
    """
    Disaggregate a monthly forecast series into a finer interval (e.g. 15min) by:
      - Splitting the monthly total equally among all days,
      - Using a typical daily profile (filtered by month, weekday, and holiday) to distribute each day's total,
      - Applying a random noise factor (clamped by daily_factor_clip).

    Parameters:
      monthly_forecast (pd.Series): Monthly consumption totals.
      daily_profile (pd.DataFrame): Typical fractions for each day segment, must include columns:
            'month', 'weekday', 'holiday', 'time', 'fraction'
      noise_daily (float): The range for random noise (e.g., 0.02 for ±2% noise).
      daily_factor_clip (float): Maximum offset allowed in the daily noise.
      interval (str): Target interval frequency (e.g., '15min').

    Returns:
      pd.Series: Forecast disaggregated at the specified interval.
    """
    forecast_interval_list = []
    # Determine the number of periods in one day based on the interval
    n_periods = int(pd.Timedelta('1 day') / pd.Timedelta(interval))
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
            raw_factor = np.random.uniform(-noise_daily, noise_daily)
            raw_factor = np.clip(raw_factor, -daily_factor_clip, daily_factor_clip)
            random_factor = 1 + raw_factor
            day_values = daily_total * random_factor * profile_day['fraction'].values
            s = pd.Series(day_values, index=times)
            forecast_interval_list.append(s)
    forecast_interval = pd.concat(forecast_interval_list)
    return forecast_interval


def aggregate_power_output_to_excel(df, output_file='Aggregated_Wind_Energy.xlsx'):
    """
    Aggregate the power output of all wind farms by timestamp and save the result to an Excel file.

    Parameters:
        df (pandas.DataFrame): DataFrame containing wind farm outputs with a 'LocalTime' column and 'power_kWh' column.
        output_file (str, optional): Path to the output Excel file.
    """
    # Ensure LocalTime is a datetime type for proper processing and Excel export.
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])

    # Aggregate power output by LocalTime and extract date components.
    aggregated = (
        df.groupby('LocalTime', as_index=False)['power_kWh']
        .sum()
        .assign(Year=lambda x: x['LocalTime'].dt.year,
                Month=lambda x: x['LocalTime'].dt.month,
                Day=lambda x: x['LocalTime'].dt.day,
                Hour=lambda x: x['LocalTime'].dt.hour)
    )
    aggregated.to_excel(output_file, index=False)
    print(f"Aggregated data saved to {output_file}")