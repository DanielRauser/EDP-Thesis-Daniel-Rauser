import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import pandas as pd
import mlflow
import holidays
import logging

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12  # Minimum font size

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_season(month):
    """Return the season name given a month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def renormalize_profile(month_df):
    """Renormalize the 'fraction' column so that it sums to 1."""
    total = month_df['fraction'].sum()
    if total > 0:
        month_df['fraction'] = month_df['fraction'] / total
    return month_df


def log_fig(fig, filename):
    """Log a given figure via MLflow and then close it."""
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def disaggregate_monthly_to_interval_mc(monthly_forecast, daily_profile, noise_daily, daily_factor_clip, interval):
    """
    Disaggregate a monthly forecast into an interval series using Monte Carlo methods.
    Caches holiday calendars and daily profiles for faster lookups.
    """
    forecast_interval_list = []
    n_periods = int(pd.Timedelta('1 day') / pd.Timedelta(interval))
    # Precompute a constant time index for reindexing the daily profile.
    base_day = pd.Timestamp('2000-01-01')
    day_time_index = pd.date_range(start=base_day, periods=n_periods, freq=interval).time

    # Caches for holiday calendars and daily profiles.
    holiday_cache = {}
    profile_cache = {}

    for month_end, monthly_value in monthly_forecast.items():
        # Get the first day of the month and all days till the end.
        month_start = month_end.replace(day=1)
        all_days = pd.date_range(start=month_start, end=month_end, freq='D')
        num_days = len(all_days)
        daily_total = monthly_value / num_days

        for day in all_days:
            day_date = pd.to_datetime(day)
            day_year = day_date.year
            if day_year not in holiday_cache:
                holiday_cache[day_year] = holidays.country_holidays('PT', years=[day_year])
            day_holiday = 1 if day_date.date() in holiday_cache[day_year] else 0
            day_month = day_date.month
            day_weekday = day_date.day_name()

            # Use a cache key based on (month, weekday, holiday)
            profile_key = (day_month, day_weekday, day_holiday)
            if profile_key in profile_cache:
                cached_profile = profile_cache[profile_key]
            else:
                profile_filter = (
                        (daily_profile['month'] == day_month) &
                        (daily_profile['weekday'] == day_weekday) &
                        (daily_profile['holiday'] == day_holiday)
                )
                cached_profile = daily_profile[profile_filter].copy()
                if cached_profile.empty:
                    # Fallback to using only month and weekday if needed.
                    profile_filter = (
                            (daily_profile['month'] == day_month) &
                            (daily_profile['weekday'] == day_weekday)
                    )
                    cached_profile = daily_profile[profile_filter].copy()
                # Reindex the daily profile to match the target time intervals.
                cached_profile = cached_profile.set_index('time').reindex(day_time_index)
                # Interpolate missing fractions and renormalize.
                cached_profile['fraction'] = cached_profile['fraction'].interpolate(method='linear')
                cached_profile['fraction'] = cached_profile['fraction'] / cached_profile['fraction'].sum()
                profile_cache[profile_key] = cached_profile

            # Create the time index for the current day.
            times = pd.date_range(start=day_date.normalize(), periods=n_periods, freq=interval)
            # Apply random noise with clipping.
            raw_factor = np.random.uniform(-noise_daily, noise_daily)
            raw_factor = np.clip(raw_factor, -daily_factor_clip, daily_factor_clip)
            random_factor = 1 + raw_factor
            day_values = daily_total * random_factor * cached_profile['fraction'].values
            s = pd.Series(day_values, index=times)
            forecast_interval_list.append(s)
    forecast_interval = pd.concat(forecast_interval_list)
    return forecast_interval


def get_extended_pt_holidays(start_year, end_year):
    """
    Get the set of Portuguese holiday dates from start_year to end_year.
    If end_year exceeds 2033, replicate the pattern from 2033.
    """
    start_year = int(start_year)
    end_year = int(end_year)
    known_holidays = holidays.country_holidays('PT', years=range(start_year, min(end_year, 2033) + 1))
    all_holiday_dates = set(known_holidays.keys())

    if end_year > 2033:
        for year in range(2034, end_year + 1):
            for dt in known_holidays.keys():
                if dt.year == 2033:
                    new_dt = dt.replace(year=year)
                    all_holiday_dates.add(new_dt)
    return all_holiday_dates


def load_data(filepath, date_col="Date/Time"):
    """
    Load an Excel file and create a 'LocalTime' column based on the provided date column.
    """
    df = pd.read_excel(filepath)
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df['LocalTime'] = pd.to_datetime(df[date_col])
    else:
        df['LocalTime'] = df[date_col]
    return df


def clip_series(series, lower_q=0.005, upper_q=0.995):
    """
    Clip a pandas Series using lower and upper quantile thresholds.
    """
    lower_bound = series.quantile(lower_q)
    upper_bound = series.quantile(upper_q)
    return series.clip(lower_bound, upper_bound)


def compute_deterministic_series(baseline_value, forecast_index, growth_rates):
    """
    Compute a monthly deterministic forecast series.
    Handles growth_rates given as a scalar or as a dictionary mapping years to rates.

    IMPORTANT FIX: When growth_rates is provided as a dict, we now convert the values from percentage
    (e.g. 2 for 2%) to decimal (0.02) by dividing by 100.
    """
    if isinstance(growth_rates, (int, float)):
        growth_dict = {year: growth_rates for year in range(forecast_index[0].year, forecast_index[-1].year + 1)}
    elif isinstance(growth_rates, dict):
        growth_dict = {}
        last_rate = 0.01
        for year in range(forecast_index[0].year, forecast_index[-1].year + 1):
            if year in growth_rates:
                # Convert percentage to a decimal rate.
                last_rate = growth_rates[year] / 100.0
            growth_dict[year] = last_rate
    else:
        raise ValueError("growth_rates must be numeric or a dict mapping years to rates.")

    series_values = []
    current_value = baseline_value
    for date in forecast_index:
        monthly_factor = (1 + growth_dict[date.year]) ** (1 / 12)
        current_value *= monthly_factor
        series_values.append(current_value)
    return pd.Series(series_values, index=forecast_index)


def build_daily_profile(df, value_col, interval, forecast_year_end):
    """
    Build a typical daily consumption profile from historical data.

    Resamples the provided DataFrame at the target interval, computes per-day fractions,
    and groups the data by month, weekday, and holiday status.
    """
    df_interval = df[value_col].resample(interval).ffill().to_frame("value")
    df_interval["date"] = df_interval.index.normalize()
    df_interval["time"] = df_interval.index.time
    df_interval["weekday"] = df_interval.index.day_name()
    df_interval["month"] = df_interval.index.month
    df_interval["season"] = df_interval["month"].map(get_season)

    hist_years = df_interval.index.year.unique()
    extended_holidays = get_extended_pt_holidays(hist_years.min(), forecast_year_end)
    df_interval["holiday"] = df_interval.index.map(lambda x: 1 if x.date() in extended_holidays else 0)

    daily_totals = df_interval.groupby("date")["value"].transform("sum")
    df_interval["fraction"] = df_interval["value"] / daily_totals

    daily_profile = (
        df_interval.groupby(["month", "weekday", "holiday", "time"])["fraction"]
        .mean()
        .reset_index()
    )
    daily_profile = daily_profile.groupby(["month", "weekday", "holiday"], group_keys=False).apply(renormalize_profile)

    return daily_profile


def run_monte_carlo_simulation(deterministic_series, daily_profile, num_simulations, noise_daily, daily_factor_clip,
                               interval):
    """
    Run Monte Carlo simulations based on the deterministic series and daily profile.
    Adds a small random perturbation to the monthly series before disaggregation.
    """
    simulation_list = []
    for sim in tqdm(range(num_simulations), desc="Processing Simulations"):
        monthly_variation = deterministic_series * (1 + np.random.uniform(-0.02, 0.02, size=len(deterministic_series)))
        sim_forecast = disaggregate_monthly_to_interval_mc(
            monthly_variation,
            daily_profile,
            noise_daily,
            daily_factor_clip,
            interval
        )
        simulation_list.append(sim_forecast)
    simulations_df = pd.concat(simulation_list, axis=1)
    simulations_df.columns = [f"sim_{i}" for i in range(num_simulations)]
    lower_clip = simulations_df.stack().quantile(0.005)
    upper_clip = simulations_df.stack().quantile(0.995)
    simulations_df = simulations_df.clip(lower=lower_clip, upper=upper_clip)
    return simulations_df


def compute_error_metrics(actual, forecast):
    """
    Compute RMSE, MAE, and MAPE evaluation metrics.
    Both actual and forecast inputs must be aligned pandas Series.
    """
    error = forecast - actual
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mape = np.mean(np.abs(error / actual)) * 100
    return rmse, mae, mape


def plot_evaluation(common_index, forecast_median, quantile_05, quantile_95, actual, unit_divisor=1e3, yaxis=""):
    """
    Plot a comparison of the forecast (median and confidence intervals) against actual data.
    """
    # Align all series to the common index.
    forecast_median = forecast_median.reindex(common_index)
    quantile_05 = quantile_05.reindex(common_index)
    quantile_95 = quantile_95.reindex(common_index)
    actual = actual.reindex(common_index)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.fill_between(common_index,
                    quantile_05 / unit_divisor,
                    quantile_95 / unit_divisor,
                    color='skyblue', alpha=0.6,
                    label='5%-95% Confidence Interval')
    ax.plot(common_index,
            forecast_median / unit_divisor,
            label='Forecast Median',
            color='darkblue',
            linestyle='--',
            linewidth=1.5)
    ax.plot(common_index,
            actual / unit_divisor,
            label='Actual Production (Unclipped)',
            color='forestgreen',
            linewidth=2)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(yaxis, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    log_fig(fig, "evaluation_forecast.png")
    return fig


def plot_forecast(forecast_df, simulations_df, unit_divisor=1e6):
    """
    Plot the final forecast alongside all simulation outcomes, displaying median, mean, and a confidence band.
    """
    quantile_05 = simulations_df.quantile(0.05, axis=1)
    quantile_95 = simulations_df.quantile(0.95, axis=1)
    median_forecast = simulations_df.median(axis=1)
    mean_forecast = simulations_df.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot each simulation as a faint line.
    for sim_col in simulations_df.columns:
        ax.plot(forecast_df['LocalTime'],
                simulations_df[sim_col] / unit_divisor,
                color='gray',
                alpha=0.1,
                label='_nolegend_')

    ax.plot(forecast_df['LocalTime'],
            median_forecast / unit_divisor,
            label='Median Forecast',
            color='blue',
            linewidth=2)

    ax.plot(forecast_df['LocalTime'],
            mean_forecast / unit_divisor,
            label='Mean Forecast',
            color='orange',
            linewidth=2)

    ax.fill_between(forecast_df['LocalTime'],
                    (quantile_05 / unit_divisor),
                    (quantile_95 / unit_divisor),
                    color='blue',
                    alpha=0.2,
                    label='5%-95% Confidence')

    ax.set_xlabel('Time')
    ax.set_ylabel('Consumption in GWh')
    ax.legend(loc='upper left')
    fig.tight_layout()
    log_fig(fig, "forecast_consumption.png")
    return fig


