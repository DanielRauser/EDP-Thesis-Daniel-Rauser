import mlflow
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import logging
import holidays

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    total = month_df['fraction'].sum()
    month_df['fraction'] = month_df['fraction'] / total
    return month_df

def log_fig(fig, filename):
    """Log a given figure via MLflow and then close it."""
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def create_and_log_forecast_plot(x, y, y_lower, y_upper, title, xlabel, ylabel, file_path, label=None, figsize=(10, 6)):
    """
    Creates a plot using the provided forecast data (median and confidence intervals),
    saves it, and logs it as an MLflow artifact.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # If x values are time objects, convert them to datetime using an arbitrary reference date.
    if all(isinstance(t, datetime.time) for t in x):
        x = [pd.Timestamp.combine(datetime.date(2000, 1, 1), t) for t in x]

        # Set major ticks every 2 hours
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Optionally fix x-limits to a single 24-hour period
        ax.set_xlim([pd.Timestamp(2000, 1, 1, 0, 0),
                     pd.Timestamp(2000, 1, 1, 23, 59)])

        # Rotate and format the dates nicely
        fig.autofmt_xdate()

    # Plot median (or mean) line
    ax.plot(x, y, label=label)

    # Fill between lower and upper confidence intervals
    if y_lower is not None and y_upper is not None:
        ax.fill_between(x, y_lower, y_upper, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if label:
        ax.legend()

    fig.tight_layout()
    log_fig(fig, file_path)

def disaggregate_monthly_to_interval_mc(monthly_forecast, daily_profile, noise_daily, daily_factor_clip, interval):
    forecast_interval_list = []
    # Determine the number of periods in one day based on the interval
    n_periods = int(pd.Timedelta('1 day') / pd.Timedelta(interval))
    # For each month in the forecast
    for month_end, monthly_value in monthly_forecast.items():
        month_start = month_end.replace(day=1)
        all_days = pd.date_range(start=month_start, end=month_end, freq='D')
        num_days = len(all_days)
        # Distribute monthly total equally among days
        daily_total = monthly_value / num_days
        for day in all_days:
            day_date = pd.to_datetime(day)
            day_month = day_date.month
            day_weekday = day_date.day_name()
            # Determine holiday flag for this future day (Portugal)
            pt_future = holidays.country_holidays('PT', years=[day_date.year])
            day_holiday = 1 if day_date.date() in pt_future else 0
            # Create time index for the day at the chosen interval
            times = pd.date_range(start=day_date.normalize(), periods=n_periods, freq=interval)
            # Fetch the typical profile for this combination
            profile_filter = (
                (daily_profile['month'] == day_month) &
                (daily_profile['weekday'] == day_weekday) &
                (daily_profile['holiday'] == day_holiday)
            )
            profile_day = daily_profile[profile_filter].copy()
            # If no match found, fallback to grouping by month and weekday
            if profile_day.empty:
                profile_filter = (
                    (daily_profile['month'] == day_month) &
                    (daily_profile['weekday'] == day_weekday)
                )
                profile_day = daily_profile[profile_filter].copy()
            # Reindex the profile to match the full set of times in the day (using the time component)
            profile_day = profile_day.set_index('time').reindex([t.time() for t in times])
            profile_day['fraction'] = profile_day['fraction'].interpolate(method='linear')
            # Ensure the fractions sum to 1
            profile_day['fraction'] = profile_day['fraction'] / profile_day['fraction'].sum()
            # Add a random daily factor with clamping to avoid extreme outliers
            raw_factor = np.random.uniform(-noise_daily, noise_daily)
            raw_factor = np.clip(raw_factor, -daily_factor_clip, daily_factor_clip)
            random_factor = 1 + raw_factor
            day_values = daily_total * random_factor * profile_day['fraction'].values
            s = pd.Series(day_values, index=times)
            forecast_interval_list.append(s)
    forecast_interval = pd.concat(forecast_interval_list)
    return forecast_interval


def get_extended_pt_holidays(start_year, end_year):
    # Gather official data up to 2033
    known_holidays = holidays.country_holidays('PT', years=range(start_year, min(end_year, 2033) + 1))
    all_holiday_dates = set(known_holidays.keys())

    # If end_year > 2033, replicate the 2033 pattern
    # (This is naive for Easter-based or any "moving" holiday)
    if end_year > 2033:
        for year in range(2034, end_year + 1):
            # replicate each holiday date from 2033 to year
            for dt in known_holidays.keys():
                if dt.year == 2033:
                    new_dt = dt.replace(year=year)
                    all_holiday_dates.add(new_dt)

    # Make sure to return the set of holiday dates
    return all_holiday_dates

