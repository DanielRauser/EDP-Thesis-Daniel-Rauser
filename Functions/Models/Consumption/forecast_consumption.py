import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm

from Functions.Models.Consumption.helpers import (
    get_season,
    renormalize_profile,
    disaggregate_monthly_to_interval_mc,
    create_and_log_forecast_plot,
    log_fig,
    get_extended_pt_holidays
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def forecast_consumption(
        redes_data_path,
        final_year,
        growth_rates,
        interval='15min',
        num_simulations=10,
        noise_daily=0.02,
        daily_factor_clip=0.03,
        output_path='forecast.pkl'
):
    """
    Long-term forecast of electricity consumption at a specified resolution using a hybrid approach.

    This function combines:
      - A deterministic monthly trend forecast, based on a specified annual growth rate (or a year-to-year mapping),
      - A Monte Carlo disaggregation that leverages historical seasonal profiles to distribute monthly totals
        into the chosen interval.

    In addition to computing the forecast, the function:
      - Logs a main forecast figure (showing the median forecast along with a 5%-95% confidence band),
      - Plots all Monte Carlo simulation paths in semi-transparent lines,
      - Logs additional consumption profile summary plots for weekdays, national holidays, Sundays, winter, summer,
        spring, and autumn,
      - Saves the final forecast DataFrame as an Excel artifact.

    Parameters:
      redes_data_path (str): Path to the redes consumption data.
      final_year (int): The final forecast year (e.g., 2045).
      growth_rates (float or dict): A single annual growth rate (e.g., 0.01) or a dictionary mapping years to rates.
      interval (str): Time resolution for the forecast (e.g., '15min', '30min').
      num_simulations (int): Number of Monte Carlo simulations to perform.
      noise_daily (float): Magnitude of daily random noise to perturb consumption.
      daily_factor_clip (float): Maximum clamping factor for daily noise.
      output_path (str): Directory path to save the final forecast DataFrame (Excel format).

    Returns:
      pd.DataFrame: A DataFrame with a datetime column 'LocalTime' and the following columns:
          - 'mean (kWh)'
          - 'median (kWh)'
          - '80% (kWh)'
          - '90% (kWh)'
          - '99% (kWh)'
          - 'weekday'
          - 'holiday'
          - Additional columns for month, season, and time.
    """
    logger.info("Creating and logging the consumption forecast...")

    forecast_output_path = os.path.join(output_path, "Forecast", f'consumption_forecast_{interval}.xlsx')

    # If forecast already exists, just return it.
    if os.path.exists(forecast_output_path):
        logger.info(f'Forecast data on {interval} level already exists in {forecast_output_path}')
        forecast_df = pd.read_excel(forecast_output_path)
        return forecast_df

    # ---------------------------
    # 1. Preprocess Input Data
    # ---------------------------
    df = pd.read_excel(redes_data_path)
    df = df.copy()
    if not np.issubdtype(df['Date/Time'].dtype, np.datetime64):
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df.set_index('Date/Time', inplace=True)
    df['Total Con (kWh)'] = pd.to_numeric(df['Total Con (kWh)'], errors='coerce')
    df_cleaned = df[~df.index.duplicated(keep='first')].copy()
    df_cleaned['Total Con (kWh)'] = df_cleaned['Total Con (kWh)'].ffill()

    # ---------------------------
    # 2. Create Deterministic Monthly Forecast
    # ---------------------------
    df_monthly = df_cleaned['Total Con (kWh)'].resample('ME').sum()
    baseline_value = df_monthly.iloc[-1]
    last_day = df_cleaned.index.max().normalize()
    sim_start = last_day + pd.Timedelta(days=1)
    sim_end = pd.to_datetime(f"{final_year}-12-31")
    forecast_index_monthly = pd.date_range(start=sim_start, end=sim_end, freq='ME')

    # Process growth rates: support a single numeric value or a dict mapping years to rates.
    if isinstance(growth_rates, (int, float)):
        growth_rates_dict = {year: growth_rates for year in range(sim_start.year, sim_end.year + 1)}
    elif isinstance(growth_rates, dict):
        growth_rates_dict = {}
        last_rate = 0.01  # Default if not specified
        for year in range(sim_start.year, sim_end.year + 1):
            if year in growth_rates:
                # Convert % to decimal if user provided e.g. 2 => 2% => 0.02
                last_rate = growth_rates[year] / 100.0
            growth_rates_dict[year] = last_rate
    else:
        raise ValueError("growth_rates must be a numeric value or a dictionary mapping years to rates.")

    deterministic_forecast_values = []
    current_value = baseline_value
    for date in forecast_index_monthly:
        annual_growth = growth_rates_dict[date.year]
        monthly_factor = (1 + annual_growth) ** (1 / 12)
        current_value *= monthly_factor
        deterministic_forecast_values.append(current_value)
    deterministic_forecast_series = pd.Series(deterministic_forecast_values, index=forecast_index_monthly)

    # ---------------------------
    # 3. Build Typical Consumption Profile from Historical Data
    # ---------------------------
    df_interval = df_cleaned['Total Con (kWh)'].resample(interval).ffill().to_frame('consumption')
    df_interval['date'] = df_interval.index.normalize()
    df_interval['time'] = df_interval.index.time
    df_interval['weekday'] = df_interval.index.day_name()
    df_interval['month'] = df_interval.index.month
    df_interval['season'] = df_interval['month'].map(get_season)

    hist_years = df_interval.index.year.unique()
    extended_holidays = get_extended_pt_holidays(hist_years.min(), final_year)
    df_interval['holiday'] = df_interval.index.map(
        lambda x: 1 if x.date() in extended_holidays else 0
    )

    daily_totals = df_interval.groupby('date')['consumption'].transform('sum')
    df_interval['fraction'] = df_interval['consumption'] / daily_totals
    daily_profile = (
        df_interval
        .groupby(['month', 'weekday', 'holiday', 'time'])['fraction']
        .mean()
        .reset_index()
    )
    # Renormalize each group so fractions sum to 1
    daily_profile = daily_profile.groupby(['month', 'weekday', 'holiday'], group_keys=False).apply(renormalize_profile)

    # ---------------------------
    # 4. Monte Carlo Simulation for Interval Forecast
    # ---------------------------
    all_simulations = []
    for sim in tqdm(range(num_simulations), desc="Processing Simulations"):
        # Add small random monthly variation around deterministic forecast
        monthly_variation = deterministic_forecast_series * (
            1 + np.random.uniform(-0.02, 0.02, size=len(deterministic_forecast_series))
        )
        sim_forecast = disaggregate_monthly_to_interval_mc(
            monthly_variation,
            daily_profile,
            noise_daily,
            daily_factor_clip,
            interval
        )
        all_simulations.append(sim_forecast)

    # Combine all simulations into one DataFrame
    simulations_df = pd.concat(all_simulations, axis=1)
    simulations_df.columns = [f"sim_{i}" for i in range(num_simulations)]

    # ---------------------------
    # 4a. Clip Outliers (both lower & upper)
    #     Adjust the quantiles if needed
    # ---------------------------
    lower_clip = simulations_df.stack().quantile(0.005)  # 0.5th percentile
    upper_clip = simulations_df.stack().quantile(0.995)  # 99.5th percentile
    simulations_df = simulations_df.clip(lower=lower_clip, upper=upper_clip)

    # ---------------------------
    # 4b. Compute Summary Statistics
    # ---------------------------
    mean_forecast = simulations_df.mean(axis=1)
    median_forecast = simulations_df.median(axis=1)
    quantile_80 = simulations_df.quantile(0.8, axis=1)
    quantile_90 = simulations_df.quantile(0.9, axis=1)
    quantile_99 = simulations_df.quantile(0.99, axis=1)
    quantile_05 = simulations_df.quantile(0.05, axis=1)
    quantile_95 = simulations_df.quantile(0.95, axis=1)

    # Build final forecast DataFrame
    forecast_df = pd.DataFrame({
        'mean (kWh)': mean_forecast,
        'median (kWh)': median_forecast,
        '80% (kWh)': quantile_80,
        '90% (kWh)': quantile_90,
        '99% (kWh)': quantile_99
    }, index=simulations_df.index)

    # ---------------------------
    # 5. Add Time, Weekday, Holiday, Season Columns
    # ---------------------------
    forecast_df['LocalTime'] = forecast_df.index
    forecast_df['weekday'] = forecast_df['LocalTime'].dt.day_name()
    forecast_df['month'] = forecast_df['LocalTime'].dt.month
    forecast_df['holiday'] = forecast_df.index.map(
        lambda x: 1 if x.date() in extended_holidays else 0
    )
    forecast_df['season'] = forecast_df['month'].map(get_season)
    forecast_df['time'] = forecast_df['LocalTime'].dt.time

    # Reorder columns
    cols = ['LocalTime'] + [col for col in forecast_df.columns if col != 'LocalTime']
    forecast_df = forecast_df[cols]

    # ---------------------------
    # 6. Plot and Log the Main Forecast Figure
    #    - Plot all simulations in gray
    #    - Median forecast in blue
    #    - 5%-95% confidence band around the median
    # ---------------------------
    fig, ax = plt.subplots(figsize=(14, 7))

    for sim_col in simulations_df.columns:
        ax.plot(
            forecast_df['LocalTime'],
            simulations_df[sim_col],
            color='gray',
            alpha=0.1,
            label='_nolegend_'
        )

    # Then add just a few meaningful legend entries
    ax.plot(
        forecast_df['LocalTime'],
        forecast_df['median (kWh)'],
        label='Median Forecast',
        color='blue',
        linewidth=2
    )
    ax.fill_between(
        forecast_df['LocalTime'],
        quantile_05,
        quantile_95,
        color='blue',
        alpha=0.2,
        label='5%-95% Range'
    )
    ax.legend()  # Now the legend only has two items

    ax.set_title(f'{interval} Consumption Forecast until {final_year} (Monte Carlo Simulation)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Con (kWh)')
    ax.legend()
    fig.tight_layout()
    log_fig(fig, "forecast.png")

    # ---------------------------
    # 7. Create and Log Additional Consumption Profile Plots
    # ---------------------------
    additional_fig_dir = 'additional_figs'
    os.makedirs(additional_fig_dir, exist_ok=True)

    def plot_category(category_filter, category_name, file_name):
        df_cat = forecast_df[category_filter]
        # Group by time-of-day and compute median consumption and confidence intervals
        grouped = df_cat.groupby('time').agg({
            'median (kWh)': 'median',
            '80% (kWh)': 'median',
            '90% (kWh)': 'median'
        }).reset_index()
        create_and_log_forecast_plot(
            x=grouped['time'],
            y=grouped['median (kWh)'],
            y_lower=grouped['80% (kWh)'],
            y_upper=grouped['90% (kWh)'],
            title=f'Consumption Forecast for {category_name}',
            xlabel='Time of Day',
            ylabel='Consumption (kWh)',
            file_path=os.path.join(additional_fig_dir, file_name),
            label=category_name
        )

    # Weekdays (Monday to Friday)
    plot_category(forecast_df['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
                  'Weekdays (Mon-Fri)', 'forecast_weekdays.png')
    # National Holidays
    plot_category(forecast_df['holiday'] == 1,
                  'National Holidays', 'forecast_holidays.png')
    # Sundays
    plot_category(forecast_df['weekday'] == 'Sunday',
                  'Sundays', 'forecast_sundays.png')
    # Winter
    plot_category(forecast_df['season'] == 'Winter',
                  'Winter Months', 'forecast_winter.png')
    # Summer
    plot_category(forecast_df['season'] == 'Summer',
                  'Summer Months', 'forecast_summer.png')
    # Spring
    plot_category(forecast_df['season'] == 'Spring',
                  'Spring Months', 'forecast_spring.png')
    # Autumn
    plot_category(forecast_df['season'] == 'Fall',
                  'Autumn Months', 'forecast_autumn.png')

    # ---------------------------
    # 8. Save and Log the Final Forecast DataFrame
    # ---------------------------
    forecast_df.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.dirname(forecast_output_path), exist_ok=True)
    forecast_df.to_excel(forecast_output_path, index=False)

    logger.info(f"Forecast results saved to: {forecast_output_path}")

    return forecast_df