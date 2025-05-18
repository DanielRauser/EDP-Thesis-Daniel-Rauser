import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import mlflow
from Functions.Models.Consumption.helpers import (
    load_and_preprocess_data,
    log_fig,
    disaggregate_monthly_to_interval_mc,
    build_daily_profile,
    get_season,
    get_extended_pt_holidays
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def forecast_consumption(redes_data_path, final_year, growth_rates, interval='15min',
                         num_simulations=10, noise_daily=0.02, daily_factor_clip=0.03, output_path='forecast'):
    logger.info("Starting consumption forecast...")
    forecast_output_path = os.path.join(output_path, "Forecast", f'consumption_forecast_{interval}.xlsx')
    if os.path.exists(forecast_output_path):
        logger.info("Forecast already exists!")
        return

    # Load and clip data
    df_actual, df_clipped = load_and_preprocess_data(redes_data_path, target_column="Consumption (kWh)")
    start_date = pd.to_datetime("2020-01-01")
    df_actual = df_actual[df_actual.index >= start_date]
    df_clipped = df_clipped[df_clipped.index >= start_date]

    # --- Hold-out Evaluation (if enough historical data) ---
    end_year_data = df_clipped.index.year.max()
    if (end_year_data - start_date.year) >= 3:
        training_end_year = end_year_data - 2
        training_end = pd.to_datetime(f"{training_end_year}-12-31")
        training_df = df_clipped.loc[df_clipped.index <= training_end]
        evaluation_df_actual = df_actual.loc[df_actual.index > training_end]

        # Historical monthly sums on training set
        df_train_monthly = training_df['Consumption (kWh)'].resample('ME').sum()
        baseline_value_train = df_train_monthly.iloc[-1]
        last_day_train = training_df.index.max().normalize()
        sim_start_eval = last_day_train + pd.Timedelta(days=1)
        sim_end_eval = evaluation_df_actual.index.max().normalize()
        forecast_index_monthly_eval = pd.date_range(start=sim_start_eval, end=sim_end_eval, freq='ME')

        # Build growth_rates dict
        if isinstance(growth_rates, (int, float)):
            growth_rates_dict = {
                year: growth_rates
                for year in range(sim_start_eval.year, sim_end_eval.year + 1)
            }
        else:
            growth_rates_dict = {
                year: growth_rates.get(year, 0.0)
                for year in range(sim_start_eval.year, sim_end_eval.year + 1)
            }

        # Compute multiplicative seasonal index from training months
        monthly_avgs = df_train_monthly.groupby(df_train_monthly.index.month).mean()
        overall_avg = df_train_monthly.mean()
        seasonal_index_train = (monthly_avgs / overall_avg).to_dict()

        # Build deterministic + seasonal series
        deterministic_values = []
        current_value = baseline_value_train
        for dt in forecast_index_monthly_eval:
            # compound growth
            annual_growth = growth_rates_dict.get(dt.year, 0)
            current_value *= (1 + annual_growth) ** (1 / 12)
            # seasonal multiplier
            season_mult = seasonal_index_train.get(dt.month, 1.0)
            deterministic_values.append(current_value * season_mult)

        deterministic_series = pd.Series(deterministic_values, index=forecast_index_monthly_eval)

        # Build daily profile from training data
        daily_profile_train = build_daily_profile(training_df, "Consumption (kWh)", interval)

        # Monte Carlo simulation for evaluation period
        sim_list = []
        for _ in tqdm(range(num_simulations), desc="Evaluation Simulations"):
            noise_factor = np.random.uniform(-noise_daily, noise_daily, size=len(deterministic_series))
            monthly_variation = deterministic_series * (1 + noise_factor)
            sim_forecast = disaggregate_monthly_to_interval_mc(
                monthly_variation,
                daily_profile_train,
                noise_daily,
                daily_factor_clip,
                interval
            )
            sim_list.append(sim_forecast)

        simulations_df = pd.concat(sim_list, axis=1)
        simulations_df.columns = [f"sim_{i}" for i in range(num_simulations)]
        # Clip extremes
        lower_clip = simulations_df.stack().quantile(0.005)
        upper_clip = simulations_df.stack().quantile(0.995)
        simulations_df = simulations_df.clip(lower=lower_clip, upper=upper_clip)
        median_forecast = simulations_df.median(axis=1)

        # Evaluation metrics (RMSE, MAE, MAPE)
        actual_eval = evaluation_df_actual['Consumption (kWh)'].resample(interval).sum()
        non_zero_mean = actual_eval[actual_eval != 0].mean()
        actual_eval = actual_eval.replace(0, non_zero_mean)

        common_idx = median_forecast.index.intersection(actual_eval.index)
        error = median_forecast.loc[common_idx] - actual_eval.loc[common_idx]
        rmse = np.sqrt((error ** 2).mean())
        mae = np.abs(error).mean()
        mape = (np.abs(error / actual_eval.loc[common_idx]).mean()) * 100

        mlflow.log_metric("consumption_rmse", rmse)
        mlflow.log_metric("consumption_mae", mae)
        mlflow.log_metric("consumption_mape", mape)
        logger.info(f"Evaluation Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

        # Plot evaluation forecast
        fig_eval, ax_eval = plt.subplots(figsize=(14, 7))
        ax_eval.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax_eval.plot(common_idx, median_forecast.loc[common_idx] / 1e6,
                     label="Forecast Median", linestyle='--')
        ax_eval.plot(common_idx, actual_eval.loc[common_idx] / 1e6,
                     label="Actual Consumption", linewidth=2)
        ax_eval.set_xlabel("Time")
        ax_eval.set_ylabel("Consumption (GWh)")
        ax_eval.legend(loc="upper right")
        fig_eval.autofmt_xdate()
        fig_eval.tight_layout()
        log_fig(fig_eval, "evaluation_consumption_forecast.png")
    else:
        logger.warning("Not enough data for evaluation. Skipping hold-out evaluation.")

    # --- Final Forecast using Full (Clipped) Historical Data ---
    df_monthly = df_clipped['Consumption (kWh)'].resample('ME').sum()
    baseline_value = df_monthly.iloc[-1]
    last_day = df_clipped.index.max().normalize()
    sim_start = last_day + pd.Timedelta(days=1)
    sim_end = pd.to_datetime(f"{final_year}-12-31")
    forecast_index_monthly = pd.date_range(start=sim_start, end=sim_end, freq='ME')

    # Build growth_rates dict
    if isinstance(growth_rates, (int, float)):
        growth_rates_dict = {
            year: growth_rates
            for year in range(sim_start.year, sim_end.year + 1)
        }
    else:
        growth_rates_dict = {
            year: growth_rates.get(year, 0.01)
            for year in range(sim_start.year, sim_end.year + 1)
        }

    # Compute seasonal index from full history
    monthly_avgs_full = df_monthly.groupby(df_monthly.index.month).mean()
    overall_avg_full = df_monthly.mean()
    seasonal_index_full = (monthly_avgs_full / overall_avg_full).to_dict()

    # Build deterministic + seasonal series for final forecast
    current_value = baseline_value
    deterministic_forecast = []
    for dt in forecast_index_monthly:
        current_value *= (1 + growth_rates_dict.get(dt.year, 0)) ** (1 / 12)
        season_mult = seasonal_index_full.get(dt.month, 1.0)
        deterministic_forecast.append(current_value * season_mult)

    deterministic_forecast_series = pd.Series(deterministic_forecast, index=forecast_index_monthly)

    # Full daily profile
    daily_profile_full = build_daily_profile(df_clipped, "Consumption (kWh)", interval)

    # Monte Carlo simulation for final forecast
    sim_list_final = []
    for _ in tqdm(range(num_simulations), desc="Final Forecast Simulations"):
        noise_factor = np.random.uniform(-noise_daily, noise_daily, size=len(deterministic_forecast_series))
        monthly_variation = deterministic_forecast_series * (1 + noise_factor)
        sim_forecast = disaggregate_monthly_to_interval_mc(
            monthly_variation,
            daily_profile_full,
            noise_daily,
            daily_factor_clip,
            interval
        )
        sim_list_final.append(sim_forecast)

    simulations_df_final = pd.concat(sim_list_final, axis=1)
    simulations_df_final.columns = [f"sim_{i}" for i in range(num_simulations)]
    lower_clip = simulations_df_final.stack().quantile(0.005)
    upper_clip = simulations_df_final.stack().quantile(0.995)
    simulations_df_final = simulations_df_final.clip(lower=lower_clip, upper=upper_clip)

    # Plot final consumption forecast
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    for col in simulations_df_final.columns:
        ax.plot(simulations_df_final.index, simulations_df_final[col] / 1e6, color='gray', alpha=0.1)
    median_forecast_final = simulations_df_final.median(axis=1)
    mean_forecast_final = simulations_df_final.mean(axis=1)
    ax.plot(median_forecast_final.index, median_forecast_final / 1e6,
            label="Median Forecast", linewidth=2)
    ax.plot(mean_forecast_final.index, mean_forecast_final / 1e6,
            label="Mean Forecast", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (GWh)")
    ax.set_title(f"{interval} Consumption Forecast until {final_year}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    log_fig(fig, "forecast_consumption.png")

    # Save to Excel
    mean_forecast = simulations_df_final.mean(axis=1)
    median_forecast = simulations_df_final.median(axis=1)
    quantile_05 = simulations_df_final.quantile(0.05, axis=1)
    quantile_95 = simulations_df_final.quantile(0.95, axis=1)

    forecast_df = pd.DataFrame({
        'mean (kWh)': mean_forecast,
        'median (kWh)': median_forecast,
        '5% (kWh)': quantile_05,
        '95% (kWh)': quantile_95
    }, index=simulations_df_final.index)
    forecast_df['LocalTime'] = forecast_df.index
    forecast_df['weekday'] = forecast_df['LocalTime'].dt.day_name()
    forecast_df['month'] = forecast_df['LocalTime'].dt.month

    hist_years = df_clipped.index.year.unique()
    extended_holidays = get_extended_pt_holidays(hist_years.min(), final_year)
    forecast_df['holiday'] = forecast_df.index.map(
        lambda x: 1 if x.date() in extended_holidays else 0
    )
    forecast_df['season'] = forecast_df['month'].map(get_season)
    forecast_df['time'] = forecast_df['LocalTime'].dt.time

    forecast_df = forecast_df[[
        'LocalTime', 'mean (kWh)', 'median (kWh)', '5% (kWh)', '95% (kWh)',
        'weekday', 'month', 'holiday', 'season', 'time'
    ]]
    os.makedirs(os.path.dirname(forecast_output_path), exist_ok=True)
    forecast_df.to_excel(forecast_output_path, index=False)
    logger.info(f"Forecast saved to {forecast_output_path}")