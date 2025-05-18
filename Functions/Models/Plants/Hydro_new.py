import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import holidays
from tqdm import tqdm

# ------------------------------------------------
# Logger Setup
# ------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------
# SUPPORT FUNCTIONS: Data Loading, Season, Holidays, Daily Profile
# ------------------------------------------------

def load_and_preprocess_data(filepath, target_column, upper_clip=0.995, lower_clip=0.005):
    """
    Load data from Excel, parse dates, convert the target column to numeric,
    and return both the raw and cleaned versions.
    """
    df = pd.read_excel(filepath)
    if not np.issubdtype(df['LocalTime'].dtype, np.datetime64):
        df['LocalTime'] = pd.to_datetime(df.get('Date/Time', df['LocalTime']))
    df = df[['LocalTime', target_column]]
    df.set_index('LocalTime', inplace=True)
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df_actual = df[~df.index.duplicated(keep='first')].copy()
    df_cleaned = df_actual.copy()
    lower_val = df_cleaned[target_column].quantile(lower_clip)
    upper_val = df_cleaned[target_column].quantile(upper_clip)
    df_cleaned[target_column] = df_cleaned[target_column].clip(lower_val, upper_val)
    return df_actual, df_cleaned


def get_season(month):
    """Return season name for a given month."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def log_fig(fig, filename):
    """Log the figure via MLflow and then close it."""
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def get_extended_pt_holidays(start_year, end_year):
    """Return a set of Portuguese holiday dates between start_year and end_year."""
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


def build_daily_profile(df, target_column, interval):
    """
    Create a typical daily profile from historical data by resampling to the target interval
    and calculating normalized fractions per time point.
    """
    df_interval = df[target_column].resample(interval).ffill().to_frame('consumption')
    df_interval['date'] = df_interval.index.normalize()
    df_interval['time'] = df_interval.index.time
    df_interval['weekday'] = df_interval.index.day_name()
    df_interval['month'] = df_interval.index.month
    df_interval['season'] = df_interval['month'].map(get_season)
    hist_years = df_interval.index.year.unique()
    extended_holidays = get_extended_pt_holidays(hist_years.min(), df_interval.index.year.max())
    df_interval['holiday'] = df_interval.index.map(lambda x: 1 if x.date() in extended_holidays else 0)
    daily_totals = df_interval.groupby('date')['consumption'].transform('sum')
    df_interval['fraction'] = df_interval['consumption'] / daily_totals
    daily_profile = (df_interval
                     .groupby(['month', 'weekday', 'holiday', 'time'])['fraction']
                     .mean()
                     .reset_index())
    daily_profile = daily_profile.groupby(['month', 'weekday', 'holiday'], group_keys=False) \
        .apply(lambda grp: grp.assign(fraction=grp['fraction'] / grp['fraction'].sum()))
    return daily_profile


# ------------------------------------------------
# PLANT-LEVEL CLIMATE MODEL FUNCTIONS
# ------------------------------------------------

HYDRO_PARAMETERS = {
    "hydro - run-of-the-river": {
        "base_cf": 0.45,
        "curve": "constant",
        "precip_sensitivity": 0.025,
        "temp_sensitivity": 0.02
    },
    "hydro - water-pumped-storage": {
        "base_cf": 0.15,
        "curve": "variable",
        "precip_sensitivity": 0.05,
        "temp_sensitivity": 0.05
    },
    "hydro - water-storage": {
        "base_cf": 0.6,
        "curve": "variable",
        "precip_sensitivity": 0.15,
        "temp_sensitivity": 0.05
    },
    "hydro - small-hydro": {
        "base_cf": 0.35,
        "curve": "constant",
        "precip_sensitivity": 0.05,
        "temp_sensitivity": 0.025
    }
}


def get_hydro_parameters(plant_type):
    """
    Retrieve hydro parameters for the given plant type.
    Defaults to "hydro - run-of-the-river" if not found.
    """
    return HYDRO_PARAMETERS.get(plant_type.lower(), HYDRO_PARAMETERS["hydro - run-of-the-river"])


def predict_interval_hydro_output(tp, t2m, rated_capacity, plant_type, avg_annual_prod_GWh,
                                  storage=0, interval_hours=0.25, season="spring", seasonal_factors=None,
                                  lag_factor=1.0, meta_cf=None, tp_mean=1.0, dynamic_factor=0.1):
    """
    Predict energy output (kWh) for one interval using:
      - Non-linear precipitation adjustment.
      - Temperature adjustment.
      - Seasonal multipliers and (if applicable) lag/storage adjustments.
      - Dynamic clamping of the capacity factor.
    """
    params = get_hydro_parameters(plant_type)
    base_cf = meta_cf if meta_cf is not None else params["base_cf"]
    t2m_c = t2m - 273.15  # Convert Kelvin to Celsius
    if seasonal_factors is None:
        seasonal_factors = {"spring": 1.0, "summer": 0.85, "autumn": 0.9, "winter": 1.15}
    seasonal_multiplier = seasonal_factors.get(season.lower(), 1.0)
    precip_anomaly = tp - tp_mean
    precip_adjustment = np.tanh(precip_anomaly)
    precip_factor = 1 + params["precip_sensitivity"] * precip_adjustment
    temp_factor = 1 - params["temp_sensitivity"] * ((t2m_c - 15.0) / 15.0)
    if params["curve"] == "constant":
        cf = base_cf * seasonal_multiplier
    else:
        cf = base_cf * precip_factor * temp_factor * seasonal_multiplier
        if plant_type.lower() == "hydro - water-storage" and storage:
            cf *= (1 + 0.03 * ((storage - 50.0) / 50.0))
        if plant_type.lower() in ["hydro - water-storage", "hydro - water-pumped-storage"]:
            cf *= lag_factor
    historical_cf_limit = (avg_annual_prod_GWh * 1000) / (rated_capacity * 8760)
    if meta_cf is not None:
        historical_cf_limit = max(historical_cf_limit, meta_cf)
    dynamic_multiplier = 1 + dynamic_factor * (tp - tp_mean) / tp_mean if tp > tp_mean else 1.0
    dynamic_cf_limit = historical_cf_limit * dynamic_multiplier
    cf = min(max(cf, 0), dynamic_cf_limit)
    energy_kWh = rated_capacity * cf * interval_hours * 1000
    return energy_kWh


def process_hydro_outputs(df, interval="15min"):
    """
    Process plant-level data: reindex/interpolate if needed, derive season,
    compute lag features, and compute predicted energy output for each interval.
    """
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])
    df['LocalTime'] = df['LocalTime'].dt.tz_localize(None)
    df.sort_values('LocalTime', inplace=True)

    # Create unique plant identifier.
    df["ID"] = df["Plant Name"].astype(str) + "_" + df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    df.rename(columns={"Type 2": "Plant Type"}, inplace=True)
    df = df[df["Sold"] == "n"]  # filter for plants still with EDP

    plant_ids = df['ID'].unique()
    interval_hours = pd.to_timedelta(interval).total_seconds() / 3600
    results = []
    storage_types = ["hydro - water-storage", "hydro - water-pumped-storage"]

    for pid in tqdm(plant_ids, desc="Processing hydro plants"):
        df_plant = df[df['ID'] == pid].copy()
        # Check resolution and reindex/interpolate if needed.
        plant_dt = pd.Series(df_plant['LocalTime'].unique()).diff().dropna().min()
        if pd.isna(plant_dt) or plant_dt > pd.Timedelta(interval):
            df_plant.set_index('LocalTime', inplace=True)
            df_plant = df_plant[~df_plant.index.duplicated(keep='first')]
            new_index = pd.date_range(start=df_plant.index.min(), end=df_plant.index.max(), freq=interval)
            # (An interpolate_series function would be here if needed.)
            df_interp = df_plant.reindex(new_index).ffill()
            df_interp = df_interp.reset_index().rename(columns={'index': 'LocalTime'})
            df_interp['ID'] = pid
            df_plant = df_interp
        else:
            df_plant = df_plant.copy()

        # Determine the season.
        df_plant['season'] = df_plant['LocalTime'].dt.month.apply(
            lambda m: "winter" if m in [12, 1, 2]
            else "spring" if m in [3, 4, 5]
            else "summer" if m in [6, 7, 8]
            else "autumn")

        # Compute lag feature for storage/pumped plants.
        plant_meta = df[df['ID'] == pid].iloc[0]
        plant_type = plant_meta['Plant Type'].lower()
        if plant_type in storage_types and 'tp' in df_plant.columns:
            window = int((7 * 24) / interval_hours)
            df_plant['lag_tp'] = df_plant['tp'].rolling(window=window, min_periods=1).sum()
            baseline = df_plant['lag_tp'].median()
            df_plant['lag_factor'] = 1 + 0.1 * ((df_plant['lag_tp'] - baseline) / baseline)
        else:
            df_plant['lag_factor'] = 1.0

        # Use plant metadata to set parameters.
        rated_capacity = plant_meta['Total Capacity (MW)']
        if 'Average Annual Productivity (GWh)' in plant_meta and not pd.isnull(
                plant_meta['Average Annual Productivity (GWh)']):
            avg_annual_productivity = plant_meta['Average Annual Productivity (GWh)']
        else:
            avg_annual_productivity = 9999
        meta_cf = None
        if 'Capacity Factor' in plant_meta and not pd.isnull(plant_meta['Capacity Factor']):
            meta_cf = plant_meta['Capacity Factor']

        def compute_output(row):
            storage = row.get('Total Storage (hm3)', 0)
            return predict_interval_hydro_output(
                tp=row.get('tp', 1.0),
                t2m=row.get('t2m', 293.15),
                rated_capacity=rated_capacity,
                plant_type=row['Plant Type'],
                avg_annual_prod_GWh=avg_annual_productivity,
                storage=storage,
                interval_hours=interval_hours,
                season=row['season'],
                lag_factor=row.get('lag_factor', 1.0),
                meta_cf=meta_cf,
                tp_mean=0.00014696307,
                dynamic_factor=0.1
            )

        df_plant['power_kWh'] = df_plant.apply(compute_output, axis=1)
        results.append(df_plant)

    return pd.concat(results, ignore_index=True)


# ------------------------------------------------
# AGGREGATION & DISAGGREGATION FUNCTIONS
# ------------------------------------------------

def aggregate_plants(pred_df):
    """
    Aggregate predicted production from individual plants for each timestamp.
    """
    aggregated = pred_df.groupby('LocalTime')['power_kWh'].sum()
    return aggregated


def disaggregate_daily_series(aggregated_series, daily_profile, target_frequency="15min"):
    """
    Disaggregate a daily aggregated series to a finer resolution (target_frequency)
    by overlaying the typical daily profile.
    The process:
      - Resample the aggregated series to obtain daily totals.
      - For each day, use the corresponding daily profile (matched by month, weekday, holiday)
        to disaggregate the daily total into finer time intervals.
    """
    daily_totals = aggregated_series.resample('D').sum()
    disagg_list = []
    for d, total in daily_totals.iteritems():
        # Extract day attributes.
        month = d.month
        weekday = d.strftime("%A")
        pt_hols = holidays.country_holidays('PT', years=[d.year])
        holiday = 1 if d.date() in pt_hols else 0

        # Get the typical profile for that day from the profile DataFrame.
        profile = daily_profile[(daily_profile['month'] == month) &
                                (daily_profile['weekday'] == weekday) &
                                (daily_profile['holiday'] == holiday)]
        if profile.empty:
            # If no exact match, fallback to matching by month and weekday.
            profile = daily_profile[(daily_profile['month'] == month) &
                                    (daily_profile['weekday'] == weekday)]
        # Create a full day's time index at the target frequency.
        times = pd.date_range(start=d, periods=int(24 * 60 / (pd.Timedelta(target_frequency).seconds / 60)),
                              freq=target_frequency)
        # Reindex the profile to these times.
        # Assume 'time' in profile is of type datetime.time.
        profile = profile.set_index('time').sort_index()
        times_only = [t.time() for t in times]
        profile_interp = profile.reindex(times_only).interpolate(method='time')
        # Normalize fractions.
        fractions = profile_interp['fraction']
        fractions = fractions / fractions.sum()
        day_values = fractions * total
        day_series = pd.Series(day_values.values, index=times)
        disagg_list.append(day_series)
    return pd.concat(disagg_list)


# ------------------------------------------------
# UNIFIED AGGREGATED FORECAST FUNCTION
# ------------------------------------------------

def unified_hydro_forecast_aggregated(redes_data_path, climate_data_path, final_year=2045,
                                      output_path="forecast_hydro", target_frequency="15min"):
    """
    This function performs the following:
      1. For each plant: predicts production using plant-specific climate data.
      2. Aggregates production from all plants (per timestamp, at the native 3-hour resolution).
      3. Resamples the aggregated series to daily totals.
      4. Disaggregates daily totals to a high-resolution forecast (e.g. 15-minute)
         by overlaying the typical daily profile from historical redes data.
    Forecast outputs are saved only if they do not already exist.
    """
    os.makedirs(output_path, exist_ok=True)
    # Output files.
    agg_forecast_file = os.path.join(output_path, f'Aggregated_Hydro_Forecast_{target_frequency}.xlsx')
    era5_file = os.path.join(output_path, f"Hydro_era5_w_predictions_{target_frequency}_aggregated.xlsx")
    for f in [agg_forecast_file, era5_file]:
        if os.path.exists(f):
            logger.info(f"Forecast file already exists: {f}")
            return

    market_share = 0.6199
    logger.info("Starting aggregated hydro forecast...")

    # 1. Load and preprocess historical countrywide production (redes) data.
    df_actual, df_clipped = load_and_preprocess_data(redes_data_path, target_column="Hydro (kWh)",
                                                     upper_clip=0.9, lower_clip=0.005)
    df_actual["Hydro (kWh)"] = pd.to_numeric(df_actual["Hydro (kWh)"], errors='coerce') * market_share
    df_clipped["Hydro (kWh)"] = pd.to_numeric(df_clipped["Hydro (kWh)"], errors='coerce') * market_share

    # -------------------------
    # Create ERA5 output (historical period) if needed.
    # -------------------------
    os.makedirs(os.path.dirname(era5_file), exist_ok=True)
    if not os.path.exists(era5_file):
        era5_mask = (df_actual.index >= "2020-01-01") & (df_actual.index <= "2025-03-31")
        df_era5 = df_actual.loc[era5_mask].copy()
        df_era5["power_kWh"] = df_era5["Hydro (kWh)"]
        df_era5["Year"] = df_era5.index.year
        df_era5["Month"] = df_era5.index.month
        df_era5["Day"] = df_era5.index.day
        df_era5["Hour"] = df_era5.index.hour
        df_era5["LocalTime"] = df_era5.index.tz_localize(None)
        df_era5 = df_era5[["LocalTime", "power_kWh", "Year", "Month", "Day", "Hour"]]
        df_era5.to_excel(era5_file, index=False)
        logger.info(f"ERA5 file created: {era5_file}")
        del df_era5

    # 2. Load future CORDEX climate data (3-hr interval) and merge with countrywide production.
    climate_df = pd.read_parquet(climate_data_path)
    climate_df['LocalTime'] = pd.to_datetime(climate_df['LocalTime'])
    climate_df = climate_df.set_index('LocalTime')
    df_actual = df_actual.merge(climate_df, left_index=True, right_index=True, how="left")
    df_clipped = df_clipped.merge(climate_df, left_index=True, right_index=True, how="left")

    # 3. Predict production for each plant using the plant-specific climate model.
    # (Assumes that df_actual contains plant-specific records from historical data)
    plant_predictions = process_hydro_outputs(df_actual, interval="3H")  # using 3-hour resolution (native CORDEX)

    # 4. Aggregate production over all plants at each timestamp.
    aggregated_series = aggregate_plants(plant_predictions)
    logger.info("Aggregated plant-level production computed.")

    # 5. Disaggregate the daily totals into high resolution using typical daily profiles.
    # Compute daily profile from the historical (redes) data at the target resolution.
    hist_daily_profile = build_daily_profile(df_clipped, "Hydro (kWh)", target_frequency)
    final_forecast = disaggregate_daily_series(aggregated_series, hist_daily_profile, target_frequency=target_frequency)

    # Save the final aggregated/disaggregated forecast.
    final_forecast.to_excel(agg_forecast_file, index=True)
    logger.info(f"Aggregated Hydro Forecast saved at {agg_forecast_file}")
    return final_forecast