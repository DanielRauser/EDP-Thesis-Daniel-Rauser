import os
import glob
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from Functions.Models.Plants.helpers import aggregate_power_output_to_excel, interpolate_series

logger = logging.getLogger(__name__)

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
    Defaults to parameters for "hydro - run-of-the-river" if type is not found.
    """
    return HYDRO_PARAMETERS.get(plant_type.lower(), HYDRO_PARAMETERS["hydro - run-of-the-river"])

def predict_interval_hydro_output(tp, t2m, rated_capacity, plant_type, avg_annual_prod_GWh,
                                  storage=0, interval_hours=0.25, season="spring", seasonal_factors=None,
                                  lag_factor=1.0, meta_cf=None, tp_mean=1.0, dynamic_factor=0.1):
    """
    Predict energy output (in kWh) for one time interval with:
      - Enhanced precipitation adjustment using a non-linear transformation.
      - Dynamic clamping of the capacity factor.
      - Seasonality applied via multipliers (which can be recalibrated).
    Parameters:
      tp: current precipitation value (e.g., in a normalized unit).
      t2m: temperature (in Kelvin).
      rated_capacity: plant's rated capacity in MW.
      plant_type: plant type (pumped-storage, water-pumped-storage, small-hydro, run-of-the-river).
      avg_annual_prod_GWh: historical average annual production in GWh.
      storage: storage value (if applicable).
      interval_hours: duration of the interval in hours.
      season: season name.
      seasonal_factors: dictionary with seasonal multipliers.
      lag_factor: adjustment factor (for storage/pumped scenarios).
      meta_cf: capacity factor from metadata if available.
      tp_mean: long-term average precipitation (set to 1.0 as default; calibrate with local data).
      dynamic_factor: factor to relax the cap if precipitation exceeds the mean.
    """
    params = get_hydro_parameters(plant_type)
    base_cf = meta_cf if meta_cf is not None else params["base_cf"]

    # Convert temperature from Kelvin to Celsius.
    t2m_c = t2m - 273.15

    # Use provided seasonal multipliers or default ones.
    if seasonal_factors is None:
        seasonal_factors = {"spring": 1.0, "summer": 0.85, "autumn": 0.9, "winter": 1.15}
    seasonal_multiplier = seasonal_factors.get(season.lower(), 1.0)

    # Compute a precipitation anomaly and apply a tanh non-linear transformation.
    precip_anomaly = tp - tp_mean
    precip_adjustment = np.tanh(precip_anomaly)
    precip_factor = 1 + params["precip_sensitivity"] * precip_adjustment

    # Temperature factor adjustment.
    temp_factor = 1 - params["temp_sensitivity"] * ((t2m_c - 15.0) / 15.0)

    if params["curve"] == "constant":
        cf = base_cf * seasonal_multiplier
    else:
        cf = base_cf * precip_factor * temp_factor * seasonal_multiplier
        # Additional adjustments for water-storage plants.
        if plant_type.lower() == "hydro - water-storage" and storage:
            cf *= (1 + 0.03 * ((storage - 50.0) / 50.0))
        # Include lag factor for storage/pumped-storage types.
        if plant_type.lower() in ["hydro - water-storage", "hydro - water-pumped-storage"]:
            cf *= lag_factor

    # Compute the historical capacity factor limit.
    historical_cf_limit = (avg_annual_prod_GWh * 1000) / (rated_capacity * 8760)
    if meta_cf is not None:
        historical_cf_limit = max(historical_cf_limit, meta_cf)
    # Apply a dynamic multiplier if precipitation exceeds the long-term mean.
    if tp > tp_mean:
        dynamic_multiplier = 1 + dynamic_factor * (tp - tp_mean) / tp_mean
    else:
        dynamic_multiplier = 1.0
    dynamic_cf_limit = historical_cf_limit * dynamic_multiplier

    # Clamp capacity factor between 0 and the dynamic cap.
    cf = min(max(cf, 0), dynamic_cf_limit)
    energy_kWh = rated_capacity * cf * interval_hours * 1000
    return energy_kWh

def process_hydro_outputs(df, interval="15min"):
    """
    Process hydro plant data: reindex/interpolate if needed, derive season, compute lag features,
    and compute predicted energy output for each interval using the updated predict_interval_hydro_output.
    """
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])
    df['LocalTime'] = df['LocalTime'].dt.tz_localize(None)
    df.sort_values('LocalTime', inplace=True)

    # Create a unique plant identifier.
    df["ID"] = df["Plant Name"].astype(str) + "_" + df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
    df.rename(columns={"Type 2": "Plant Type"}, inplace=True)

    df = df[df["Sold"]=="n"] # Filter on the plants that still belong to EDP

    plant_ids = df['ID'].unique()
    interval_hours = pd.to_timedelta(interval).total_seconds() / 3600
    results = []

    storage_types = ["hydro - water-storage", "hydro - water-pumped-storage"]

    for pid in tqdm(plant_ids, desc="Processing hydro plants"):
        df_plant = df[df['ID'] == pid].copy()

        # Determine the current data resolution using unique timestamps.
        plant_dt = pd.Series(df_plant['LocalTime'].unique()).diff().dropna().min()
        # Reindex/interpolate if the current resolution is coarser than the desired interval.
        if pd.isna(plant_dt) or plant_dt > pd.Timedelta(interval):
            # Set index and remove duplicate timestamps.
            df_plant.set_index('LocalTime', inplace=True)
            df_plant = df_plant[~df_plant.index.duplicated(keep='first')]
            new_index = pd.date_range(start=df_plant.index.min(), end=df_plant.index.max(), freq=interval)
            old_t_numeric = df_plant.index.astype(np.int64) // 10**9
            new_t_numeric = new_index.astype(np.int64) // 10**9

            # Reindex the dataframe on the new time index.
            df_interp = df_plant.reindex(new_index)
            # Identify climate variables to interpolate.
            cosine_cols = {"t2m"}
            linear_cols = {"tp", "tcc"}
            climate_vars = cosine_cols.union(linear_cols)
            # Forward-fill non-climate (static) columns.
            non_climate_cols = [col for col in df_plant.columns if col not in climate_vars]
            df_interp[non_climate_cols] = df_interp[non_climate_cols].ffill()

            # Interpolate climate variables.
            for col in climate_vars:
                if col in df_plant.columns:
                    orig_values = df_plant[col].values
                    method = "cosine" if col in cosine_cols else "linear"
                    interp_vals = interpolate_series(
                        values=orig_values,
                        orig_times_numeric=old_t_numeric,
                        new_times_numeric=new_t_numeric,
                        method=method
                    )
                    df_interp[col] = interp_vals

            df_interp = df_interp.reset_index().rename(columns={'index': 'LocalTime'})
            df_interp['ID'] = pid
            df_plant = df_interp
        else:
            # If the data is already at the desired (or finer) resolution, use it as is.
            df_plant = df_plant.copy()

        # Derive season from the month.
        df_plant['season'] = df_plant['LocalTime'].dt.month.apply(
            lambda m: "winter" if m in [12, 1, 2]
            else "spring" if m in [3, 4, 5]
            else "summer" if m in [6, 7, 8]
            else "autumn"
        )

        # Compute lag features for storage/pumped-storage plants.
        plant_meta = df[df['ID'] == pid].iloc[0]
        plant_type = plant_meta['Plant Type'].lower()
        if plant_type in storage_types and 'tp' in df_plant.columns:
            window = int((7 * 24) / interval_hours)
            df_plant['lag_tp'] = df_plant['tp'].rolling(window=window, min_periods=1).sum()
            baseline = df_plant['lag_tp'].median()
            df_plant['lag_factor'] = 1 + 0.1 * ((df_plant['lag_tp'] - baseline) / baseline)
        else:
            df_plant['lag_factor'] = 1.0

        # Retrieve static plant parameters.
        rated_capacity = plant_meta['Total Capacity (MW)']
        if 'Average Annual Productivity (GWh)' in plant_meta and not pd.isnull(plant_meta['Average Annual Productivity (GWh)']):
            avg_annual_productivity = plant_meta['Average Annual Productivity (GWh)']
        else:
            avg_annual_productivity = 9999  # effectively no cap

        # Check for metadata capacity factor if provided.
        meta_cf = None
        if 'Capacity Factor' in plant_meta and not pd.isnull(plant_meta['Capacity Factor']):
            meta_cf = plant_meta['Capacity Factor']

        # Compute power output for each row.
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
                tp_mean=0.00014696307,  # Average of 2023-2025
                dynamic_factor=0.1
            )

        df_plant['power_kWh'] = df_plant.apply(compute_output, axis=1)
        results.append(df_plant)

    return pd.concat(results, ignore_index=True)

def generate_hydro_predictions(input_path, output_path, interval="15min"):
    """
    Process all Parquet files in input_path with filenames starting with 'Hydro' and:
      - Reindexes and interpolates data if needed.
      - Derives season and computes lag features.
      - Computes predicted energy output using the enhanced model.
      - Exports both a Parquet file and an aggregated Excel file.
    """
    os.makedirs(output_path, exist_ok=True)
    pattern = os.path.join(input_path, "Hydro*.parquet")
    file_list = sorted(glob.glob(pattern))

    for file in file_list:
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_parquet = os.path.join(output_path, f"{base_name}_w_predictions_{interval}.parquet")
        output_excel = os.path.join(output_path, f"{base_name}_w_predictions_{interval}_aggregated.xlsx")

        if os.path.exists(output_parquet) and os.path.exists(output_excel):
            continue

        df = pd.read_parquet(file, engine="fastparquet")
        results = process_hydro_outputs(df, interval=interval)

        # Annualize the total predicted production.
        start_time = results['LocalTime'].min()
        end_time = results['LocalTime'].max()
        duration_days = (end_time - start_time).days + 1
        duration_years = duration_days / 365.25
        total_predicted = results['power_kWh'].sum()
        annual_predicted = total_predicted / duration_years

        target_lower = 6050e6  # 7000 GWh in kWh
        target_upper = 11000e6  # 8500 GWh in kWh

        if annual_predicted < target_lower:
            scaling_factor = target_lower / annual_predicted
            logger.info(f"Aggregated annual production {annual_predicted:.0f} kWh is below target; scaling up by {scaling_factor:.3f}.")
            results['power_kWh'] *= scaling_factor
        elif annual_predicted > target_upper:
            scaling_factor = target_upper / annual_predicted
            logger.info(f"Aggregated annual production {annual_predicted:.0f} kWh is above target; scaling down by {scaling_factor:.3f}.")
            results['power_kWh'] *= scaling_factor
        else:
            logger.info(f"Aggregated annual production {annual_predicted:.0f} kWh is within target range.")

        results.to_parquet(output_parquet, index=False)
        logger.info(f"Exported: {output_parquet}")

        aggregate_power_output_to_excel(results, output_excel)
        logger.info(f"Exported: {output_excel}")