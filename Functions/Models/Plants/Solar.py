import os
import glob
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from Functions.Models.Plants.helpers import (
    interpolate_variable_series,
    aggregate_power_output_to_excel,
    add_solar_elevation,
    compute_clear_sky_index,
    compute_minutes_sunrise_sunset_from_elevation
)

logger = logging.getLogger(__name__)


def generate_solar_predictions(preprocessor, model, input_path, output_path, interval='15min'):
    """
    Processes each photovoltaic file in 'input_path' whose filenames start with 'Photovoltaic'.
    For each file:
      - Processes data plant by plant (grouped by 'ID') with the following steps:
          * If the minimum time difference in a plant is coarser than the desired interval,
            a new time index is created and each climate variable is interpolated individually:
              - For 'ssrd', 't2m', and 'strd': cosine-based interpolation.
              - For 'tp' and 'tcc': linear interpolation.
            Extra (static) columns are reindexed and forward–filled.
          * Otherwise, the plant’s data is used as is.
          * Derived features (net radiation and solar elevation) are recalculated.
          * The data is transformed with the preprocessor, the model predicts power,
            and the transformation is inverted.
          * Power is set to zero when solar elevation is below the horizon and negatives are prevented.
          * Energy (in kWh) is computed for each interval.
      - Exports the enriched predictions for that file as a parquet file and an aggregated Excel file.
    """
    os.makedirs(output_path, exist_ok=True)
    pattern = os.path.join(input_path, "Photovoltaic*.parquet")
    file_list = sorted(glob.glob(pattern))

    # Define which columns are climate variables and which interpolation method to use.
    # (Change these names as needed based on your dataset.)
    cosine_cols = {'ssrd', 't2m', 'strd'}  # smooth, diurnal variables
    linear_cols = {"tp", "tcc"}             # variables that use linear interpolation
    climate_vars = cosine_cols.union(linear_cols)

    for file in file_list:
        base_filename = os.path.basename(file).replace('.parquet', '')
        output_parquet = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}.parquet")
        output_excel = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}_aggregated.xlsx")

        # Skip if output files already exist.
        if os.path.exists(output_parquet) and os.path.exists(output_excel):
            continue

        # Read the photovoltaic file.
        df = pd.read_parquet(file, engine="fastparquet")
        df['LocalTime'] = pd.to_datetime(df['LocalTime'])
        df.sort_values('LocalTime', inplace=True)
        # Create an identifier for each plant.
        df["ID"] = df["Plant Name"].astype(str) + "_" + df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

        interval_timedelta = pd.Timedelta(interval)
        interval_hours = interval_timedelta.total_seconds() / 3600

        plant_ids = df['ID'].unique()
        plant_results = []
        for pid in tqdm(plant_ids, desc=f"Processing plants in {os.path.basename(file)}"):
            df_plant = df[df['ID'] == pid].copy()
            # Compute the minimum time difference in this plant’s data.
            plant_dt = pd.Series(df_plant['LocalTime'].unique()).diff().dropna().min()

            if pd.isna(plant_dt) or plant_dt > pd.Timedelta(interval):
                # Use the original plant data (with its native timestamps)
                df_orig = df_plant.copy()
                df_orig.set_index('LocalTime', inplace=True)
                # Make sure the Activated column is timezone-aware.
                df_orig['Activated'] = pd.to_datetime(df_orig['Activated']).dt.tz_localize('UTC')
                df_orig = df_orig[~df_orig.index.duplicated(keep='first')]

                # Create a new regular time index.
                new_index = pd.date_range(start=df_orig.index.min(), end=df_orig.index.max(), freq=interval)
                # Prepare a new DataFrame with the new time index.
                df_interp = pd.DataFrame(index=new_index)

                # For non-climate (static) columns, reindex and forward–fill.
                non_climate_cols = [col for col in df_orig.columns if col not in climate_vars]
                df_interp[non_climate_cols] = df_orig[non_climate_cols].reindex(new_index).ffill()

                # For each climate variable, perform individual interpolation.
                for col in climate_vars:
                    if col in df_orig.columns:
                        series = df_orig[col].dropna()  # Get available data points for this variable.
                        if series.empty:
                            df_interp[col] = np.nan
                        else:
                            method = 'cosine' if col in cosine_cols else 'linear'
                            interpolated = interpolate_variable_series(series.index.values, series.values,
                                                                       new_index.values, method=method)
                            df_interp[col] = interpolated

                # Reset index to turn the time index into a column.
                df_interp = df_interp.reset_index().rename(columns={'index': 'LocalTime'})
                # Ensure proper timezone handling.
                if df_interp['LocalTime'].dt.tz is None:
                    df_interp['time'] = pd.to_datetime(df_interp['LocalTime']).dt.tz_localize('UTC')
                else:
                    df_interp['time'] = df_interp['LocalTime'].dt.tz_convert('UTC')
                df_plant = df_interp
            else:
                # Data is already at or finer than the target resolution.
                df_plant = df_plant.copy()

            # --- Derived Features ---
            df_plant["net_radiation"] = df_plant["ssrd"] - df_plant["strd"]
            df_plant['hour_sin'] = np.sin(2 * np.pi * df_plant['hour'] / 24)
            df_plant['hour_cos'] = np.cos(2 * np.pi * df_plant['hour'] / 24)
            df_plant = add_solar_elevation(df_plant)

            df_plant = df_plant.groupby('identifier', group_keys=False).apply(compute_clear_sky_index)

            df_plant = df_plant.groupby(['identifier', df_plant['LocalTime'].dt.date], group_keys=False) \
                .apply(compute_minutes_sunrise_sunset_from_elevation)

            # --- Prediction Pipeline for this Plant ---
            transformed_data = preprocessor.transform(df_plant)
            raw_predictions = model.predict(transformed_data)
            actual_predictions = preprocessor.inverse_transform_predictions(raw_predictions, df_plant)
            df_plant["Power(MW)"] = actual_predictions

            # Set power to zero when solar elevation is below the horizon and ensure no negatives.
            df_plant["Power(MW)"] = np.where(df_plant["solar_elevation"] < 0, 0, df_plant["Power(MW)"])
            df_plant["Power(MW)"] = np.where(df_plant["Power(MW)"] < 0, 0, df_plant["Power(MW)"])

            #df_plant = apply_morning_and_evening_ramps(df_plant, threshold=10)

            # Compute energy (kWh) for each interval.
            df_plant["power_kWh"] = df_plant["Power(MW)"] * 1000 * interval_hours
            df_plant["power_kWh"] = df_plant.apply(
                lambda x: 0 if x["Activated"] > x["LocalTime"] else x["power_kWh"], axis=1
            )

            plant_results.append(df_plant)

        # Combine predictions from all plants in the current file.
        df_all = pd.concat(plant_results, ignore_index=True)

        # --- Export as Parquet ---
        df_all.to_parquet(output_parquet, index=False)
        logger.info(f"Exported: {output_parquet}")

        # --- Aggregated Excel Export ---
        df_all_agg = df_all.copy()
        df_all_agg['LocalTime'] = df_all_agg['LocalTime'].dt.tz_localize(None)
        if pd.Timedelta(interval) % pd.Timedelta(hours=1) == pd.Timedelta(0):
            df_all_agg['LocalTime'] = pd.to_datetime(df_all_agg['LocalTime']).dt.floor(interval)
        aggregate_power_output_to_excel(df_all_agg, output_excel)
        logger.info(f"Exported: {output_excel}")