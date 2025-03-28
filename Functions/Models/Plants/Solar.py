import os
import glob
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from Functions.Models.Plants.helpers import (
    interpolate_series,
    aggregate_power_output_to_excel,
    add_solar_elevation
)

logger = logging.getLogger(__name__)


def generate_solar_predictions(preprocessor, model, input_path, output_path, interval='15min'):
    """
    Processes each photovoltaic file in 'input_path' whose filenames start with 'Photovoltaic'.
    For each file:
      - Processes data plant by plant (grouped by 'ID') with the following steps:
          * If the minimum time difference (dt) in a plant is larger than the desired interval,
            a new time index is created and climate variables are interpolated:
                - For 'ssrd', 't2m', and 'strd': cosine interpolation via interpolate_series.
                - For 'tcc' and 'tp': linear interpolation via interpolate_series.
            Extra (static) columns are filled with the first value.
          * Otherwise, the plant’s data is reindexed to the desired interval and climate variables
            are interpolated with the previously defined method (cosine for 'ssrd', 't2m', 'strd',
            and linear for 'tcc' and 'tp').
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

    for file in file_list:
        base_filename = os.path.basename(file).replace('.parquet', '')
        output_parquet = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}.parquet")
        output_excel = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}_aggregated.xlsx")

        # Check if both output files already exist
        if os.path.exists(output_parquet) and os.path.exists(output_excel):
            continue

        # Read the individual photovoltaic file.
        df = pd.read_parquet(file, engine="fastparquet")
        df['LocalTime'] = pd.to_datetime(df['LocalTime'])
        df.sort_values('LocalTime', inplace=True)
        df["ID"] = df["Plant Name"].astype(str) + "_" + df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

        # Convert interval to timedelta and compute hours.
        interval_timedelta = pd.Timedelta(interval)
        interval_hours = interval_timedelta.total_seconds() / 3600

        # Process each solar plant in the current file.
        plant_ids = df['ID'].unique()
        plant_results = []
        for pid in tqdm(plant_ids, desc=f"Processing plants in {os.path.basename(file)}"):
            df_plant = df[df['ID'] == pid].copy()
            # Compute the minimum time difference in this plant's data.
            plant_dt = pd.Series(df_plant['LocalTime'].unique()).diff().dropna().min()

            # Reindex/interpolate if the current resolution is coarser than the desired interval.
            if pd.isna(plant_dt) or plant_dt > pd.Timedelta(interval):
                # Set index and remove duplicate timestamps.
                df_plant.set_index('LocalTime', inplace=True)
                df_plant = df_plant[~df_plant.index.duplicated(keep='first')]
                new_index = pd.date_range(start=df_plant.index.min(), end=df_plant.index.max(), freq=interval)
                old_t_numeric = df_plant.index.astype(np.int64) // 10 ** 9
                new_t_numeric = new_index.astype(np.int64) // 10 ** 9

                # Reindex the dataframe on the new time index.
                df_interp = df_plant.reindex(new_index)
                cosine_cols = {'ssrd', 't2m', 'strd'}
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

                # Check if 'LocalTime' is timezone-aware. If it is, use tz_convert; if not, use tz_localize.
                if df_interp['LocalTime'].dt.tz is None:
                    df_interp['time'] = pd.to_datetime(df_interp['LocalTime']).dt.tz_localize('UTC')
                else:
                    df_interp['time'] = df_interp['LocalTime'].dt.tz_convert('UTC')

                df_plant = df_interp
            else:
                # If the data is already at the desired (or finer) resolution, use it as is.
                df_plant = df_plant.copy()

            df_plant["net_radiation"] = df_plant["ssrd"] - df_plant["strd"]
            df_plant = add_solar_elevation(df_plant)

            # --- Prediction Pipeline for this Plant ---
            transformed_data = preprocessor.transform(df_plant)
            raw_predictions = model.predict(transformed_data)
            actual_predictions = preprocessor.inverse_transform_predictions(raw_predictions, df_plant)
            df_plant["Power(MW)"] = actual_predictions

            # Set power to zero when solar elevation is below horizon and negative.
            df_plant["Power(MW)"] = np.where(df_plant["solar_elevation"] < 0, 0, df_plant["Power(MW)"])
            df_plant["Power(MW)"] = np.where(df_plant["Power(MW)"] < 0, 0, df_plant["Power(MW)"])

            # Compute energy (kWh) for each interval.
            df_plant["power_kWh"] = df_plant["Power(MW)"] * 1000 * interval_hours

            plant_results.append(df_plant)

        # Combine predictions from all plants in the current file.
        df_all = pd.concat(plant_results, ignore_index=True)

        # --- Export the Enriched DataFrame as Parquet ---
        df_all.to_parquet(output_parquet, index=False)
        logger.info(f"Exported: {output_parquet}")

        # --- Export Aggregated Power Outputs as Excel ---
        df_all_agg = df_all.copy()
        df_all_agg['LocalTime'] = df_all_agg['LocalTime'].dt.tz_localize(None)
        if pd.Timedelta(interval) % pd.Timedelta(hours=1) == pd.Timedelta(0):
            df_all_agg['LocalTime'] = pd.to_datetime(df_all_agg['LocalTime']).dt.floor(interval)
        aggregate_power_output_to_excel(df_all_agg, output_excel)
        logger.info(f"Exported: {output_excel}")