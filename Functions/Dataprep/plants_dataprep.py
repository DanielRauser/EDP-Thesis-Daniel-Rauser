import os
import logging
import re
import pandas as pd
import numpy as np
import xarray as xr
import pytz
from timezonefinder import TimezoneFinder
from Functions.Dataprep.helpers import (
    compute_grid_mapping,
    compute_elevation,
    compute_clear_sky_index,
    add_solar_elevation
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tf = TimezoneFinder()

def build_timezone_cache(df):
    """
    Build a cache of timezone objects keyed by (latitude, longitude)
    for all unique coordinates in the DataFrame.
    """
    unique_coords = df[['latitude', 'longitude']].drop_duplicates()
    tz_cache = {}
    for _, row in unique_coords.iterrows():
        lat, lon = row['latitude'], row['longitude']
        tz_str = tf.timezone_at(lng=lon, lat=lat)
        if tz_str is None:
            # Fallback to Lisbon if timezone not found.
            tz_str = 'Europe/Lisbon'
        tz_cache[(lat, lon)] = pytz.timezone(tz_str)
    return tz_cache


def _map_ds_to_plants(plant_df, ds):
    """
    Map each plant's location to the nearest grid cell in the dataset (ERA5 or CORDEX)
    and extract the full time series for that grid cell.
    """
    # Determine the time coordinate: 'valid_time' (ERA5) or 'time' (CORDEX)
    time_coord = ds.get("valid_time", ds.get("time", None))
    if time_coord is None:
        raise KeyError("Dataset must contain either 'valid_time' or 'time' coordinate.")
    times = time_coord.values

    # Precompute grid mapping and merge into the plant DataFrame.
    mapping_df = compute_grid_mapping(plant_df, ds)
    plant_df = plant_df.merge(mapping_df, on=["latitude", "longitude"], how="left")

    ds_vars = list(ds.data_vars)
    records = []
    # Loop over each plant and extract the time series at the mapped grid cell.
    for _, row in plant_df.iterrows():
        lat_idx, lon_idx = int(row["lat_idx"]), int(row["lon_idx"])
        for t_idx, t in enumerate(times):
            record = row.to_dict()
            record["time"] = t
            for var in ds_vars:
                try:
                    record[var] = ds[var].values[t_idx, lat_idx, lon_idx]
                except IndexError:
                    record[var] = np.nan
            records.append(record)

    result_df = pd.DataFrame(records)
    result_df.drop(columns=["lat_idx", "lon_idx"], inplace=True)
    if "valid_time" in result_df.columns:
        result_df.rename(columns={"valid_time": "time"}, inplace=True)
    return result_df


def _add_solar_features(df, dataset_type='ERA5'):
    """
    Add solar-specific features (net radiation and time features) to the time series DataFrame.
    For CORDEX, the cloud cover (provided as 'clt' in %) is converted to a fraction.
    """
    # Define variable mappings.
    variable_mappings = {
        'ERA5': {
            'ssrd': ('ssrd', 'J m-2'),
            'strd': ('strd', 'J m-2'),
            'tcc': ('tcc', 'dimensionless'),
            't2m': ('t2m', 'K'),
            'tp': ('tp', 'm')
        },
        'CORDEX': {
            'ssrd': ('rsds', 'W m-2'),
            'strd': ('rlds', 'W m-2'),
            'tcc': ('clt', '%'),
            't2m': ('tas', 'K'),
            'tp': ('pr', 'kg m-2 s-1')
        }
    }
    mapping = variable_mappings.get(dataset_type)
    if mapping is None:
        raise ValueError("Unsupported dataset_type. Choose either 'ERA5' or 'CORDEX'.")

    # Rename columns using the mapping.
    df = df.rename(columns={v[0]: k for k, v in mapping.items()})

    # Determine time step in hours.
    time_step_hours = 3 if dataset_type == 'CORDEX' else 1
    seconds = time_step_hours * 3600
    minutes = seconds // 60

    # Unit conversions.
    for var, (col, unit) in mapping.items():
        if var in df.columns:
            if dataset_type == 'ERA5' and unit == 'J m-2':
                # Convert accumulated energy (J/m²) to instantaneous flux (W/m²).
                df[var] = df[var] / seconds
            elif var == 'tp':
                if dataset_type == 'ERA5' and unit == 'm':
                    # Convert accumulated precipitation (m over 1h) to mm/h.
                    df[var] = df[var] * 1000
                elif dataset_type == 'CORDEX' and unit == 'kg m-2 s-1':
                    # Convert precipitation rate to mm/h.
                    df[var] = df[var] * 1200 # : 3 * 3600
            elif var == 'tcc' and unit == '%':
                df[var] = df[var] / 100  # Convert from % to fraction (0-1)

    # Ensure coordinates exist; create a unique identifier.
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise KeyError("Columns 'latitude' and 'longitude' must be present in the DataFrame.")
    df["identifier"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    # Compute elevation and clear sky index per unique location.
    df = df.groupby("identifier", group_keys=False).apply(compute_elevation)
    df = df.groupby("identifier", group_keys=False).apply(compute_clear_sky_index)

    # Adding the solar elevation
    df = df.reset_index(drop=True)
    df = add_solar_elevation(df)

    # Compute net radiation.
    if "ssrd" in df.columns and "strd" in df.columns:
        df["net_radiation"] = df["ssrd"] - df["strd"]
    else:
        raise KeyError("Columns 'ssrd' and 'strd' must be present to compute net radiation.")

    # Add time features.
    if 'LocalTime' in df.columns:
        df["LocalTime"] = pd.to_datetime(df["LocalTime"])
        df["hour"] = df["LocalTime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_year"] = df["LocalTime"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    else:
        raise KeyError("Column 'time' must be present in the DataFrame.")

    return df


def process_and_save_sheet(file_path, sheet, ds, output_dir, file_suffix=""):
    """
    Process a single Excel sheet: load plant data, map dataset time series to grid cells,
    add solar features (for photovoltaic plants), compute local time, and save the output as a parquet file.
    """
    output_file = os.path.join(output_dir, f"{sheet}{file_suffix}.parquet")
    if os.path.exists(output_file):
        logger.info(f"{output_file} already exists, skipping processing.")
        return
    logger.info(f"Loading {sheet} locations from {file_path}")
    plant_df = pd.read_excel(file_path, sheet_name=sheet)
    plant_df["plant_type"] = sheet

    # Map dataset to plants.
    plant_df_full_ts = _map_ds_to_plants(plant_df, ds)

    # Add LocalTime column.
    # First, assume the 'time' column is in UTC and create a LocalTime column.
    plant_df_full_ts["LocalTime"] = pd.to_datetime(plant_df_full_ts["time"]).dt.tz_localize('UTC')
    # Build a timezone cache for each unique (latitude, longitude) pair.
    tz_cache = build_timezone_cache(plant_df_full_ts)
    # Convert each UTC time to local time using the cached timezone.
    plant_df_full_ts["LocalTime"] = plant_df_full_ts.apply(
        lambda row: row["LocalTime"].astimezone(tz_cache[(row["latitude"], row["longitude"])]),
        axis=1
    )

    if sheet == "Photovoltaic":
        if file_suffix=="_era5":
            plant_df_full_ts = _add_solar_features(plant_df_full_ts,
                                                   dataset_type='ERA5')
        else:
            plant_df_full_ts = _add_solar_features(plant_df_full_ts,
                                                   dataset_type='CORDEX')

    plant_df_full_ts.to_parquet(output_file, index=False)
    logger.info(f"Saved {sheet} data to: {output_file}")
    return output_file


def prepare_plant_era5_data(file_name, sheet_names, era5_file, output_path):
    """
    Prepare ERA5 data by processing each plant sheet using the ERA5 dataset.
    """
    file_path = os.path.join("Data", file_name)
    os.makedirs(output_path, exist_ok=True)
    era5_path = os.path.join(output_path, "Climate Data", era5_file)
    logger.info(f"Loading ERA5 dataset from {era5_path}")

    with xr.open_dataset(era5_path) as era5_ds:
        saved_files = [
            process_and_save_sheet(file_path, sheet, era5_ds, output_path, file_suffix="_era5")
            for sheet in sheet_names
        ]
    return saved_files


def prepare_plant_cordex_data(file_name, sheet_names, output_path):
    """
    Prepare CORDEX data for each plant sheet and for all CORDEX scenario files found.
    For 'Wind', separately process historical (2023–24) and future (2025+) wind NetCDFs,
    then merge their output Parquet files into one per scenario to avoid large in-memory xarray ops.
    Other sheets use the full future NetCDF directly.
    """
    file_path = os.path.join("Data", file_name)
    os.makedirs(output_path, exist_ok=True)

    cordex_pattern = re.compile(r"copernicus_(rcp_\d+_\d+)_portugal_future\.nc")
    climate_data_path = os.path.join(output_path, "Climate Data")
    if not os.path.isdir(climate_data_path):
        logger.warning(f"Climate Data directory not found: {climate_data_path}")
        return []

    all_saved_files = []
    for fname in os.listdir(climate_data_path):
        m = cordex_pattern.match(fname)
        if not m:
            continue
        scenario = m.group(1)
        full_path = os.path.join(climate_data_path, fname)
        wind_hist = os.path.join(
            climate_data_path,
            f"wind_2023_2024_{scenario}.nc"
        )

        for sheet in sheet_names:
            if sheet == "Wind":
                # Process historical wind
                if os.path.isfile(wind_hist):
                    ds_hist = xr.open_dataset(wind_hist)
                    hist_parquet = process_and_save_sheet(
                        file_path,
                        sheet,
                        ds_hist,
                        output_path,
                        file_suffix=f"_{scenario}_hist"
                    )
                    ds_hist.close()
                else:
                    logger.error(f"Missing historical wind file: {wind_hist}")
                    continue
                # Process future wind
                ds_fut = xr.open_dataset(full_path)
                fut_parquet = process_and_save_sheet(
                    file_path,
                    sheet,
                    ds_fut,
                    output_path,
                    file_suffix=f"_{scenario}_fut"
                )
                ds_fut.close()
                # Merge the two Parquet outputs
                if hist_parquet and fut_parquet:
                    relevant_cols = ['LocalTime', 'ID', 'uas', 'vas', 'Total Capacity (MW)', 'Number of Turbines', 'Hub Height (m)', 'Turbine Model', 'longitude', 'latitude']
                    df_hist = pd.read_parquet(hist_parquet)
                    df_hist = df_hist[relevant_cols]
                    df_fut = pd.read_parquet(fut_parquet)
                    df_fut = df_fut[relevant_cols]
                    df_comb = pd.concat([df_hist, df_fut], ignore_index=True)
                    final_file = os.path.join(output_path, f"{sheet}_{scenario}.parquet")
                    df_comb.to_parquet(final_file, index=False)
                    logger.info(f"Merged Wind data saved to: {final_file}")
                    all_saved_files.append(final_file)
            else:
                # Other sheets: use future NetCDF only
                logger.info(f"Loading {sheet} [{scenario}] from {full_path}")
                with xr.open_dataset(full_path) as ds:
                    out = process_and_save_sheet(
                        file_path,
                        sheet,
                        ds,
                        output_path,
                        file_suffix=f"_{scenario}"
                    )
                    if out:
                        all_saved_files.append(out)
    return all_saved_files