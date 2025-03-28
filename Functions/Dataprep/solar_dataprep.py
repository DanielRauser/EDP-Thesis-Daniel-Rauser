import os
import logging
import pandas as pd
import xarray as xr
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from Functions.Dataprep.helpers import (cache_timezones, vectorized_localize,
                                        compute_grid_mapping, compute_clear_sky_index,
                                        compute_elevation, concatenate_parquet_files,
                                        reduce_memory_usage, add_solar_elevation)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _extract_metadata(filename):
    """
    Extract metadata from a filename following the pattern:
      DataType_Latitude_Longitude_WeatherYear_PVType_CapacityMW_TimeInterval_Min.csv

    Parameters:
        filename (str): The filename to parse, e.g.,
                        "Actual_38.05_-75.45_2006_UPV_61MW_5_Min.csv".

    Returns:
        dict: A dictionary containing:
            - 'latitude' (float): Latitude value.
            - 'longitude' (float): Longitude value.
            - 'year' (int): Weather year.
            - 'capacity_MW' (float): Capacity in megawatts.

    Raises:
        ValueError: If the filename does not conform to the expected pattern or parsing fails.
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    if len(parts) < 7:
        raise ValueError(f"Filename {filename} does not conform to expected pattern.")
    try:
        return {
            "latitude": float(parts[1]),
            "longitude": float(parts[2]),
            "year": int(parts[3]),
            "capacity_MW": float(parts[5].replace("MW", ""))
        }
    except Exception as e:
        raise ValueError(f"Error parsing filename {filename}: {e}")


def _get_all_files(input_path):
    """
    Retrieve all CSV file paths in a directory tree that start with 'Actual_'
    and contain '_UPV_' (Utility-scale PV) in the filename.

    Parameters:
        input_path (str): The root directory path to search.

    Returns:
        list: A list of file paths matching the criteria.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(input_path)
        for file in files
        if file.startswith("Actual_") and "_UPV_" in file and file.endswith(".csv")
    ]


def _process_file(file_path):
    """
    Process an individual CSV file: read the file, convert the 'LocalTime' column to datetime,
    and append metadata extracted from the filename.

    Parameters:
        file_path (str): The full path of the CSV file.

    Returns:
        pandas.DataFrame or None: The processed DataFrame with additional metadata columns,
                                  or None if an error occurred.
    """
    try:
        metadata = _extract_metadata(os.path.basename(file_path))
        df = pd.read_csv(file_path)
        df['LocalTime'] = pd.to_datetime(df['LocalTime'], format="%m/%d/%y %H:%M")
        for key, value in metadata.items():
            df[key] = value
        return df
    except Exception as e:
        logger.error(f"Skipping {file_path} due to error: {e}")
        return None


def _map_era5_to_solar_vectorized_with_interp(batch_df, era5_ds):
    """
    Vectorized mapping of solar records to ERA5 data with variable‐specific interpolation.

    For variables that follow a smooth diurnal cycle (e.g. 2m temperature, surface solar
    and thermal radiation), a cosine-based interpolation is applied. For cloud cover, precipitation,
    and snow depth, linear interpolation is used (note: for precipitation, more advanced methods
    such as stochastic disaggregation might be desirable).

    Assumes ERA5 dataset variables have dimensions (time, latitude, longitude).
    """
    # Ensure time arrays are in minutes for consistency.
    era5_times = era5_ds['time'].values.astype('datetime64[m]')

    # Merge in precomputed grid mapping (adds 'lat_idx' and 'lon_idx' columns)
    mapping_df = compute_grid_mapping(batch_df, era5_ds)
    batch_df = batch_df.merge(mapping_df, on=['latitude', 'longitude'], how='left')

    # Convert solar UTC times to the same minute resolution.
    solar_times = batch_df['UTC'].values.astype('datetime64[m]')

    num_rows = batch_df.shape[0]
    # Preallocate output arrays for each variable.
    rsds_interp = np.full(num_rows, np.nan)
    rlds_interp = np.full(num_rows, np.nan)
    clt_interp = np.full(num_rows, np.nan)
    tas_interp = np.full(num_rows, np.nan)
    pr_interp = np.full(num_rows, np.nan)
    snd_interp = np.full(num_rows, np.nan)

    # Identify records that fall within the ERA5 time range.
    valid_mask = (solar_times >= era5_times[0]) & (solar_times <= era5_times[-1])
    valid_idx = np.where(valid_mask)[0]
    solar_valid = solar_times[valid_idx]

    # Find ERA5 indices surrounding each valid solar time.
    # side='right' ensures that if solar_time exactly equals an ERA5 time, it acts as the upper bound.
    time_idx_upper = np.searchsorted(era5_times, solar_valid, side='right')
    time_idx_lower = time_idx_upper - 1

    # For solar times equal to the last ERA5 timestamp, force indices to be equal.
    at_end = time_idx_upper == len(era5_times)
    time_idx_upper[at_end] = time_idx_lower[at_end]

    # Compute the time difference between lower and upper indices (in minutes)
    delta = (era5_times[time_idx_upper] - era5_times[time_idx_lower]).astype('timedelta64[m]').astype(float)

    # Calculate fractional position for interpolation; avoid division by zero.
    frac = np.zeros_like(delta)
    nonzero = delta != 0
    frac[nonzero] = ((solar_valid - era5_times[time_idx_lower]).astype('timedelta64[m]').astype(float))[nonzero] / \
                    delta[nonzero]

    # Get grid indices for the valid records.
    lat_idx = batch_df.loc[valid_idx, 'lat_idx'].values.astype(int)
    lon_idx = batch_df.loc[valid_idx, 'lon_idx'].values.astype(int)

    batch_df["identifier"] = batch_df["latitude"].astype(str) + "_" + batch_df["longitude"].astype(str)

    batch_df = batch_df.groupby('identifier', group_keys=False).apply(compute_elevation)

    # Helper to perform variable-specific interpolation.
    def interpolate_var(var):
        data = era5_ds[var].values  # shape: (time, latitude, longitude)
        lower_vals = data[time_idx_lower, lat_idx, lon_idx]
        upper_vals = data[time_idx_upper, lat_idx, lon_idx]

        # For smooth diurnal variables, use cosine-based interpolation.
        if var in ['rsds', 'tas', 'rlds']:
            # Cosine weighting: forces a sinusoidal transition between hourly values.
            weight = (1 - np.cos(np.pi * frac)) / 2
            return lower_vals + (upper_vals - lower_vals) * weight
        # For cloud cover, precipitation, and snow depth, use linear interpolation.
        elif var in ['clt', 'pr', 'snd']:
            return lower_vals + (upper_vals - lower_vals) * frac
        else:
            # Default to linear interpolation if unspecified.
            return lower_vals + (upper_vals - lower_vals) * frac

    # Apply interpolation for each ERA5 variable.
    rsds_interp[valid_idx] = interpolate_var('rsds')
    rlds_interp[valid_idx] = interpolate_var('rlds')
    tas_interp[valid_idx] = interpolate_var('tas')
    clt_interp[valid_idx] = interpolate_var('clt')
    pr_interp[valid_idx] = interpolate_var('pr')
    snd_interp[valid_idx] = interpolate_var('snd')

    # Assign the interpolated values to new DataFrame columns.
    batch_df["ssrd"] = rsds_interp
    batch_df["strd"] = rlds_interp
    batch_df["tcc"] = clt_interp
    batch_df["t2m"] = tas_interp
    batch_df["tp"] = pr_interp
    batch_df["snd"] = snd_interp

    # Eliminate observations with snow impact
    batch_df = batch_df[batch_df["snd"]==0]

    batch_df = batch_df.groupby('identifier', group_keys=False).apply(compute_clear_sky_index)

    batch_df['hour'] = batch_df['LocalTime'].dt.hour
    batch_df['hour_sin'] = np.sin(2 * np.pi * batch_df['hour'] / 24)
    batch_df['hour_cos'] = np.cos(2 * np.pi * batch_df['hour'] / 24)

    batch_df['day_of_year'] = batch_df['LocalTime'].dt.dayofyear
    batch_df['doy_sin'] = np.sin(2 * np.pi * batch_df['day_of_year'] / 365)
    batch_df['doy_cos'] = np.cos(2 * np.pi * batch_df['day_of_year'] / 365)

    batch_df['net_radiation'] = batch_df['ssrd'] - batch_df['strd']

    batch_df = add_solar_elevation(batch_df)

    # keeping some negatives to let the model learn the transition phase
    batch_df = batch_df[batch_df["solar_elevation"]>-5]

    batch_df = reduce_memory_usage(batch_df, exclude_cols=["tp"])

    # Remove temporary grid mapping columns.
    batch_df.drop(columns=['lat_idx', 'lon_idx'], inplace=True)

    return batch_df


def process_batch(file_list, batch_number, era5_ds=None):
    """
    Process a batch of CSV files: read, parse metadata, and optionally apply timezone localization
    and ERA5 data mapping.

    Parameters:
        file_list (list): List of CSV file paths to process.
        batch_number (int): Identifier for the current batch (used for logging).
        era5_ds (xarray.Dataset, optional): The ERA5 dataset for mapping. Defaults to None.

    Returns:
        pandas.DataFrame or None: The concatenated DataFrame for the batch, or None if no valid data was processed.
    """
    dfs = []
    with ProcessPoolExecutor() as executor:
        for df in executor.map(_process_file, file_list):
            if df is not None:
                dfs.append(df)

    if not dfs:
        return None

    batch_df = pd.concat(dfs, ignore_index=True)
    if era5_ds is not None:
        logger.info(f"Applying timezone localization for batch {batch_number}...")
        tz_cache = cache_timezones(batch_df)
        batch_df = vectorized_localize(batch_df, tz_cache)
        batch_df = batch_df[(batch_df['LocalTime'].dt.hour >= 6) & (batch_df['LocalTime'].dt.hour < 21)]
        logger.info(f"Mapping ERA5 data for batch {batch_number}...")
        batch_df = _map_era5_to_solar_vectorized_with_interp(batch_df, era5_ds)
    return batch_df


def prepare_solar_train_data(input_path, output_path, era5_file=None, batch_size=100, temp_dir="temp_batches"):
    """
    Prepare and combine solar training data by processing multiple CSV files in batches.
    The process includes reading CSV files, extracting metadata, applying timezone localization,
    and optionally mapping ERA5 climate data. Instead of loading all batches into memory at once,
    this implementation writes each batch to a temporary parquet file and then streams them together
    into a single parquet file using PyArrow's ParquetWriter.

    Parameters:
        input_path (str): Root directory containing solar CSV files.
        output_path (str): Directory where the final processed data and intermediate outputs will be saved.
        era5_file (str, optional): Filename of the ERA5 dataset (expected under a 'Climate Data' folder in output_path).
                                   Defaults to None.
        batch_size (int, optional): Number of files to process per batch. Defaults to 100.
        temp_dir (str, optional): Temporary directory to store batch outputs. Defaults to "temp_batches".
    """
    os.makedirs(output_path, exist_ok=True)
    solar_data_dir = os.path.join(output_path, "Solar Data")
    temp_data_dir = os.path.join(output_path, temp_dir)
    os.makedirs(solar_data_dir, exist_ok=True)
    os.makedirs(temp_data_dir, exist_ok=True)

    output_file = os.path.join(solar_data_dir, "solar_data.parquet")
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists, skipping processing.")
        return pd.read_parquet(output_file, engine='fastparquet')

    logger.info("Preparing solar train data...")
    file_paths = _get_all_files(input_path)  # Assumes _get_all_files is defined elsewhere.
    total_files = len(file_paths)
    logger.info(f"Found {total_files} matching files.")

    # Split file paths into batches
    batches = [file_paths[i:i + batch_size] for i in range(0, total_files, batch_size)]

    era5_ds = None
    if era5_file:
        era5_path = os.path.join(output_path, "Climate Data", era5_file)
        logger.info(f"Loading ERA5 dataset from {era5_path}")
        era5_ds = xr.open_dataset(era5_path)

    # Process each batch and write to temporary parquet files
    for batch_number, batch_files in enumerate(batches, start=1):
        batch_file_path = os.path.join(temp_data_dir, f"batch_{batch_number}.parquet")
        if os.path.exists(batch_file_path):
            logger.info(f"Batch file {batch_file_path} already exists.")
        else:
            logger.info(f"Processing batch {batch_number} with {len(batch_files)} files...")
            batch_df = process_batch(batch_files, batch_number, era5_ds)
            if batch_df is not None:
                batch_df.to_parquet(batch_file_path, index=False)
                logger.info(f"Saved batch {batch_number} with {len(batch_df)} rows to {batch_file_path}.")
            else:
                logger.warning(f"No data processed for batch {batch_number}.")

    if era5_ds is not None:
        era5_ds.close()

    # Concatenate the batch parquet files in a memory-efficient way
    concatenate_parquet_files(temp_data_dir, output_file)

    try:
        shutil.rmtree(temp_data_dir)
    except Exception as e:
        logger.warning(f"Failed to remove temp directory {temp_data_dir}: {e}")