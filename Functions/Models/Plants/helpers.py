import os
import glob
import pandas as pd
import numpy as np
import pvlib
import rasterio
import logging

logger = logging.getLogger(__name__)


def get_weibull_params(lon, lat, scale_tif, shape_tif):
    with rasterio.open(scale_tif) as src_scale:
        row, col = src_scale.index(lon, lat)
        c_hr = src_scale.read(1)[row, col]
    with rasterio.open(shape_tif) as src_shape:
        row, col = src_shape.index(lon, lat)
        k_hr = src_shape.read(1)[row, col]
    return c_hr, k_hr

def add_solar_elevation(df):
    # Reset index if duplicates exist
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)

    if 'time' in df.columns:
        time_column = 'time'
    elif 'UTC' in df.columns:
        time_column = 'UTC'
    else:
        raise KeyError("The dataframe must contain 'time' or 'UTC' as a time column.")

    # Ensure unique time values
    if df[time_column].duplicated().any():
        df = df.drop_duplicates(subset=[time_column, 'latitude', 'longitude'])

    def compute_solar(df_group):
        """Compute solar position for each group."""
        solpos = pvlib.solarposition.get_solarposition(
            time=df_group[time_column],
            latitude=df_group['latitude'].iloc[0],
            longitude=df_group['longitude'].iloc[0]
        )
        df_group['solar_elevation'] = solpos['apparent_elevation'].values
        return df_group

    # Group by an identifier combining latitude and longitude
    if "identifier" not in df.columns:
        df["identifier"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    df = df.groupby("identifier", group_keys=False).apply(compute_solar)

    return df

def interpolate_series(values, orig_times_numeric, new_times_numeric, method):
    """
    Interpolates a 1D array 'values' given original time stamps and new time stamps.
    Uses cosine-based interpolation if method=='cosine', and linear interpolation otherwise.
    """
    # For each new timestamp, find the surrounding indices in the original time array.
    time_idx_upper = np.searchsorted(orig_times_numeric, new_times_numeric, side='right')
    time_idx_lower = time_idx_upper - 1

    # Prepare an array for interpolated values (defaulting to NaN).
    interp = np.full(new_times_numeric.shape, np.nan, dtype=float)

    # Identify new timestamps that fall within the original time range.
    valid_mask = (time_idx_lower >= 0) & (time_idx_upper < len(orig_times_numeric))
    if not np.any(valid_mask):
        return interp

    lower_vals = values[time_idx_lower[valid_mask]]
    upper_vals = values[time_idx_upper[valid_mask]]
    lower_times = orig_times_numeric[time_idx_lower[valid_mask]]
    upper_times = orig_times_numeric[time_idx_upper[valid_mask]]

    # Compute fractional distance between the lower and upper times.
    delta = (upper_times - lower_times).astype(float)
    frac = np.zeros_like(lower_vals, dtype=float)
    nonzero = delta != 0
    frac[nonzero] = (new_times_numeric[valid_mask][nonzero] - lower_times[nonzero]) / delta[nonzero]

    # Apply the appropriate interpolation.
    if method == 'cosine':
        weight = (1 - np.cos(np.pi * frac)) / 2
        interp[valid_mask] = lower_vals + (upper_vals - lower_vals) * weight
    else:  # linear interpolation
        interp[valid_mask] = lower_vals + (upper_vals - lower_vals) * frac

    return interp


def aggregate_power_output_to_excel(df, output_file='Aggregated_Wind_Energy.xlsx'):
    """
    Aggregate the power output of all wind farms by timestamp and save the result to an Excel file.

    Parameters:
        df (pandas.DataFrame): DataFrame containing wind farm outputs with a 'LocalTime' column and 'power_kWh' column.
        output_file (str, optional): Path to the output Excel file.
    """
    # Ensure LocalTime is a datetime type for proper processing and Excel export.
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])

    # Aggregate power output by LocalTime and extract date components.
    aggregated = (
        df.groupby('LocalTime', as_index=False)['power_kWh']
        .sum()
        .assign(Year=lambda x: x['LocalTime'].dt.year,
                Month=lambda x: x['LocalTime'].dt.month,
                Day=lambda x: x['LocalTime'].dt.day,
                Hour=lambda x: x['LocalTime'].dt.hour)
    )
    aggregated.to_excel(output_file, index=False)
    print(f"Aggregated data saved to {output_file}")


def merge_era5_and_rcp_data(input_dir, output_file="Merged_Power_Output.xlsx"):
    """
    Reads all aggregated ERA5 and RCP scenario files from `input_dir` (Excel or Parquet).
    Creates a single DataFrame with columns [LocalTime, Scenario, Source, power_kWh].
    Pivots to wide format so each scenario+source is a separate column.
    Keeps ERA5 as the main total, and places RCP scenario values alongside it.
    """

    output_path = os.path.join(input_dir, output_file)
    all_dfs = []
    # Match any file with "_aggregated." in the name
    pattern = os.path.join(input_dir, "*_aggregated.*")
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path).lower()

        # Identify the power source
        if "hydro" in filename:
            source = "Hydro"
        elif "wind" in filename:
            source = "Wind"
        elif "photovoltaic" in filename:
            source = "Solar"  # or "Photovoltaic"
        else:
            print(f"Skipping file with unknown source: {file_path}")
            continue

        # Identify the scenario
        if "era5" in filename:
            scenario = "ERA5"
        elif "rcp_2_6" in filename:
            scenario = "RCP2_6"
        elif "rcp_4_5" in filename:
            scenario = "RCP4_5"
        elif "rcp_8_5" in filename:
            scenario = "RCP8_5"
        else:
            scenario = "Unknown"

        # Read the file
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            print(f"Skipping unsupported format: {file_path}")
            continue

        # Ensure LocalTime is datetime
        df["LocalTime"] = pd.to_datetime(df["LocalTime"])

        # Keep only the essential columns
        df = df[["LocalTime", "power_kWh"]].copy()

        # Add scenario & source
        df["Scenario"] = scenario
        df["Source"] = source

        all_dfs.append(df)

    if not all_dfs:
        print("No valid files found to merge.")
        return

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Pivot to wide format:
    #   index   = LocalTime
    #   columns = (Scenario, Source)
    #   values  = power_kWh
    pivot_df = combined_df.pivot_table(
        index="LocalTime",
        columns=["Scenario", "Source"],
        values="power_kWh",
        aggfunc="sum"  # sum if there are duplicates
    ).reset_index()

    # Flatten multi-level columns -> "Scenario_Source"
    pivot_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in pivot_df.columns.values
    ]

    # Create total columns per scenario, e.g. "ERA5_Total", "RCP2_6_Total", etc.
    scenarios = combined_df["Scenario"].unique()
    for sc in scenarios:
        # Find all columns that start with, e.g., "ERA5_"
        scenario_cols = [c for c in pivot_df.columns if c.startswith(sc + "_")]
        if scenario_cols:
            pivot_df[f"{sc}_Total"] = pivot_df[scenario_cols].sum(axis=1)

    # ---------------------------------------------------------------
    # Make a "PreferredTotal" column that uses ERA5 if it exists.
    # ---------------------------------------------------------------
    if "ERA5_Total" in pivot_df.columns:
        pivot_df["PreferredTotal"] = pivot_df["ERA5_Total"]
    else:
        pivot_df["PreferredTotal"] = None  # fallback if no ERA5 data at all

    # Add convenience columns for year, month, day, hour
    pivot_df["Year"] = pivot_df["LocalTime"].dt.year
    pivot_df["Month"] = pivot_df["LocalTime"].dt.month
    pivot_df["Day"]   = pivot_df["LocalTime"].dt.day
    pivot_df["Hour"]  = pivot_df["LocalTime"].dt.hour

    # Save to Excel
    pivot_df.to_excel(output_path, index=False)
    logger.info(f"Merged data saved to {output_path}")


def generate_analysis_data(input_path, redes_data_path, output_path=None, interval="15min"):
    """
    Reads all aggregated ERA5 and RCP scenario files from `input_path` (Excel or Parquet).
    Produces two datasets:

      1. Reanalysis.xlsx:
         - Uses ERA5 data only.
         - Expected columns:
             LocalTime,
             Total Prod in GWh EDP (sum of ERA5 Hydro, Photovoltaics and Wind, converted from kWh to GWh),
             Hydro kWH EDP,
             Photovoltaics kWH EDP,
             Wind kWH EDP.
         - Merged with additional redes_data (from redes_data_path) on LocalTime.

      2. Forecasts.xlsx:
         - Uses forecast data (RCP scenarios).
         - Pivots data so that for each scenario (e.g., RCP2_6, RCP4_5, RCP8_5) you get columns for Hydro, Photovoltaics and Wind.
         - Additionally computes a total (aggregated) column for each scenario.

    The output filenames will include the interval based on the difference between consecutive LocalTime values,
    for example: "Forecasts_15min.xlsx" or "Reanalysis_15min.xlsx".

    Parameters:
        input_path (str): Directory containing the ERA5/RCp aggregated files.
        redes_data_path (str): Path to the redes_data Excel file.
        output_path (str, optional): Directory to save output files. Defaults to input_dir.
        interval (str): Time interval of files
    """
    max_reanalysis_time = pd.to_datetime("31.12.2024", format="%d.%m.%Y")
    if output_path is None:
        output_path = input_path

    input_path = os.path.join(input_path, "Predictions")

    all_dfs = []
    # Match any file with "_aggregated." in the name
    pattern = os.path.join(input_path, f"*_{interval}_aggregated.*")
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path).lower()

        # Identify the power source based on the filename
        if "hydro" in filename:
            source = "Hydro"
        elif "wind" in filename:
            source = "Wind"
        elif "photovoltaic" in filename or "solar" in filename:
            source = "Photovoltaics"
        else:
            logger.info(f"Skipping file with unknown source: {file_path}")
            continue

        # Identify the scenario
        if "era5" in filename:
            scenario = "ERA5"
        elif "rcp_2_6" in filename:
            scenario = "RCP2_6"
        elif "rcp_4_5" in filename:
            scenario = "RCP4_5"
        elif "rcp_8_5" in filename:
            scenario = "RCP8_5"
        else:
            scenario = "Unknown"

        # Read the file
        try:
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path, engine="openpyxl")
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                logger.info(f"Skipping unsupported format: {file_path}")
                continue
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

        # Ensure the timestamp column is named "LocalTime"
        if "LocalTime" not in df.columns:
            if "Date/Time" in df.columns:
                df.rename(columns={"Date/Time": "LocalTime"}, inplace=True)
        df["LocalTime"] = pd.to_datetime(df["LocalTime"])

        # Check for expected power column
        if "power_kWh" not in df.columns:
            logger.error(f"'power_kWh' column not found in {file_path}. Skipping.")
            continue

        # Keep only relevant columns and add Scenario and Source identifiers
        df = df[["LocalTime", "power_kWh"]].copy()
        df["Scenario"] = scenario
        df["Source"] = source

        all_dfs.append(df)

    if not all_dfs:
        logger.error("No valid files found to merge.")
        return None, None

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # -------------------------------
    # Create Reanalysis dataset
    # -------------------------------
    reanalysis_df = combined_df[combined_df["Scenario"] == "ERA5"].copy()
    # Pivot: index = LocalTime, columns = Source, values = power_kWh
    reanalysis_pivot = reanalysis_df.pivot_table(
        index="LocalTime",
        columns="Source",
        values="power_kWh",
        aggfunc="sum"
    ).reset_index()

    # Ensure expected sources exist; if missing, fill with 0
    for col in ["Hydro", "Photovoltaics", "Wind"]:
        if col not in reanalysis_pivot.columns:
            reanalysis_pivot[col] = 0

    # Calculate Total Production in GWh (converting kWh to GWh)
    reanalysis_pivot["Total Prod in GWh EDP"] = (
        reanalysis_pivot["Hydro"] + reanalysis_pivot["Photovoltaics"] + reanalysis_pivot["Wind"]
    ) / 1_000_000

    # Rename source columns to include "EDP"
    reanalysis_pivot.rename(columns={
        "Hydro": "Hydro EDP",
        "Photovoltaics": "Photovoltaics EDP",
        "Wind": "Wind EDP"
    }, inplace=True)

    # Merge with redes_data based on LocalTime
    try:
        redes_data = pd.read_excel(redes_data_path)
        if "Date/Time" in redes_data.columns:
            redes_data.rename(columns={"Date/Time": "LocalTime"}, inplace=True)
        redes_data["LocalTime"] = pd.to_datetime(redes_data["LocalTime"])
    except Exception as e:
        logger.error(f"Error reading redes_data file: {e}")
        redes_data = None

    if redes_data is not None:
        reanalysis_merged = pd.merge(reanalysis_pivot, redes_data, on="LocalTime", how="outer")
    else:
        reanalysis_merged = reanalysis_pivot.copy()

    max_reanalysis_time = reanalysis_merged["LocalTime"].max()

    # Determine the time interval for the Reanalysis dataset
    if interval is not None:
        reanalysis_filename = f"Reanalysis_{interval}.xlsx"
    else:
        reanalysis_filename = "Reanalysis.xlsx"
    reanalysis_output_path = os.path.join(output_path, reanalysis_filename)
    reanalysis_merged.to_excel(reanalysis_output_path, index=False)
    logger.info(f"Reanalysis data saved to {reanalysis_output_path}")

    # -------------------------------
    # Create Forecasts dataset
    # -------------------------------
    forecasts_df = combined_df[combined_df["Scenario"] != "ERA5"].copy()
    # Pivot with MultiIndex columns: index = LocalTime, columns = [Scenario, Source]
    forecasts_pivot = forecasts_df.pivot_table(
        index="LocalTime",
        columns=["Scenario", "Source"],
        values="power_kWh",
        aggfunc="sum"
    ).reset_index()

    # Flatten multi-level columns so that we have names like "RCP2_6_Hydro"
    forecasts_pivot.columns = (
        ["LocalTime"] +
        [f"{sc}_{src}" for sc, src in forecasts_pivot.columns if sc != "LocalTime"]
    )

    # For each forecast scenario, compute an aggregated total (sum of the sources)
    forecast_scenarios = forecasts_df["Scenario"].unique()
    for scenario in forecast_scenarios:
        scenario_cols = [col for col in forecasts_pivot.columns if col.startswith(f"{scenario}_")]
        if scenario_cols:
            forecasts_pivot[f"{scenario}_Total"] = forecasts_pivot[scenario_cols].sum(axis=1)

    forecasts_pivot = forecasts_pivot[forecasts_pivot["LocalTime"]>max_reanalysis_time]

    if interval is not None:
        forecasts_filename = f"Forecasts_{interval}.xlsx"
    else:
        forecasts_filename = "Forecasts.xlsx"
    forecasts_output_path = os.path.join(output_path, forecasts_filename)
    forecasts_pivot.to_excel(forecasts_output_path, index=False)
    logger.info(f"Forecasts data saved to {forecasts_output_path}")

    return reanalysis_merged, forecasts_pivot