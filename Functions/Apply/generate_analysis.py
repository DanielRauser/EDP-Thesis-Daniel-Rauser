import os
import glob
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_analysis_data(input_path, redes_data_path, output_path=None, interval="15min"):
    """
    Reads all aggregated ERA5 and RCP scenario files from input_path (Excel or Parquet).
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
    logger.info("Generating analysis data")
    max_reanalysis_time = pd.to_datetime("31.03.2025", format="%d.%m.%Y")
    min_reanalysis_time = pd.to_datetime("01.01.2020", format="%d.%m.%Y")
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

    del reanalysis_df

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
        redes_data["LocalTime"] = pd.to_datetime(redes_data["LocalTime"])
        redes_data = redes_data[redes_data["LocalTime"] >= min_reanalysis_time]
    except Exception as e:
        logger.error(f"Error reading redes_data file: {e}")
        redes_data = None

    if redes_data is not None:
        reanalysis_merged = pd.merge(reanalysis_pivot, redes_data, on="LocalTime", how="outer")
    else:
        reanalysis_merged = reanalysis_pivot.copy()

    del reanalysis_pivot

    reanalysis_merged["Hydro EDP"] = reanalysis_merged["Hydro (kWh)"] * (5078/8188) # Hydro Market Share of EDP

    pv_cap = 2.5 * 0.174 * reanalysis_merged["Photovoltaics (kWh)"]
    pv_clipped = (reanalysis_merged["Photovoltaics EDP"] > pv_cap).sum()
    reanalysis_merged["Photovoltaics EDP"] = reanalysis_merged.apply(
        lambda x: min(x["Photovoltaics EDP"], 2.5 * 0.174 * x["Photovoltaics (kWh)"]), axis=1
    )
    percentage = round((pv_clipped / len(reanalysis_merged)) * 100, 2)
    logger.info(f"Clipped {percentage}% of Photovoltaics EDP values")

    wind_cap = 2.5 * 0.209 * reanalysis_merged["Wind (kWh)"]
    wind_clipped = (reanalysis_merged["Wind EDP"] > wind_cap).sum()
    reanalysis_merged["Wind EDP"] = reanalysis_merged.apply(
        lambda x: min(x["Wind EDP"], 2 * 0.209 * x["Wind (kWh)"]), axis=1
    )
    percentage = round((wind_clipped / len(reanalysis_merged)) * 100, 2)
    logger.info(f"Clipped {percentage}% of Wind EDP values")

    reanalysis_merged["ETS price"] = reanalysis_merged["ETS price"].replace("", np.nan)

    # Convert column to numeric (just in case some non-numeric strings are in there)
    reanalysis_merged["ETS price"] = pd.to_numeric(reanalysis_merged["ETS price"], errors='coerce')

    # Fill NaNs with rolling average
    reanalysis_merged["ETS price"] = reanalysis_merged["ETS price"].fillna(
        reanalysis_merged["ETS price"].rolling(window=3, min_periods=1).mean()
    )

    reanalysis_merged = reanalysis_merged[reanalysis_merged['LocalTime'].dt.date <= max_reanalysis_time.date()]

    # Determine the time interval for the Reanalysis dataset
    if interval is not None:
        reanalysis_filename = f"Reanalysis_{interval}.xlsx"
    else:
        reanalysis_filename = "Reanalysis.xlsx"
    reanalysis_output_path = os.path.join(output_path, reanalysis_filename)
    reanalysis_merged.to_excel(reanalysis_output_path, index=False)
    logger.info(f"Reanalysis data saved to {reanalysis_output_path}")

    del reanalysis_merged

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

    del combined_df

    # Flatten multi-level columns
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

    forecasts_pivot["LocalTime"] = pd.to_datetime(forecasts_pivot["LocalTime"])

    forecasts_pivot['Year'] = forecasts_pivot['LocalTime'].dt.year

    forecasts_pivot = forecasts_pivot[forecasts_pivot["LocalTime"].dt.date > max_reanalysis_time.date()]

    try:
        consumption_forecast_path = os.path.join(input_path, "Forecast", f"consumption_forecast_{interval}.xlsx")
        consumption_df = pd.read_excel(consumption_forecast_path, engine="openpyxl")
        if "LocalTime" in consumption_df.columns:
            consumption_df.rename(columns={"LocalTime": "LocalTime"}, inplace=True)
        consumption_df["LocalTime"] = pd.to_datetime(consumption_df["LocalTime"])

        forecasts_pivot = pd.merge(forecasts_pivot, consumption_df, on="LocalTime", how="left")
    except Exception as e:
        logger.error(f"Error reading or merging consumption forecast: {e}")

    if interval is not None:
        forecasts_filename = f"Forecasts_{interval}.xlsx"
    else:
        forecasts_filename = "Forecasts.xlsx"
    forecasts_output_path = os.path.join(output_path, forecasts_filename)
    forecasts_pivot.to_excel(forecasts_output_path, index=False, engine="openpyxl")
    logger.info(f"Forecasts data saved to {forecasts_output_path}")