import os
import cdsapi
import logging
import time
from typing import List, Dict, Tuple, Union
import xarray as xr
import zipfile
import shutil


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('cdsapi').setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def fetch_climate_data(output_path: str, climate_config: Dict) -> None:
    """
    Fetches ERA5 baseline data for Portugal and the US and Copernicus future projection data for Portugal.
    Downloads one variable at a time and then concatenates the individual files into a single dataset.
    The combined datasets are saved under a new directory (specified by 'output' in the climate config)
    inside the given output_path.

    Parameters:
        output_path (str): Base directory where climate data will be saved.
        climate_config (dict): Configuration dictionary with the climate settings.
    """
    base_dir_name = climate_config.get("output", "Climate Data")
    base_dir = os.path.join(output_path, base_dir_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logger.info(f"Created directory for climate data: {base_dir}")
    else:
        logger.info(f"Using existing climate data directory: {base_dir}")

    # Time parameters for ERA5 baseline.
    time_config = climate_config.get("time", {})
    baseline_years: List[str] = time_config.get("baseline_years", [])
    train_years: List[str] = time_config.get("train_years", [])
    # ERA5 request uses full months/days/times.
    months: List[str] = [f"{m:02d}" for m in range(1, 13)]
    days: List[str] = [f"{d:02d}" for d in range(1, 32)]
    times: List[str] = [f"{t:02d}:00" for t in range(0, 24)]

    # Areas.
    areas_config = climate_config.get("areas", {})
    portugal_area: List[float] = areas_config.get("portugal")
    us_area: List[float] = areas_config.get("us")

    # ERA5 variables.
    variables_era5: Dict[str, Union[str, List[str]]] = climate_config.get("variables_era5", {})

    # --- Fetch ERA5 Baseline Data for Portugal ---
    logger.info("Fetching ERA5 baseline data for Portugal...")
    _download_and_combine_era5_data(
        base_dir, variables_era5, baseline_years, months, days, times,
        portugal_area, "era5_portugal_baseline.nc"
    )

    # --- Fetch ERA5 Baseline Data for the US ---
    logger.info("Fetching ERA5 2006 data for the US...")
    if isinstance(variables_era5["precipitation"], str):
        variables_era5["precipitation"] = [variables_era5["precipitation"]]

    variables_era5["precipitation"].append("snow_depth")
    _download_and_combine_era5_data(
        base_dir, variables_era5, train_years, months, days, times,
        us_area, "era5_us_2006.nc"
    )

    # --- Fetch Cordex Future Projection Data for Portugal ---
    variables_cordex: Dict[str, str] = climate_config.get("variables_cordex", {})
    copernicus_scenarios: List[str] = variables_cordex.get("copernicus_scenarios", [])
    # For future projections, use the time config to derive start and end years.
    for scenario in copernicus_scenarios:
        logger.info(f"Fetching Copernicus future projection data for Portugal under {scenario.upper()} scenario...")
        target_file = f"copernicus_{scenario}_portugal_future.nc"
        _download_and_combine_cordex_data(
            base_dir, variables_cordex, time_config, scenario, target_file
        )


def _retry_retrieve(client: cdsapi.Client, collection_id: str, request_params: Dict, full_path: str,
                    max_attempts: int = 10, initial_delay: int = 30) -> None:
    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            client.retrieve(collection_id, request_params, full_path)
            logger.debug(f"Data successfully retrieved and saved to {full_path}")
            return
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    raise Exception(f"Failed to retrieve data after {max_attempts} attempts.")


def _download_and_combine_era5_data(base_dir: str, variables_config: Dict[str, str],
                                         years: List[str], months: List[str], days: List[str], times: List[str],
                                         area: List[float], target_file: str) -> None:
    combined_output = os.path.join(base_dir, target_file)
    if os.path.exists(combined_output):
        logger.info(f"Combined ERA5 dataset already exists at {combined_output}. Skipping download.")
        return

    temp_files: List[str] = []
    client = cdsapi.Client()
    for var_key, var_names in variables_config.items():
        # Ensure var_names is a list
        if isinstance(var_names, str):
            var_names = [var_names]

        # Change temporary file extension to .grib
        temp_filename = f"temp_era5_{var_key}.grib"
        full_temp_path = os.path.join(base_dir, temp_filename)
        if os.path.exists(full_temp_path):
            logger.info(f"Temp ERA5 dataset already exists at {full_temp_path}. Skipping download.")
            temp_files.append(full_temp_path)
            continue

        request_params = {
            'product_type': 'reanalysis',
            'variable': var_names,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': area,  # [North, West, South, East]
            'format': 'grib'  # Request GRIB format
        }
        logger.info(f"Requesting ERA5 data for variable: {var_names} in GRIB format...")
        _retry_retrieve(client, 'reanalysis-era5-single-levels', request_params, full_temp_path)
        temp_files.append(full_temp_path)

    # Use cfgrib engine to read GRIB files and combine them by coordinates
    combined_ds = xr.open_mfdataset(
        temp_files,
        combine='by_coords',
        engine='cfgrib',
        compat='override',
        join='override',
        backend_kwargs={'decode_timedelta': None}
    )
    combined_ds.to_netcdf(combined_output)
    logger.info(f"Combined ERA5 dataset saved to {combined_output}")

    # Clean up temporary files
    for f in temp_files:
        os.remove(f)
        logger.debug(f"Removed temporary file: {f}")


def _get_future_years(time_config: Dict) -> Tuple[List[str], List[str]]:
    baseline_years = time_config.get("baseline_years", [])
    future_years = time_config.get("future_years", [])
    if not baseline_years or not future_years:
        raise ValueError("Both 'baseline_years' and 'future_years' must be provided in the time config.")
    last_baseline = baseline_years[-1]
    start_years = [last_baseline] + future_years
    end_years = start_years[:-1] + [str(int(start_years[-1]) + 1)]
    return start_years, end_years


def _extract_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extracts the ZIP archive at zip_path into extract_dir, while intelligently
    handling extra folders (like __MACOSX) and nested directory structures.
    Returns a list of all extracted file paths.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Remove the __MACOSX directory if it exists
    macosx_path = os.path.join(extract_dir, '__MACOSX')
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)

    # Check if the extraction created a single non-empty folder (like temp_cordex_rcp_8_5_wind_v)
    items = [item for item in os.listdir(extract_dir) if not item.startswith('.')]
    if len(items) == 1 and os.path.isdir(os.path.join(extract_dir, items[0])):
        extract_root = os.path.join(extract_dir, items[0])
    else:
        extract_root = extract_dir

    extracted_files = []
    # Walk the directory tree and collect file paths
    for root, dirs, files in os.walk(extract_root):
        for file in files:
            extracted_files.append(os.path.join(root, file))
    return extracted_files


def _subset_to_portugal(ds: xr.Dataset) -> xr.Dataset:
    """
    Subsets a dataset to a defined Portugal region using cell bounds and drops grid cells
    that have NA values for any of the allowed variables, including dropping points where
    'rsds' is entirely NaN over the time dimension.

    Assumes the dataset contains:
      - 'lat' and 'lon' as 2D arrays (cell centers)
      - 'bounds_lat' and 'bounds_lon' as 3D arrays (cell vertex coordinates; dims: y, x, nvertex)
      - a time dimension

    The target region is defined as:
       - Latitude between 36 and 42 degrees
       - Longitude: [-9.5, -6.0] if data are in [-180,180],
         or [350.5, 354.0] if in [0,360].
    """
    # Determine target longitude bounds.
    lon_min = float(ds.lon.min().values)
    lon_max = float(ds.lon.max().values)
    if lon_min >= 0 and lon_max <= 360:
        target_lon_min, target_lon_max = 350.5, 354.0
    else:
        target_lon_min, target_lon_max = -9.5, -6.0
    target_lat_min, target_lat_max = 36.0, 42.5

    # Compute cell bounds from vertex coordinates.
    cell_lat_min = ds.bounds_lat.min(dim="nvertex")
    cell_lat_max = ds.bounds_lat.max(dim="nvertex")
    cell_lon_min = ds.bounds_lon.min(dim="nvertex")
    cell_lon_max = ds.bounds_lon.max(dim="nvertex")

    # Create a mask for grid cells that intersect the target region.
    mask = (
            (cell_lat_max >= target_lat_min) &
            (cell_lat_min <= target_lat_max) &
            (cell_lon_max >= target_lon_min) &
            (cell_lon_min <= target_lon_max)
    )
    ds_subset = ds.where(mask, drop=True)

    # Round the cell center coordinates.
    for coord in ['lat', 'lon']:
        if coord in ds_subset:
            ds_subset[coord] = ds_subset[coord].round(3)

    # Allowed keys to retain.
    allowed_keys = {
        "time", "y", "x", "nvertex", "axis_nbounds", "lat", "lon",
        "rsds", "wind_u", "wind_v", "cloud_cover", "temperature",
        "mrro", "clt", "rsus", "rlds", "ps", "orog", "tasmin", "psl",
        "pr", "evspsbl", "tasmax", "sftlf", "va850", "ua850", "zg500",
        "va200", "ua200", "ta200", "sfcWind", "vas", "uas", "tas", "huss", "hurs"
    }


    # Stack the spatial dimensions to treat each grid cell as a single entry.
    ds_stack = ds_subset.stack(points=("y", "x"))

    subset_vars = [var for var in allowed_keys if var in ds_stack.data_vars]

    # Drop grid points where 'rsds' is entirely NaN over the time dimension.
    ds_stack = ds_stack.dropna(dim="points", how="all", subset=subset_vars)

    # Unstack to restore original spatial dimensions.
    ds_clean = ds_stack.unstack("points")

    # Prune the dataset to only include allowed variables and coordinates.
    new_vars = {var: ds_clean[var] for var in ds_clean.data_vars if var in allowed_keys}
    new_coords = {coord: ds_clean.coords[coord] for coord in ds_clean.coords if coord in allowed_keys}

    return xr.Dataset(data_vars=new_vars, coords=new_coords)


def _download_and_combine_cordex_data(
        base_dir: str,
        cordex_config: Dict[str, str],
        time_config: Dict,
        scenario: str,
        target_file: str
):
    """
    Downloads Cordex data for specified variables and time range for the given scenario.
    For each variable:
      - A ZIP archive is downloaded (if not already present)
      - It is extracted (yielding one file per year)
      - Each file is subset to Portugal using _subset_to_portugal
    The yearly subsetted files are concatenated along time.
    Finally, all variable files are merged into a combined dataset, which is converted to a pandas DataFrame and returned.
    Temporary files and directories are removed afterward.
    """
    combined_output = os.path.join(base_dir, target_file)
    if os.path.exists(combined_output):
        logger.info(
            f"Combined Cordex dataset for scenario {scenario} already exists at {combined_output}. Skipping download.")
        return

    temp_var_files: List[str] = []
    client = cdsapi.Client()
    var_keys = ["solar", "thermal", "wind_u", "wind_v", "cloud_cover", "temperature", "precipitation"]

    # Temporary directory for zip extraction.
    temp_extract_dir = os.path.join(base_dir, "cordex_temp_extract")

    for var_key in var_keys:
        start_years, end_years = _get_future_years(time_config)
        if var_key not in cordex_config:
            continue
        elif var_key in ["precipitation"]:
            start_years = [x for x in start_years if int(x) <= 2049]
            end_years = [x for x in end_years if int(x) <= 2049]
        elif var_key in ["cloud_cover"]:
            start_years = [x for x in start_years if int(x) <= 2048]
            end_years = [x for x in end_years if int(x) <= 2048]
        elif var_key in ["temperature"]:
            start_years = [x for x in start_years if int(x) <= 2048]
            end_years = [x for x in end_years if int(x) <= 2049]
        var_name = cordex_config[var_key]
        temp_var_filename = f"temp_cordex_{scenario}_{var_key}_subset.nc"
        temp_zip_filename = f"temp_cordex_{scenario}_{var_key}.nc.zip"
        full_zip_path = os.path.join(base_dir, temp_zip_filename)
        full_var_path = os.path.join(base_dir, temp_var_filename)

        # Skip download if zip file exists
        if os.path.exists(full_zip_path):
            logger.info(f"Temporary file {full_zip_path} already exists. Skipping download for {var_name}.")
        elif os.path.exists(full_var_path):
            logger.info(f"Temporary file {full_var_path} already exists. Skipping download and processing for {var_name}.")
            temp_var_files.append(full_var_path)
            continue
        else:
            desired_temporal = "6_hours" if var_key in ["wind_u", "wind_v"] else "3_hours"
            request_params = {
                "domain": cordex_config["domain"],
                "experiment": scenario,
                "horizontal_resolution": cordex_config["horizontal_resolution"],
                "temporal_resolution": desired_temporal,
                "variable": [var_name],
                "gcm_model": cordex_config["gcm_model"],
                "rcm_model": cordex_config["rcm_model"],
                "ensemble_member": cordex_config["ensemble_member"],
                "start_year": start_years,
                "end_year": end_years
            }
            logger.debug(f"Requesting Cordex data for variable: {var_name} under scenario: {scenario}...")
            _retry_retrieve(client, "projections-cordex-domains-single-levels", request_params, full_zip_path)
        # Extract all files from the zip.
        os.makedirs(temp_extract_dir, exist_ok=True)
        extracted_files = _extract_zip(full_zip_path, temp_extract_dir)
        logger.info(f"Extracted {len(extracted_files)} files from {full_zip_path} for variable {var_name}.")
        ds_list = []
        for f in extracted_files:
            try:
                ds = xr.open_dataset(f, engine="netcdf4")
                ds_subset = _subset_to_portugal(ds)
                ds_list.append(ds_subset)
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
        if not ds_list:
            logger.error(f"No datasets processed for variable {var_name}.")
            continue
        try:
            combined_var_ds = xr.concat(ds_list, dim="time")
        except Exception as e:
            logger.error(f"Error concatenating datasets for variable {var_name}: {e}")
            continue
        combined_var_ds.to_netcdf(full_var_path)
        logger.info(f"Saved subsetted data for variable {var_name} to {full_var_path}.")
        temp_var_files.append(full_var_path)
        for f in extracted_files:
            os.remove(f)
            logger.debug(f"Removed extracted file: {f}")
        shutil.rmtree(temp_extract_dir)
        os.remove(full_zip_path)
        logger.debug(f"Removed zip file: {full_zip_path}")

    if not temp_var_files:
        raise RuntimeError("No temporary variable files were produced.")

    # Combine all variable files into one dataset.
    combined_ds = xr.open_mfdataset(temp_var_files, combine='by_coords', engine='netcdf4', compat='override')
    # Optionally, load into memory (if the subset is small enough)
    combined_ds.load()

    logger.info(f"Combined Cordex dataset for scenario {scenario} converted to DataFrame.")
    combined_output = os.path.join(base_dir, target_file)
    combined_ds.to_netcdf(combined_output)

    # Cleanup temporary files and directories.
    for f in temp_var_files:
        os.remove(f)
        logger.debug(f"Removed temporary file: {f}")
    logger.info(f"Cleaned up temporary extraction directory {temp_extract_dir}.")

