import os
import glob
import re
import logging
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pytz
import pvlib
import srtm
from scipy.spatial import cKDTree

from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

tf = TimezoneFinder()
elev_data = srtm.get_data()

ALLOWED_KEYS = {
        "time", "y", "x", "nvertex", "axis_nbounds", "lat", "lon",
        "rsds", "wind_u", "wind_v", "cloud_cover", "temperature",
        "mrro", "clt", "rsus", "rlds", "ps", "orog", "tasmin", "psl",
        "pr", "evspsbl", "tasmax", "sftlf", "va850", "ua850", "zg500",
        "va200", "ua200", "ta200", "sfcWind", "vas", "uas", "tas", "huss", "hurs"
    }

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns the distance in kilometers.
    """
    # Convert decimal degrees to radians.
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers.
    return c * r

def cache_timezones(df):
    """
    Build a cache (dictionary) of timezone objects keyed by unique (latitude, longitude)
    so that we do not repeatedly look up the same timezone.
    """
    unique_coords = df[['latitude', 'longitude']].drop_duplicates()
    tz_cache = {}
    for _, row in unique_coords.iterrows():
        coord = (row['latitude'], row['longitude'])
        tz_str = tf.timezone_at(lng=row['longitude'], lat=row['latitude'])
        if tz_str is None:
            raise ValueError(f"Timezone not found for coordinates: {coord}")
        tz_cache[coord] = pytz.timezone(tz_str)
    return tz_cache


def vectorized_localize(df, tz_cache):
    """
    Convert naive 'LocalTime' in df to UTC, grouping by (latitude, longitude)
    and using vectorized dt.tz_localize where possible.
    """
    def safe_localize(ts, tz):
        # Fallback to per-timestamp processing if vectorized method fails.
        try:
            localized = tz.localize(ts, is_dst=None)
        except pytz.AmbiguousTimeError:
            localized = tz.localize(ts, is_dst=True)
        except pytz.NonExistentTimeError:
            shifted = ts + pd.Timedelta(hours=1)
            localized = tz.localize(shifted, is_dst=True) - pd.Timedelta(hours=1)
        except Exception:
            return pd.NaT
        return localized.astimezone(pytz.utc)

    localized_groups = []
    # Group by unique (latitude, longitude) since each group shares the same tz.
    for coords, group in df.groupby(['latitude', 'longitude']):
        tz = tz_cache.get(coords)
        if tz is None:
            raise ValueError(f"Timezone not found for coordinates: {coords}")
        # Try vectorized localization for the group.
        try:
            # Here we use ambiguous=True to resolve ambiguous times as in your original code.
            # For nonexistent times, we use 'shift_forward'. (This may not perfectly match the custom fix.)
            localized = group['LocalTime'].dt.tz_localize(tz, ambiguous=True, nonexistent='shift_forward')
            localized = localized.dt.tz_convert('UTC')
        except Exception:
            # If the vectorized operation fails (e.g. due to DST transition complexities),
            # fall back to applying a safe localization function row by row.
            localized = group['LocalTime'].apply(lambda ts: safe_localize(ts, tz))
        group = group.copy()  # Avoid SettingWithCopy warnings.
        group['UTC'] = localized
        localized_groups.append(group)
    # Recombine the groups and return the full DataFrame.
    return pd.concat(localized_groups).sort_index()

def compute_grid_mapping(df, ds):
    """
    Maps each unique coordinate in `df` to the nearest grid point in the dataset `ds`
    using a KD-tree, supporting different variations of latitude and longitude column names.
    Also supports CORDEX grids where the lat/lon arrays may be 2D.

    Parameters:
      df (DataFrame): DataFrame containing at least 'latitude'/'lat' and 'longitude'/'lon' columns.
      ds (xarray.Dataset or dict-like): Dataset containing grid information. Must have
          'latitude'/'lat' and 'longitude'/'lon' arrays.

    Returns:
      mapping_df (DataFrame): DataFrame with columns:
          - latitude, longitude: Unique coordinates from df.
          - lat_idx, lon_idx: Indices of the nearest grid cell in the original ds.
          - grid_latitude, grid_longitude: The grid cell coordinates from ds.
          - distance: Euclidean distance (in degrees) between the df coordinate and the grid cell.
          - dist_in_km: Haversine distance (in kilometers) between the df coordinate and the grid cell.
    """
    # If ds has 'y' and 'x' dims, reformat it for consistency.
    if hasattr(ds, 'dims') and 'y' in ds.dims and 'x' in ds.dims:
        ds_stack = ds.stack(points=("y", "x"))
        subset_vars = [var for var in ALLOWED_KEYS if var in ds_stack.data_vars]
        ds_stack = ds_stack.dropna(dim="points", how="all", subset=subset_vars)
        ds = ds_stack.unstack("points")

    # Determine which column names are used for latitude and longitude in df.
    lat_col = next((col for col in ['latitude', 'lat'] if col in df.columns), None)
    lon_col = next((col for col in ['longitude', 'lon'] if col in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError("DataFrame must contain 'latitude'/'lat' and 'longitude'/'lon' columns.")

    # Determine latitude and longitude keys in ds.
    ds_lat_key = next((key for key in ['latitude', 'lat'] if key in ds), None)
    ds_lon_key = next((key for key in ['longitude', 'lon'] if key in ds), None)

    if ds_lat_key is None or ds_lon_key is None:
        raise ValueError("Dataset must contain 'latitude'/'lat' and 'longitude'/'lon' arrays.")

    # Extract grid coordinates from ds.
    grid_lats = ds[ds_lat_key].values
    grid_lons = ds[ds_lon_key].values

    # Prepare grid points and index mapping depending on the dimensionality.
    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        grid_points = []
        index_mapping = []
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                if pd.notna(lat) and pd.notna(lon):
                    grid_points.append([lat, lon])
                    index_mapping.append((i, j))
        grid_points = np.array(grid_points)
    elif grid_lats.ndim == 2 and grid_lons.ndim == 2:
        grid_points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
        index_mapping = [(i, j) for i in range(grid_lats.shape[0]) for j in range(grid_lats.shape[1])]
    else:
        raise ValueError("Grid latitude and longitude arrays have unsupported dimensions.")

    # Filter out any grid points with non-finite values and filter the index mapping accordingly.
    valid_mask = np.isfinite(grid_points).all(axis=1)
    grid_points_clean = grid_points[valid_mask]
    index_mapping_clean = [m for m, valid in zip(index_mapping, valid_mask) if valid]

    # Build a KD-tree on the filtered grid points.
    tree = cKDTree(grid_points_clean)

    # Extract unique coordinates from the DataFrame.
    unique_coords = df[[lat_col, lon_col]].drop_duplicates().to_numpy()

    # Query the tree for the nearest grid point for each coordinate.
    distances, filtered_indices = tree.query(unique_coords)

    # Map filtered indices back to the original grid indices.
    lat_idx = [index_mapping_clean[idx][0] for idx in filtered_indices]
    lon_idx = [index_mapping_clean[idx][1] for idx in filtered_indices]

    # Retrieve the grid coordinates for these nearest points from the filtered grid.
    grid_latitudes = grid_points_clean[filtered_indices, 0]
    grid_longitudes = grid_points_clean[filtered_indices, 1]

    # Compute the Haversine distance (in km).
    df_lats = unique_coords[:, 0]
    df_lons = unique_coords[:, 1]
    dist_in_km = haversine_distance(df_lons, df_lats, grid_longitudes, grid_latitudes)

    # Create and return the mapping DataFrame.
    mapping_df = pd.DataFrame({
        'latitude': unique_coords[:, 0],
        'longitude': unique_coords[:, 1],
        'lat_idx': lat_idx,
        'lon_idx': lon_idx,
        'grid_latitude': grid_latitudes,
        'grid_longitude': grid_longitudes,
        'distance': distances,      # Euclidean distance in degrees.
        'dist_in_km': dist_in_km     # Haversine distance in kilometers.
    })

    return mapping_df


def compute_clear_sky_index(group):
    # Extract the unique latitude and longitude for the group.
    lat = group['latitude'].iloc[0]
    lon = group['longitude'].iloc[0]
    # Create a Location object for this plant.
    location = pvlib.location.Location(latitude=lat, longitude=lon, tz='UTC')
    # Convert LocalTime to a DatetimeIndex (which pvlib expects)
    times = pd.DatetimeIndex(group['LocalTime'])
    cs = location.get_clearsky(times)

    # Extract GHI values
    ghi = cs['ghi'].values
    # Small threshold to avoid division by zero.
    epsilon = 1e-6
    # Get ssrd values (ensure alignment)
    ssrd_vals = group['ssrd'].values.astype(float)

    # Create an empty array for clear sky index.
    clear_sky_index = np.zeros_like(ghi, dtype=float)

    # For rows where GHI is above the threshold, compute the ratio.
    valid = ghi > epsilon
    clear_sky_index[valid] = ssrd_vals[valid] / ghi[valid]

    # Replace any inf or nan values with 0 (or another default if desired).
    clear_sky_index = np.nan_to_num(clear_sky_index, nan=-1.0, posinf=-2.0, neginf=-1.0)

    # Assign the cleaned clear sky index back to the group.
    group['clear_sky_index'] = clear_sky_index
    return group

def compute_elevation(group):
    # Extract the unique latitude and longitude for the group.
    lat = group['latitude'].iloc[0]
    lon = group['longitude'].iloc[0]
    # Get the elevation for this location.
    elevation = elev_data.get_elevation(lat, lon)
    # Assign the computed elevation to all rows in the group.
    group['elevation'] = elevation
    if group['elevation'].isna().any():
        group['elevation'] = -1
    return group


def natural_sort_key(s):
    """Generate a sort key that treats numbers in the string as numeric."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def concatenate_parquet_files(batch_dir, output_file, pattern="batch_*.parquet"):
    """
    Concatenates multiple parquet files from batch_dir into one output file in a memory efficient manner.

    Parameters:
        batch_dir (str): Directory containing batch parquet files.
        output_file (str): Path to the final combined parquet file.
        pattern (str, optional): Glob pattern to match batch files.
    """
    batch_files = sorted(glob.glob(os.path.join(batch_dir, pattern)), key=natural_sort_key)
    if not batch_files:
        logger.warning("No batch files found to concatenate.")
        return

    writer = None
    try:
        for file in batch_files:
            logger.info(f"Processing {file}...")
            table = pq.read_table(file)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
        logger.info(f"Saved combined data to {output_file}")
    finally:
        if writer:
            writer.close()


def reduce_memory_usage(df, exclude_cols=None):
    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    # Downcast floats to float32
    for col in df.select_dtypes(include=['float']).columns:
        if col in exclude_cols:
            continue  # Skip columns that shouldn't be downcasted
        # Convert to float32
        downcasted = df[col].astype(np.float32)
        # Check for acceptable precision loss
        if np.allclose(df[col].values, downcasted.values, rtol=1e-6, atol=1e-9, equal_nan=True):
            df[col] = downcasted

    return df


def add_solar_elevation(df):
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)

    # Determine the time column ('time' or 'UTC')
    time_column = 'time' if 'time' in df.columns else 'UTC' if 'UTC' in df.columns else None
    if time_column is None:
        raise KeyError("The dataframe must contain either a 'time' or 'UTC' column.")

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

    # Group by identifier (latitude, longitude combined)
    if "identifier" not in df.columns:
        df["identifier"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    df = df.groupby("identifier", group_keys=False).apply(compute_solar)

    return df

