import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from tqdm import tqdm
from scipy.special import gamma
from scipy.stats import weibull_min
import logging

from Functions.Models.Plants.helpers import aggregate_power_output_to_excel, get_weibull_params

logger = logging.getLogger(__name__)

# Turbine parameters (unchanged)
TURBINE_PARAMETERS = {
    "UNKNOWN":         {"v_cut_in": 3.5,  "v_rated": 12.0, "v_cut_out": 25.0, "curve": "flat"},
    "E82/2300":        {"v_cut_in": 3.0,  "v_rated": 12.0, "v_cut_out": 34.0, "curve": "flat"},
    "E82/2000":        {"v_cut_in": 2.0,  "v_rated": 12.5, "v_cut_out": 34.0, "curve": "flat"},
    "E92/2350":        {"v_cut_in": 2.0,  "v_rated": 14.0, "v_cut_out": 25.0, "curve": "flat"},
    "1.5s":            {"v_cut_in": 3.5,  "v_rated": 12.0, "v_cut_out": 25.0, "curve": "flat"},
    "V80/2000":        {"v_cut_in": 3.5,  "v_rated": 14.5, "v_cut_out": 25.0, "curve": "flat"},
    "E40/600":         {"v_cut_in": 2.5,  "v_rated": 13.0, "v_cut_out": 25.0, "curve": "flat"},
    "E66/2000":        {"v_cut_in": 2.5,  "v_rated": 14.0, "v_cut_out": 28.0, "curve": "flat"},
    "E70/2300":        {"v_cut_in": 2.0,  "v_rated": 15.0, "v_cut_out": 25.0, "curve": "flat"},
    "G83/2000":        {"v_cut_in": 3.5,  "v_rated": 16.5, "v_cut_out": 25.0, "curve": "flat"},
    "E70/2000":        {"v_cut_in": 2.0,  "v_rated": 14.0, "v_cut_out": 25.0, "curve": "flat"},
    "3.6M114 NES":     {"v_cut_in": 2.5,  "v_rated": 12.0, "v_cut_out": 21.0, "curve": "flat"},
    "V100/1800":       {"v_cut_in": 4.0,  "v_rated": 12.0, "v_cut_out": 20.0, "curve": "flat"},
    "SG 3.4-132":      {"v_cut_in": 3.0,  "v_rated": 10.3, "v_cut_out": 25.0, "curve": "flat"},
    "V90/3000":        {"v_cut_in": 3.5,  "v_rated": 15.75, "v_cut_out": 25.0, "curve": "flat"},
    "V162/6.2MW":      {"v_cut_in": 3.0,  "v_rated": 10.5, "v_cut_out": 24.0, "curve": "flat"},
    "B62/1300":        {"v_cut_in": 2.5,  "v_rated": 18.0, "v_cut_out": 25.0, "curve": "flat"},
    "V117/4000-4200":  {"v_cut_in": 3.0,  "v_rated": 13.5, "v_cut_out": 25.0, "curve": "flat"},
    "MM92/2050":       {"v_cut_in": 3.5,  "v_rated": 13.0, "v_cut_out": 22.0, "curve": "flat"},
    "G80/2000":        {"v_cut_in": 3.5,  "v_rated": 15.0, "v_cut_out": 25.0, "curve": "flat"},
    "E48/600":         {"v_cut_in": 2.0,  "v_rated": 13.0, "v_cut_out": 25.0, "curve": "flat"},
    "E40/500":         {"v_cut_in": 2.5,  "v_rated": 13.5, "v_cut_out": 25.0, "curve": "flat"},
    "V42/600":         {"v_cut_in": 4.5,  "v_rated": 16.0, "v_cut_out": 25.0, "curve": "flat"},
    "Ecotecnia 74":    {"v_cut_in": 3.0,  "v_rated": 13.0, "v_cut_out": 25.0, "curve": "flat"},
    "80 1.6":          {"v_cut_in": 3.0,  "v_rated": 12.5, "v_cut_out": 25.0, "curve": "fade_out", "fade": 0.05}
}

def get_turbine_parameters(model_name):
    return TURBINE_PARAMETERS.get(model_name, TURBINE_PARAMETERS["UNKNOWN"])

def compute_wind_speed(u, v):
    return np.sqrt(u ** 2 + v ** 2)

def extrapolate_wind_speed(v10, hub_height, alpha=1/7):
    return v10 * (hub_height / 10) ** alpha

def turbine_power_output(v_hub, P_rated, v_cut_in=3.5, v_rated=12.5, v_cut_out=25.5, curve_type="flat", fade=0.0):
    power = np.zeros_like(v_hub)
    mask1 = (v_hub >= v_cut_in) & (v_hub < v_rated)
    power[mask1] = P_rated * ((v_hub[mask1] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    mask2 = (v_hub >= v_rated) & (v_hub < v_cut_out)
    if curve_type == "fade_out":
        fraction = (v_hub[mask2] - v_rated) / (v_cut_out - v_rated)
        power[mask2] = P_rated * (1 - fade * fraction)
    else:
        power[mask2] = P_rated
    return power

def process_windfarm_outputs(df, interval='15min',
                             alpha_10_100=1 / 7, alpha_100_hub=1 / 7,
                             weibull_scale_tif=None, weibull_shape_tif=None,
                             p_loss=0.05, t_loss=0.00):
    """
    Process wind data to compute power output. The approach is modified so that:
      - If the original temporal resolution is finer than or equal to hourly, it uses the original (CubicSpline) interpolation and empirical Weibull parameter estimation.
      - If the original resolution is coarser than hourly (e.g. 3 hourly), it uses a Hermite (PCHIP) interpolator and recalibrates the Weibull parameters via a direct fit.

    New loss parameters:
      - p_loss: Fractional loss representing wake/internal losses. (e.g., 0.25 for 25% loss)
      - t_loss: Fractional loss representing transmission/external losses.
    """
    results = []
    # Expected metadata columns: 'ID', 'Total Capacity (MW)', 'Number of Turbines',
    # 'Hub Height (m)', 'Turbine Model', 'longitude', 'latitude'
    windfarm_meta = extract_windfarm_metadata(df).set_index("ID")
    interval_timedelta = pd.Timedelta(interval)
    interval_hours = interval_timedelta.total_seconds() / 3600

    for wf in tqdm(df['ID'].unique(), desc="Processing Windparks"):
        try:
            meta = windfarm_meta.loc[wf]
            P_rated = meta['Total Capacity (MW)']
            num_turbines = meta['Number of Turbines']
            turbine_model = meta.get('Turbine Model', 'UNKNOWN')
            hub_height_metadata = meta['Hub Height (m)']
            lon_plant = meta['longitude']
            lat_plant = meta['latitude']
            params = get_turbine_parameters(turbine_model)
            v_cut_in = params['v_cut_in']
            v_rated = params['v_rated']
            v_cut_out = params['v_cut_out']
            curve_type = params.get('curve', 'flat')
            fade = params.get('fade', 0.0)
        except KeyError:
            print(f"Warning: Incomplete metadata for plant ID {wf}. Skipping...")
            continue

        df_wf = df[df['ID'] == wf].copy()
        if not pd.api.types.is_datetime64_any_dtype(df_wf['LocalTime']):
            df_wf['LocalTime'] = pd.to_datetime(df_wf['LocalTime'])
        df_wf.sort_values('LocalTime', inplace=True)

        # Standardize u10 and v10 column names.
        if 'u10' not in df_wf.columns and 'uas' in df_wf.columns:
            df_wf.rename(columns={'uas': 'u10'}, inplace=True)
        if 'v10' not in df_wf.columns and 'vas' in df_wf.columns:
            df_wf.rename(columns={'vas': 'v10'}, inplace=True)
        df_wf.dropna(subset=['u10', 'v10'], inplace=True)

        # Determine the original temporal resolution.
        dt = df_wf['LocalTime'].diff().dropna().min()

        # Interpolate to the desired interval if needed.
        if dt > interval_timedelta:
            new_index = pd.date_range(start=df_wf['LocalTime'].min(),
                                      end=df_wf['LocalTime'].max(),
                                      freq=interval)
            time_numeric = df_wf['LocalTime'].astype(np.int64) // 10 ** 9
            new_time_numeric = new_index.astype(np.int64) // 10 ** 9

            # Use Hermite (PCHIP) interpolation if original resolution is coarser than hourly,
            # otherwise use CubicSpline method.
            if dt > pd.Timedelta("1h"):
                cs_u = PchipInterpolator(time_numeric, df_wf['u10'].values)
                cs_v = PchipInterpolator(time_numeric, df_wf['v10'].values)
            else:
                cs_u = CubicSpline(time_numeric, df_wf['u10'].values)
                cs_v = CubicSpline(time_numeric, df_wf['v10'].values)

            new_u10 = cs_u(new_time_numeric)
            new_v10 = cs_v(new_time_numeric)
            df_interp = pd.DataFrame({'LocalTime': new_index, 'u10': new_u10, 'v10': new_v10})
            extra_cols = [col for col in df_wf.columns if col not in ['LocalTime', 'u10', 'v10']]
            for col in extra_cols:
                df_interp[col] = df_wf[col].iloc[0]
            df_wf = df_interp

        # Step 1: Compute wind speed at 10 m.
        df_wf['wind_speed_10m'] = compute_wind_speed(df_wf['u10'].values, df_wf['v10'].values)

        # Step 2: Extrapolate from 10 m to 100 m.
        df_wf['v_100m'] = extrapolate_wind_speed(df_wf['wind_speed_10m'].values, 100, alpha=alpha_10_100)

        # Step 3: Compute low-resolution Weibull parameters.
        mean_lr = df_wf['v_100m'].mean()
        std_lr = df_wf['v_100m'].std()
        if dt > pd.Timedelta("1h"):
            # Recalibrate via fitting when using coarser (e.g., 3-hourly) data.
            k_lr, loc, c_lr = weibull_min.fit(df_wf['v_100m'].values, floc=0)
        else:
            k_lr = (std_lr / mean_lr) ** (-1.086) if mean_lr > 0 else 1.0
            c_lr = mean_lr / gamma(1 + 1 / k_lr) if k_lr != 0 else mean_lr

        # Step 4: Extract high-resolution Weibull parameters from TIFFs (if provided).
        if weibull_scale_tif is not None and weibull_shape_tif is not None:
            c_hr, k_hr = get_weibull_params(lon_plant, lat_plant, weibull_scale_tif, weibull_shape_tif)
        else:
            c_hr, k_hr = c_lr, k_lr

        # Step 5: Downscale the 100 m wind speed using Weibull matching.
        df_wf['v_downscaled_100m'] = c_hr * (df_wf['v_100m'].values / c_lr) ** (k_lr / k_hr)

        # Step 6: Extrapolate from 100 m to the turbine's hub height.
        df_wf['v_hub'] = extrapolate_wind_speed(df_wf['v_downscaled_100m'].values, hub_height_metadata,
                                                alpha=alpha_100_hub)

        # --- New Step: Apply wake/internal losses ---
        # Reduce the effective wind speed such that the incoming wind energy is reduced
        # by the factor (1 - p_loss). Since power ~ v^3, we scale v_hub by (1 - p_loss)^(1/3)
        effective_v_hub = df_wf['v_hub'].values * ((1 - p_loss) ** (1 / 3))

        # Step 7: Compute power output using the turbine power curve.
        P_rated_per_turbine = P_rated / num_turbines
        power_per_turbine = turbine_power_output(
            effective_v_hub,
            P_rated_per_turbine,
            v_cut_in=v_cut_in,
            v_rated=v_rated,
            v_cut_out=v_cut_out,
            curve_type=curve_type,
            fade=fade
        )
        total_power_MW = power_per_turbine * num_turbines

        # --- Apply transmission/external losses ---
        # Multiply the power output by (1 - t_loss) to represent losses in transformer, lines, etc.
        df_wf['power_kWh'] = total_power_MW * 1000 * interval_hours * (1 - t_loss)

        results.append(df_wf)

    return pd.concat(results, ignore_index=True)

def extract_windfarm_metadata(df):
    """
    Extract distinct wind farm metadata.
    Expected columns: 'ID', 'Total Capacity (MW)', 'Number of Turbines',
    'Hub Height (m)', 'Turbine Model', 'longitude', 'latitude'
    """
    metadata_columns = ['ID', 'Total Capacity (MW)', 'Number of Turbines',
                        'Hub Height (m)', 'Turbine Model', 'longitude', 'latitude']
    return df[metadata_columns].drop_duplicates().reset_index(drop=True)

def generate_wind_predictions(input_path, output_path, interval='15min',
                              alpha_10_100=1/7, alpha_100_hub=1/7,
                              weibull_scale_tif=None, weibull_shape_tif=None,
                              p_loss=0.05, t_loss=0.0):
    """
    Processes all parquet files in 'input_path' whose filenames start with 'Wind'.
    For each file, it:
      - Reads and interpolates the wind data.
      - Computes 10 m wind speed and extrapolates to 100 m.
      - Computes low-res Weibull parameters from the 100 m wind speeds.
      - Extracts high-res Weibull parameters (from TIFFs) based on plant location.
      - Downscales the 100 m wind speed using Weibull matching.
      - Extrapolates the downscaled wind speed to the turbine's actual hub height.
      - Applies a wake/internal loss correction (p_loss) and further transmission/external loss (t_loss).
      - Computes turbine power output using a simplified power curve.
      - Exports results as both parquet and Excel files.
    """
    os.makedirs(output_path, exist_ok=True)
    pattern = os.path.join(input_path, "Wind*.parquet")
    file_list = sorted(glob.glob(pattern))

    for file in file_list:
        base_filename = os.path.basename(file).replace('.parquet', '')
        output_parquet = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}.parquet")
        output_excel = os.path.join(output_path, f"{base_filename}_w_predictions_{interval}_aggregated.xlsx")

        if os.path.exists(output_parquet) and os.path.exists(output_excel):
            continue

        df = pd.read_parquet(file, engine="fastparquet")
        df['LocalTime'] = pd.to_datetime(df['LocalTime'])
        df.sort_values('LocalTime', inplace=True)
        results = process_windfarm_outputs(
            df,
            interval=interval,
            alpha_10_100=alpha_10_100,
            alpha_100_hub=alpha_100_hub,
            weibull_scale_tif=weibull_scale_tif,
            weibull_shape_tif=weibull_shape_tif,
            p_loss=p_loss,
            t_loss=t_loss
        )
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_parquet = os.path.join(output_path, f"{base_name}_w_predictions_{interval}.parquet")
        results.to_parquet(output_parquet, index=False)
        results['LocalTime'] = results['LocalTime'].dt.tz_localize(None)
        aggregate_power_output_to_excel(results, output_excel)
        print(f"Exported: {output_parquet}")