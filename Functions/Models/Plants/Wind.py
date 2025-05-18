import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import gamma
from scipy.stats import weibull_min
import logging

from Functions.Models.Plants.helpers import aggregate_power_output_to_excel, get_weibull_params


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Turbine parameters (unchanged)
# ----------------------------------------------------------------------------
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
    return np.sqrt(u**2 + v**2)

def extrapolate_wind_speed(v_ref, target_height, alpha=1/7):
    return v_ref * (target_height / 10.0) ** alpha

def turbine_power_output(v_hub, P_rated, v_cut_in=3.5, v_rated=12.5,
                         v_cut_out=25.5, curve_type="flat", fade=0.0):
    power = np.zeros_like(v_hub)
    mask1 = (v_hub >= v_cut_in) & (v_hub < v_rated)
    power[mask1] = P_rated * ((v_hub[mask1] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    mask2 = (v_hub >= v_rated) & (v_hub < v_cut_out)
    if curve_type == "fade_out":
        frac = (v_hub[mask2] - v_rated) / (v_cut_out - v_rated)
        power[mask2] = P_rated * (1 - fade * frac)
    else:
        power[mask2] = P_rated
    return power

def temporal_weibull_downscale(v_coarse, coarse_hours, fine_hours=0.25, n = None):
    """
    Downscale sparse coarse (e.g., 6 h) wind-speed samples to finer resolution
    (e.g., 1 h or 0.25 h) by fitting the scaling law of Weibull mean vs. temporal scale.
    """
    scales = np.array([coarse_hours, 3, 1, fine_hours])
    raw_means = []
    series = pd.Series(v_coarse)
    for h in scales:
        if h == coarse_hours:
            vals = series.values
        else:
            window = max(1, int(h / coarse_hours * len(series)))
            vals = series.rolling(window, min_periods=1).mean().dropna().values
        raw_means.append(np.mean(vals))
    coeffs = np.polyfit(np.log(scales), np.log(raw_means), 1)
    target_mean = np.exp(np.polyval(coeffs, np.log(fine_hours)))
    k, loc, c = weibull_min.fit(v_coarse, floc=0)
    scale_fine = target_mean / gamma(1 + 1 / k)
    if n is None:
        n = int(len(v_coarse) * coarse_hours / fine_hours)
    return weibull_min.rvs(k, scale=scale_fine, size=n)

def quantile_delta_mapping(x_mod, ref_obs, ref_mod_hist, method='multiplicative'):
    """
    Trend-preserving Quantile Delta Mapping (Cannon et al., 2015)
    """
    obs_sorted = np.sort(ref_obs)
    mod_sorted = np.sort(ref_mod_hist)
    q = np.interp(x_mod, mod_sorted, np.linspace(0,1,len(mod_sorted)))
    obs_q = np.interp(q, np.linspace(0,1,len(obs_sorted)), obs_sorted)
    base = np.interp(q, np.linspace(0,1,len(mod_sorted)), mod_sorted)
    if method == 'additive':
        return obs_q + (x_mod - base)
    else:
        return obs_q * (x_mod / base)

def extract_windfarm_metadata(df):
    cols = ['ID', 'Total Capacity (MW)', 'Number of Turbines',
            'Hub Height (m)', 'Turbine Model', 'longitude', 'latitude']
    return df[cols].drop_duplicates().reset_index(drop=True)

def process_windfarm_outputs(df, era5_ref=None, interval='15min',
                             alpha_10_100=1/7, alpha_100_hub=1/7,
                             use_gwa_tifs=False,
                             weibull_scale_tif=None, weibull_shape_tif=None,
                             p_loss=0.1, t_loss=0.00):
    results = []
    meta = extract_windfarm_metadata(df).set_index('ID')
    interval_td = pd.Timedelta(interval)
    hours = interval_td.total_seconds() / 3600.0
    era5 = None

    # Prepare ERA5 reference series for QDM if provided
    if era5_ref is not None:
        era5 = era5_ref.copy()
        era5['LocalTime'] = pd.to_datetime(era5['LocalTime'])
        era5['wind_speed_10m'] = compute_wind_speed(
            era5.get('u10', era5.get('uas')),
            era5.get('v10', era5.get('vas'))
        )
        era5 = (
            era5[['LocalTime', 'wind_speed_10m']]
            .drop_duplicates(subset='LocalTime', keep='first')
            .set_index('LocalTime')
        )


    for wf_id in tqdm(meta.index, desc="Processing Windfarms"):
        m = meta.loc[wf_id]
        df_wf = df[df['ID'] == wf_id].copy()
        df_wf['LocalTime'] = pd.to_datetime(df_wf['LocalTime'])
        df_wf.sort_values('LocalTime', inplace=True)

        # Standardize wind components
        if 'u10' not in df_wf and 'uas' in df_wf:
            df_wf.rename(columns={'uas':'u10'}, inplace=True)
        if 'v10' not in df_wf and 'vas' in df_wf:
            df_wf.rename(columns={'vas':'v10'}, inplace=True)
        df_wf.dropna(subset=['u10','v10'], inplace=True)

        # Step 1: compute 10 m wind speed
        df_wf['wind_speed_10m'] = compute_wind_speed(df_wf['u10'], df_wf['v10'])

        # Step 2a: downscale to hourly for CORDEX before QDM
        dt0 = df_wf['LocalTime'].diff().min()
        if era5_ref is not None and not use_gwa_tifs and dt0 > pd.Timedelta('1h'):
            coarse_h = dt0.total_seconds() / 3600.0
            idx_hr = pd.date_range(df_wf['LocalTime'].min(),
                                   df_wf['LocalTime'].max(), freq='1h')
            ws_hr = temporal_weibull_downscale(
                df_wf['wind_speed_10m'].values, coarse_h, fine_hours=1, n=len(idx_hr))
            df_wf = pd.DataFrame({
                'LocalTime': idx_hr,
                'wind_speed_10m': ws_hr,
                'ID': wf_id,
                **{c: m[c] for c in [
                    'Total Capacity (MW)','Number of Turbines',
                    'Hub Height (m)','longitude','latitude'
                ]}
            })

        # Step 2b: Bias-correct vs. ERA5 using QDM (hourly sample for CORDEX, hourly unchanged for ERA5)
        if era5_ref is not None:
            hist = df_wf[(df_wf['LocalTime'].dt.year >= 2023) & (df_wf['LocalTime'].dt.year <= 2024)]
            if not hist.empty:
                ref_obs = era5.reindex(hist['LocalTime'], method='nearest')['wind_speed_10m'].values
                ref_mod = hist['wind_speed_10m'].values
                df_wf['wind_speed_10m'] = quantile_delta_mapping(
                    df_wf['wind_speed_10m'].values, ref_obs, ref_mod, method='multiplicative')

        # Step 3: Temporal downscaling/interpolation to target interval
        dt = df_wf['LocalTime'].diff().min()
        if dt > interval_td:
            coarse_h = dt.total_seconds() / 3600.0
            idx_fine = pd.date_range(df_wf['LocalTime'].min(),
                                     df_wf['LocalTime'].max(), freq=interval)
            ws_fine = temporal_weibull_downscale(
                df_wf['wind_speed_10m'].values, coarse_h, hours, n=len(idx_fine))
            df_wf = pd.DataFrame({
                'LocalTime': idx_fine,
                'wind_speed_10m': ws_fine,
                'ID': wf_id,
                **{c: m[c] for c in [
                    'Total Capacity (MW)','Number of Turbines',
                    'Hub Height (m)','longitude','latitude'
                ]}
            })

        # Step 4: vertical extrapolation to 100 m
        df_wf['v_100m'] = extrapolate_wind_speed(
            df_wf['wind_speed_10m'], 100, alpha=alpha_10_100)

        # Step 5: High-res Weibull parameters
        if use_gwa_tifs and weibull_scale_tif and weibull_shape_tif:
            c_hr, k_hr = get_weibull_params(
                m['longitude'], m['latitude'],
                weibull_scale_tif, weibull_shape_tif)
        else:
            k_hr, loc, c_hr = weibull_min.fit(df_wf['v_100m'].values, floc=0)

        # Step 6: local Weibull fit for matching
        mean_lr = df_wf['v_100m'].mean()
        std_lr  = df_wf['v_100m'].std()
        if dt0 > pd.Timedelta('1h'):
            k_lr, loc, c_lr = weibull_min.fit(df_wf['v_100m'].values, floc=0)
        else:
            k_lr = (std_lr / mean_lr)**(-1.086) if mean_lr>0 else 1.0
            c_lr = mean_lr / gamma(1 + 1/k_lr) if k_lr!=0 else mean_lr
        df_wf['v_downscaled_100m'] = c_hr * (df_wf['v_100m']/c_lr)**(k_lr/k_hr)

        # Step 7: vertical extrapolation to hub height
        df_wf['v_hub'] = extrapolate_wind_speed(
            df_wf['v_downscaled_100m'], m['Hub Height (m)'], alpha=alpha_100_hub)

        # Steps 8-10: losses, power curve, kWh
        eff_v = df_wf['v_hub'] * ((1 - p_loss)**(1/3))
        P_t = m['Total Capacity (MW)'] / m['Number of Turbines']
        params = get_turbine_parameters(m.get('Turbine Model','UNKNOWN'))
        pow_turb = turbine_power_output(
            eff_v, P_t,
            v_cut_in=params['v_cut_in'],
            v_rated=params['v_rated'],
            v_cut_out=params['v_cut_out'],
            curve_type=params.get('curve','flat'),
            fade=params.get('fade',0.0))
        total_MW = pow_turb * m['Number of Turbines']
        df_wf['power_kWh'] = total_MW * 1000 * hours * (1 - t_loss)

        results.append(df_wf)

    return pd.concat(results, ignore_index=True)

def generate_wind_predictions(input_path, output_path,
                              weibull_scale_tif=None, weibull_shape_tif=None,
                              interval='15min', **kwargs):
    os.makedirs(output_path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_path, "Wind*.parquet")))
    if not files:
        logger.warning("No Wind*.parquet files found.")
        return

    # 1) ERA5 (hourly -> 15min with GWA)
    era5_file = files[0]
    era5_ref = pd.read_parquet(era5_file, engine="fastparquet")
    df_era5 = era5_ref.copy()
    df_era5.sort_values('LocalTime', inplace=True)
    base = os.path.splitext(os.path.basename(era5_file))[0]
    out_pq = os.path.join(output_path, f"{base}_w_predictions_{interval}.parquet")
    out_xl = os.path.join(output_path, f"{base}_w_predictions_{interval}_aggregated.xlsx")
    if not (os.path.exists(out_pq) and os.path.exists(out_xl)):
        results_era5 = process_windfarm_outputs(
            df_era5, era5_ref=None, interval=interval,
            use_gwa_tifs=True,
            weibull_scale_tif=weibull_scale_tif,
            weibull_shape_tif=weibull_shape_tif,
            p_loss=0.1, **kwargs)
        results_era5.to_parquet(out_pq, index=False)
        results_era5['LocalTime'] = results_era5['LocalTime'].dt.tz_localize(None)
        aggregate_power_output_to_excel(results_era5, out_xl)
        print(f"Exported ERA5 outputs: {out_pq}")

    # 2) CORDEX (6h -> 1h -> QDM -> 15min)
    for file in files[1:]:
        base = os.path.splitext(os.path.basename(file))[0]
        out_pq = os.path.join(output_path, f"{base}_w_predictions_{interval}.parquet")
        out_xl = os.path.join(output_path, f"{base}_w_predictions_{interval}_aggregated.xlsx")
        if os.path.exists(out_pq) and os.path.exists(out_xl):
            continue

        df = pd.read_parquet(file, engine="fastparquet")
        df.sort_values('LocalTime', inplace=True)
        results = process_windfarm_outputs(
            df, era5_ref=era5_ref, interval=interval,
            use_gwa_tifs=False,
            weibull_scale_tif=None, weibull_shape_tif=None,
            p_loss=0.1, **kwargs)
        results = results[(results['LocalTime'].dt.year > 2024) & (results['LocalTime'].dt.year < 2046)]
        results.to_parquet(out_pq, index=False)
        results['LocalTime'] = results['LocalTime'].dt.tz_localize(None)
        aggregate_power_output_to_excel(results, out_xl)
        print(f"Exported CORDEX outputs: {out_pq}")