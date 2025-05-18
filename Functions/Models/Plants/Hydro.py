import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm

from Functions.Models.Consumption.helpers import (
    load_and_preprocess_data, log_fig, get_season,
    disaggregate_monthly_to_interval_mc, build_daily_profile,
    load_and_prepare_weather, train_point_and_quantile_models,
    create_time_features, add_exog_lags_and_roll, convert_scen_to_filename
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_hydro_forecast(
    redes_data_path,
    final_year=2045,
    num_simulations=10,
    noise_daily=0.02,
    daily_factor_clip=0.03,
    output_path="forecast_hydro"
):
    market_share = 0.60836228585
    interval = "15min"
    logger.info("Starting hydro forecast...")

    # Paths
    era5_file = os.path.join(output_path, "Hydro_era5_w_predictions_15min_aggregated.xlsx")
    scenarios = {
        "rcp_2_6": os.path.join(output_path, "Hydro_rcp_2_6_w_predictions_15min_aggregated.xlsx"),
        "rcp_4_5": os.path.join(output_path, "Hydro_rcp_4_5_w_predictions_15min_aggregated.xlsx"),
        "rcp_8_5": os.path.join(output_path, "Hydro_rcp_8_5_w_predictions_15min_aggregated.xlsx"),
    }
    os.makedirs(output_path, exist_ok=True)

    # 1. Load & preprocess
    df_actual, df_clipped = load_and_preprocess_data(
        redes_data_path, target_column="Hydro (kWh)",
        upper_clip=0.9, lower_clip=0.005
    )
    for df in (df_actual, df_clipped):
        df["Hydro (kWh)"] = pd.to_numeric(df["Hydro (kWh)"], errors='coerce') * market_share

    # 2. Write ERA5 history if missing
    if not os.path.exists(era5_file):
        mask = (df_actual.index >= "2020-01-01") & (df_actual.index <= "2025-03-31")
        df_e = df_actual.loc[mask].copy().drop(columns="LocalTime", errors="ignore")
        df_e["power_kWh"] = df_e["Hydro (kWh)"]
        df_e["Year"]      = df_e.index.year
        df_e["Month"]     = df_e.index.month
        df_e["Day"]       = df_e.index.day
        df_e["Hour"]      = df_e.index.hour
        df_e["LocalTime"] = df_e.index.tz_localize(None)
        df_e[["LocalTime", "power_kWh", "Year", "Month", "Day", "Hour"]] \
            .to_excel(era5_file, index=False)
        logger.info(f"ERA5 file created: {era5_file}")

    # Trim to ≥2015
    df_actual  = df_actual[df_actual.index >= "2015-01-01"]
    df_clipped = df_clipped[df_clipped.index >= "2015-01-01"]

    # ——————————————————————————————————————————————
    # Baseline: average of the last N full calendar years
    # ——————————————————————————————————————————————
    ann_totals = df_actual['Hydro (kWh)'].resample('YE').sum()
    valid      = ann_totals[ann_totals.index <= df_clipped.index.max()]
    if valid.empty:
        raise RuntimeError("Not enough data to define a baseline.")

    n_years        = 5
    baseline_years = valid if len(valid) < n_years else valid.iloc[-n_years:]
    annual_baseline= baseline_years.mean()
    last_full_year = valid.index[-1].year

    # Build monthly_props from those baseline years
    start_cal      = f"{last_full_year - n_years + 1}-01-01"
    end_cal        = f"{last_full_year}-12-31"
    monthly_series = df_clipped['Hydro (kWh)'].loc[start_cal:end_cal]
    monthly_sums   = (
        monthly_series
        .groupby([monthly_series.index.year, monthly_series.index.month])
        .sum()
    )
    monthly_sums.index.names = ['Year', 'Month']
    monthly_df     = monthly_sums.to_frame(name='sum').reset_index()
    monthly_prop   = monthly_df.groupby('Month')['sum'].mean()
    monthly_prop   = monthly_prop / monthly_prop.sum()


    def deterministic_hydro_series(baseline, index, monthly_props, hcf_by_year_and_season):
        seasons = [get_season(dt.month) for dt in index]
        hcfs    = np.array([
            hcf_by_year_and_season[dt.year][seasons[i]]
            for i, dt in enumerate(index)
        ])
        props   = np.array([monthly_props.loc[dt.month] for dt in index])
        base_m  = baseline * props
        return pd.Series(base_m * hcfs, index=index)


    def forecast_rcp(scenario):
        base_year = last_full_year
        horizon   = 2050

        # 2050 HCFs from your table (Projected HCF for 2050)
        no_cc  = {"Winter":0.627, "Spring":0.700, "Summer":0.650, "Fall":1.936}
        raw_B2a= {"Winter":1.013, "Spring":0.835, "Summer":0.621, "Fall":0.742}
        raw_45 = {"Winter":0.874, "Spring":0.782, "Summer":0.616, "Fall":0.810}
        raw_85 = {"Winter":0.638, "Spring":0.570, "Summer":0.450, "Fall":0.591}

        # Build 2050 ratios vs No_CC
        HCF2050 = {
            "rcp_2_6": {s: raw_B2a[s]/no_cc[s]   for s in no_cc},  # RV_B2a under RCP2.6
            "rcp_4_5": {s: raw_45[s]/no_cc[s]    for s in no_cc},
            "rcp_8_5": {s: raw_85[s]/no_cc[s]    for s in no_cc},
        }

        # Interpolate year-by-year from 1.0 @ base_year → ratio @ 2050
        years = np.arange(base_year, final_year+1)
        hcf_by_year = {}
        for yr in years:
            frac = (yr - base_year) / (horizon - base_year)
            hcf_by_year[yr] = {
                s: 1.0 + (HCF2050[scenario][s] - 1.0)*frac
                for s in no_cc
            }

        # Build month-end index
        start = df_clipped.index.max().normalize() + pd.Timedelta(days=1)
        idx   = pd.date_range(start=start, end=f"{final_year}-12-31", freq='ME')

        # Deterministic monthly series
        det     = deterministic_hydro_series(annual_baseline, idx, monthly_prop, hcf_by_year)
        profile = build_daily_profile(df_clipped, "Hydro (kWh)", interval)

        # Monte Carlo & disaggregate
        sims = []
        for _ in tqdm(range(num_simulations), desc=f"{scenario} sims"):
            noise   = np.random.uniform(-noise_daily, noise_daily, size=len(det))
            monthly = det * (1 + noise)
            sims.append(disaggregate_monthly_to_interval_mc(
                monthly, profile, noise_daily, daily_factor_clip, interval
            ))

        sims_df = pd.concat(sims, axis=1)
        sims_df.columns = [f"sim_{i}" for i in range(num_simulations)]
        low, high = sims_df.stack().quantile(0.05), sims_df.stack().quantile(0.95)
        sims_df = sims_df.clip(lower=low, upper=high)

        # Plot & save
        fig, ax = plt.subplots(figsize=(14,7))
        for col in sims_df:
            ax.plot(sims_df.index, sims_df[col]/1e6, color='gray', alpha=0.1)
        med, mean = sims_df.median(axis=1), sims_df.mean(axis=1)
        ax.plot(med.index, med/1e6, label="Median", linewidth=2)
        ax.plot(mean.index, mean/1e6, label="Mean", linewidth=2)
        ax.fill_between(
            med.index,
            sims_df.quantile(0.05,axis=1)/1e6,
            sims_df.quantile(0.95,axis=1)/1e6,
            alpha=0.2, label="5%-95%"
        )
        ax.set_title(f"{interval} Hydro Forecast {scenario.upper()} until {final_year}")
        ax.set_xlabel("Time"); ax.set_ylabel("Hydro (GWh)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        log_fig(fig, f"forecast_hydro_{scenario}.png")

        # Build output table
        out = pd.DataFrame({"power_kWh": med}, index=med.index)
        out = out.drop(columns="LocalTime", errors="ignore")
        out["LocalTime"] = out.index
        out["Year"]      = out.index.year
        out["Month"]     = out.index.month
        out["Day"]       = out.index.day
        out["Hour"]      = out.index.hour

        return out[["LocalTime","power_kWh","Year","Month","Day","Hour"]]


    # Run all scenarios
    for sc, path in scenarios.items():
        if os.path.exists(path):
            logger.info(f"{sc} exists, skipping.")
            continue
        df_fc = forecast_rcp(scenario=sc)
        df_fc.to_excel(path, index=False)
        logger.info(f"Saved {sc} forecast to {path}")


def generate_hydro_forecast_ml(
    redes_data_path,
    input_path,
    final_year=2045,
    interval='15min',
    quantiles=None,
    resid_split_ratio=0.2,
    output_path='forecast_hydro_ml'
):
    os.makedirs(output_path, exist_ok=True)
    forecast_path = os.path.join(output_path, "Forecast")
    os.makedirs(forecast_path, exist_ok=True)

    if quantiles is None:
        quantiles = [q / 100 for q in range(5, 100, 10)]

    # 1) load data
    weather      = load_and_prepare_weather(input_path)
    base_weather = weather['baseline']
    df_act, _    = load_and_preprocess_data(redes_data_path,
                                            target_column='Hydro (kWh)')
    df_act.index = pd.to_datetime(df_act.index)

    # bring actual consumption into base_weather
    base_weather = base_weather.copy()
    base_weather['consumption'] = df_act['Consumption (kWh)']

    # load baseline wind & PV predictions for era5
    for src in ['Wind', 'Photovoltaic']:
        fn = f"{src}_era5_w_predictions_15min_aggregated.xlsx"
        df_ex = pd.read_excel(os.path.join(output_path, fn), engine='openpyxl')
        df_ex['LocalTime'] = pd.to_datetime(df_ex['LocalTime'])
        df_ex.set_index('LocalTime', inplace=True)
        base_weather[f"{src.lower()}_power"] = df_ex['power_kWh']

    # 2) filter to recent era and convert hydro units
    df_act = df_act.loc[df_act.index >= '2020-01-01']
    df_act['Hydro (kWh)'] = pd.to_numeric(df_act['Hydro (kWh)'],
                                          errors='coerce') * 0.60836228585

    # 3) train/eval split
    dt_max   = df_act.index.max()
    hold     = dt_max - pd.DateOffset(years=2)
    df_train = df_act.loc[:hold]
    df_eval  = df_act.loc[hold + pd.Timedelta(interval):]
    years    = sorted(df_act.index.year.unique())

    # 4) train quantile models with exogenous features via extended base_weather
    model_dir = os.path.join(output_path, 'quantile_model')
    train_point_and_quantile_models(
        df_train, years, model_dir,
        target_col='Hydro (kWh)',
        quantiles=quantiles,
        weather_df=base_weather,
        resid_split_ratio=resid_split_ratio
    )

    # 5) load trained models & feature columns
    pm        = joblib.load(os.path.join(model_dir, 'point_model.pkl'))
    feat_cols = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    q_models  = {
        q: joblib.load(os.path.join(
            model_dir, f'quantile_{int(q*100):03d}.pkl'))
        for q in quantiles
    }

    # 6) evaluation: build full eval features including exogenous lags
    max_lag    = max([1,4,96] + [4,96,672])
    hist       = df_train.iloc[-max_lag:]
    df_eval_full = pd.concat([hist, df_eval])

    # create baseline features for evaluation
    X_e_full = create_time_features(df_eval_full,
                                    years,
                                    base_weather)
    # hydro lags/roll
    X_e_full = add_exog_lags_and_roll(X_e_full,
                                      df_eval_full['Hydro (kWh)'],
                                      prefix="hydro_")
    # exogenous lags/roll: wind, pv, consumption
    for var in ['wind_power', 'photovoltaic_power', 'consumption']:
        X_e_full = add_exog_lags_and_roll(
            X_e_full,
            base_weather[var].reindex(df_eval_full.index),
            prefix=f"{var}_"
        )

    X_e = X_e_full.loc[df_eval.index].reindex(columns=feat_cols,
                                              fill_value=0)
    y_e = df_eval['Hydro (kWh)']
    y_pt_e = np.clip(pm.predict(X_e), 0, None)

    Xq_e = X_e.copy()
    Xq_e['point_forecast'] = y_pt_e
    df_q = pd.DataFrame(
        {int(q*100): m.predict(Xq_e) for q, m in q_models.items()},
        index=y_e.index
    )

    lower  = np.clip(df_q[5]  + y_pt_e, 0, None)
    upper  = np.clip(df_q[95] + y_pt_e, 0, None)
    median = np.clip(((df_q[45] + df_q[55]) / 2) + y_pt_e, 0, None)
    bias   = (y_e - median).mean()

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(y_e.index, (lower + bias) / 1e6, (upper + bias) / 1e6, alpha=0.3, label='90% CI')
    ax.plot(y_e.index, y_e / 1e6, label='Actual')
    ax.plot(y_e.index, (median + bias) / 1e6, linestyle='--', label='Median')
    ax.set(xlabel='Time', ylabel='Hydro (GWh)')
    ax.legend()
    fig.tight_layout()
    log_fig(fig, 'baseline_hydro_eval.png')

    # metrics
    non_zero_mean = y_e[y_e != 0].mean()
    y_e_for_mape = y_e.replace(0, non_zero_mean)
    median_corr = median + bias

    mlflow.log_metric('eval_rmse', np.sqrt(((median_corr - y_e) ** 2).mean()))
    mlflow.log_metric('eval_mae', np.abs(median_corr - y_e).mean())
    mlflow.log_metric('eval_mape', (np.abs((median_corr - y_e_for_mape) / y_e_for_mape).mean()) * 100)

    # HCF interpolation setup (unchanged)...
    no_cc = {'Winter': 0.627, 'Spring': 0.700, 'Summer': 0.650, 'Fall': 1.936}
    raw_B2a = {'Winter': 1.013, 'Spring': 0.835, 'Summer': 0.621, 'Fall': 0.742}
    raw_45 = {'Winter': 0.874, 'Spring': 0.782, 'Summer': 0.616, 'Fall': 0.810}
    raw_85 = {'Winter': 0.638, 'Spring': 0.570, 'Summer': 0.450, 'Fall': 0.591}
    #HCF2050 = {
    #    'rcp26': {s: raw_B2a[s] / no_cc[s] for s in no_cc},
    #    'rcp45': {s: raw_45[s] / no_cc[s] for s in no_cc},
    #    'rcp85': {s: raw_85[s] / no_cc[s] for s in no_cc},
    #}

    HCF2050 = {
        'rcp26': {'Winter': 1.013, 'Spring': 0.835, 'Summer': 0.621, 'Fall': 0.742},
        'rcp45': {'Winter': 0.874, 'Spring': 0.782, 'Summer': 0.616, 'Fall': 0.81},
        'rcp85': {'Winter': 0.638, 'Spring': 0.570, 'Summer': 0.450, 'Fall': 0.591},
    }
    base_year = dt_max.year
    horizon = 2050
    hcf_by_year = {
        yr: {
            sc: {
                season: 1.0 + (HCF2050[sc][season] - 1.0) *
                        ((yr - base_year) / (horizon - base_year))
                for season in no_cc
            }
            for sc in HCF2050
        }
        for yr in range(base_year, horizon + 1)
    }

    wind_forecasts, pv_forecasts, cons_forecasts = {}, {}, {}
    for scen in weather:
        if scen == 'baseline': continue
        # VRE
        for src, d in [('Wind', wind_forecasts), ('Photovoltaic', pv_forecasts)]:
            scen_file = convert_scen_to_filename(scen)
            fn = f"{src}_{scen_file}_w_predictions_15min_aggregated.xlsx"
            df_ex = pd.read_excel(os.path.join(output_path, fn), engine='openpyxl')
            df_ex['LocalTime'] = pd.to_datetime(df_ex['LocalTime'])
            df_ex.set_index('LocalTime', inplace=True)
            d[scen] = df_ex['power_kWh']
        # consumption
        fnc = f"consumption_quantile_15min_{scen}_baseline.xlsx"
        df_c = pd.read_excel(os.path.join(forecast_path, fnc), engine='openpyxl')
        df_c['timestamp'] = pd.to_datetime(df_c['timestamp'])
        df_c.set_index('timestamp', inplace=True)
        cons_forecasts[scen] = df_c['median']

    # 8) forecast horizon for each scenario
    future_idx = pd.date_range(
        dt_max + pd.Timedelta(interval),
        pd.to_datetime(f'{final_year}-12-31 23:59'),
        freq=interval
    )
    # seasonal/climate factors omitted for brevity…

    for scen in weather:
        if scen == 'baseline':
            continue

        # meteorological features for this scenario
        weather_df = weather[scen].copy()
        weather_df.index = pd.to_datetime(weather_df.index)
        ex_df = pd.DataFrame(index=future_idx)
        ex_df['wind_power']       = wind_forecasts[scen].reindex(future_idx, method='nearest')
        ex_df['photovoltaic_power']= pv_forecasts[scen].reindex(future_idx, method='nearest')
        ex_df['consumption']      = cons_forecasts[scen].reindex(future_idx, method='nearest')

        # 8a) build feature matrix
        X_f = create_time_features(pd.DataFrame(index=future_idx),
                                   years + [final_year],
                                   weather_df)
        # hydro lags (using last known hydro)
        hist_full = pd.concat([df_train, df_eval])
        X_f = add_exog_lags_and_roll(X_f,
                                     hist_full['Hydro (kWh)'].reindex(
                                         future_idx.union(hist_full.index)
                                     ).loc[future_idx],
                                     prefix="hydro_")
        # exogenous lags/roll
        for var in ['wind_power', 'photovoltaic_power', 'consumption']:
            X_f = add_exog_lags_and_roll(X_f,
                                         ex_df[var],
                                         prefix=f"{var}_")

        X_f = X_f.reindex(columns=feat_cols, fill_value=0)

        # 8b) point & quantile predictions
        y_pt_f = np.clip(pm.predict(X_f), 0, None)
        Xq_f   = X_f.copy(); Xq_f['point_forecast'] = y_pt_f
        df_q_f = pd.DataFrame(
            {int(q*100): m.predict(Xq_f)
             for q, m in q_models.items()},
            index=future_idx
        )

        # 8c) apply seasonal/climate adjustment factors…
        # (as in original: HCF interpolation per scenario & year)

        # 8d) save to Excel
        rows = []
        for ts in future_idx:
            yr     = ts.year
            season= get_season(ts.month)
            factor= hcf_by_year[yr][scen][season]
            row={'timestamp': ts}
            for q in quantiles:
                p=int(q*100)
                val=(df_q_f[p].loc[ts] + y_pt_f[
                    df_q_f.index.get_loc(ts)
                ]) * factor
                row[f'{p}%']= max(0, val)
            row['median']=(row['45%'] + row['55%'])/2
            rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out['year']= df_out['timestamp'].dt.year
        fname = f'hydro_forecast_ml_{scen}_{interval}.xlsx'
        df_out.to_excel(os.path.join(output_path, fname), index=False)
        logger.info(f"Saved ML hydro forecast for {scen} → {fname}")

    logger.info('ML hydro forecasting complete')