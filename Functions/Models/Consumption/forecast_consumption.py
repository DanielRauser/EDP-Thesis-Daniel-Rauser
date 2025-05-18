import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import mlflow
import joblib
from Functions.Models.Consumption.helpers import (
    load_and_preprocess_data,
    log_fig,
    load_and_prepare_weather,
    create_time_features,
    train_point_and_quantile_models
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def forecast_consumption(
        redes_data_path, input_path, final_year,
        interval='15min', output_path='/Users/darcor/Data/EDP Thesis/Output/Predictions', quantiles=None
):
    """
    Forecast pipeline: train & evaluate once on baseline, then apply forecasts across all climate scenarios.

    Models are trained on baseline weather. Evaluation (with CI) is done once for baseline.
    Forecast horizons (baseline + RCPs) are then generated using the pretrained models.
    """
    logger.info('Starting forecasting pipeline...')
    # prepare weather scenarios
    weather = load_and_prepare_weather(input_path)
    # load consumption data
    df_act, df_clipped = load_and_preprocess_data(redes_data_path, 'Consumption (kWh)')
    df_clipped = df_clipped[df_clipped.index >= '2020-01-01']
    dt_max = df_clipped.index.max()
    hold = dt_max - pd.DateOffset(years=2)
    df_train = df_clipped[df_clipped.index <= hold]
    df_eval  = df_clipped[df_clipped.index > hold]
    years    = sorted(df_clipped.index.year.unique())
    if quantiles is None:
        quantiles = [q/100 for q in range(5,100,10)]
    # train and evaluate on baseline
    logger.info('Training and evaluating on baseline scenario')
    base_wdf = weather['baseline']
    base_dir = os.path.join(output_path, 'Forecast')
    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(output_path, 'quantile_model')
    os.makedirs(model_dir, exist_ok=True)
    # train on baseline
    train_point_and_quantile_models(
        df_train, years, model_dir, "Consumption (kWh)",
        quantiles, weather_df=base_wdf
    )
    # load models and features
    pm = joblib.load(os.path.join(model_dir, 'point_model.pkl'))
    feat_cols = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    q_models = {
        q: joblib.load(os.path.join(model_dir, f'quantile_{int(q * 100):03d}.pkl'))
        for q in quantiles
    }
    # evaluate with CI on baseline
    X_e = create_time_features(df_eval, years, base_wdf)
    X_e = X_e.reindex(columns=feat_cols, fill_value=0)
    y_e = df_eval['Consumption (kWh)']
    y_pt_e = pm.predict(X_e)
    Xq_e = X_e.copy(); Xq_e['point_forecast'] = y_pt_e
    df_q = pd.DataFrame({f'{int(q*100)}%': m.predict(Xq_e)
                         for q,m in q_models.items()}, index=y_e.index)
    lower = df_q['5%'] + y_pt_e
    upper = df_q['95%'] + y_pt_e
    median = ((df_q['45%'] + df_q['55%'])/2 + y_pt_e)
    bias = (y_e - median).mean()
    # plot evaluation
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1) draw CI first, in light gray
    ax.fill_between(
        y_e.index,
        (lower + bias) / 1e6,
        (upper + bias) / 1e6,
        color='lightgray',
        alpha=0.4,
        label='90% CI',
        zorder=1
    )

    # 2) then draw the actual and median on top
    ax.plot(
        y_e.index,
        y_e / 1e6,
        color='tab:blue',
        label='Actual',
        zorder=2
    )
    ax.plot(
        y_e.index,
        (median + bias) / 1e6,
        linestyle='--',
        color='tab:orange',
        label='Median',
        zorder=3
    )

    ax.set(
        xlabel='Time',
        ylabel='GWh'
    )
    ax.legend()
    fig.tight_layout()
    log_fig(fig, 'baseline/eval_ci.png')

    # compute evaluation metrics (RMSE, MAE, MAPE)
    # replace zeros for MAPE stability
    non_zero_mean = y_e[y_e != 0].mean()
    y_e_for_mape = y_e.replace(0, non_zero_mean)
    median_corr = median + bias
    rmse = np.sqrt(((median_corr - y_e) ** 2).mean())
    mae  = np.abs(median_corr - y_e).mean()
    mape = (np.abs((median_corr - y_e_for_mape) / y_e_for_mape).mean()) * 100
    mlflow.log_metric('eval_rmse', rmse)
    mlflow.log_metric('eval_mae', mae)
    mlflow.log_metric('eval_mape', mape)

    # define growth profile
    growth_profile = [
        (2025,2030,0.005),(2030,2035,0.01),(2035,2037,0.0075),
        (2037,2041,0.005),(2041,2046,0.00025)
    ]
    def _grow(year):
        f = 1
        for y in range(dt_max.year+1, year+1):
            for s,e,r in growth_profile:
                if s <= y < e:
                    f *= 1 + r; break
        return f

    # forecast horizon: apply pretrained models to each scenario
    for scen, wdf in weather.items():
        if scen == 'baseline':
            continue
        for growth in ['baseline','growth']:
            scen_dir = os.path.join(output_path, 'Forecast', scen, growth)
            os.makedirs(scen_dir, exist_ok=True)
            logger.info(f'Forecasting {scen} / {growth}')
            # build future features
            idx_f = pd.date_range(dt_max + pd.Timedelta(interval),
                                  pd.to_datetime(f'{final_year}-12-31 23:59'),
                                  freq=interval)
            df_f = pd.DataFrame(index=idx_f)
            X_f = create_time_features(df_f, years + [final_year], wdf)
            X_f = X_f.reindex(columns=feat_cols, fill_value=0)
            # predict
            y_pt_f = pm.predict(X_f)
            Xq_f = X_f.copy(); Xq_f['point_forecast'] = y_pt_f
            df_rf = pd.DataFrame({f'{int(q*100)}%': m.predict(Xq_f)
                                  for q,m in q_models.items()}, index=idx_f)
            df_base = df_rf.add(y_pt_f, axis=0) + bias
            if growth == 'growth':
                factors = idx_f.year.map(_grow)
                df_base = df_base.mul(factors, axis=0)
            df_base['median'] = (df_base['45%'] + df_base['55%']) / 2

            df_base['year'] = df_base.index.year

            df_out = df_base.reset_index().rename(columns={'index': 'timestamp'})

            # save
            out_fname = f'consumption_quantile_{interval}_{scen}_{growth}.xlsx'
            out_path = os.path.join(base_dir, out_fname)

            df_out.to_excel(out_path, index=False)
            logger.info(f'Saved forecast → {out_path}')
    logger.info('Forecasting complete for all scenarios')

