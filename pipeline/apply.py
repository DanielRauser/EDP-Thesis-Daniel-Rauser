import os
import mlflow
import pickle
import joblib
import logging

from Functions.Models.Plants.Solar import generate_solar_predictions
from Functions.Models.Plants.Wind import generate_wind_predictions
from Functions.Models.Plants.Hydro import generate_hydro_predictions
from Functions.Models.Consumption.forecast_consumption import forecast_consumption
from Functions.Models.Plants.helpers import generate_analysis_data

logger = logging.getLogger(__name__)

def load_preprocessor(run_id):
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor/preprocessor.pkl")
    with open(artifact_path, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor


def load_model(run_id):
    """Loads the model file that starts with 'final_model' from MLflow artifacts."""
    try:
        # Download the model directory
        model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/")

        # Find the model file that starts with "final_model"
        model_file = next((f for f in os.listdir(model_dir) if f.startswith("final_model")), None)
        if not model_file:
            raise FileNotFoundError("No model file found with the pattern 'final_model'.")

        model_path = os.path.join(model_dir, model_file)

        # Use joblib.load to load the model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def apply_task(params: dict):
    run_id = params["solar"].get("run_id")
    input_path = params.get("input_path")
    output_path = params.get("output_path")
    redes_path = params.get("redes_path")
    interval = params.get("interval")

    solar_preprocessor = load_preprocessor(run_id)
    solar_model = load_model(run_id)

    # --- 1) Generate Predictions ---

    ## 1.1 Solar
    generate_solar_predictions(preprocessor=solar_preprocessor,
                               model=solar_model,
                               input_path=input_path,
                               output_path=output_path,
                               interval=interval)

    ## 1.2 Wind
    generate_wind_predictions(input_path=input_path,
                              output_path=output_path,
                              interval=interval,
                              weibull_scale_tif=params["wind"].get("weibull_scale_tif"),
                              weibull_shape_tif=params["wind"].get("weibull_shape_tif"))

    ## 1.3 Hydro
    generate_hydro_predictions(input_path=input_path,
                               output_path=output_path,
                               interval=interval)

    # --- 2) Generate Consumption Forecast
    final_year = params["forecast"].get("final_year")
    growth_rates = params["forecast"].get("growth_rates")
    num_simulations = params["forecast"].get("num_mc_simulations")
    daily_noise = params["forecast"].get("daily_noise")
    daily_factor_clip = params["forecast"].get("daily_factor_clip")

    consumption_forecast = forecast_consumption(redes_data_path=redes_path,
                                                final_year=final_year,
                                                growth_rates=growth_rates,
                                                interval=interval,
                                                num_simulations=num_simulations,
                                                noise_daily=daily_noise,
                                                daily_factor_clip=daily_factor_clip,
                                                output_path=output_path)

    # --- 3) Generate Reanalysis & Forecasting Data ---
    reanalysis_data, forecast_data = generate_analysis_data(input_path=input_path,
                                                            redes_data_path=redes_path,
                                                            output_path=output_path,
                                                            interval=interval
                                                            )

    # ---4) Merge Consumption Forecast to Forecasted Data
    forecast_data.merge(consumption_forecast, on=["LocalTime"], how="left")

    forecast_data.to_excel(os.path.join(output_path, f"Forecasts_{interval}.xlsx"), index=False)






