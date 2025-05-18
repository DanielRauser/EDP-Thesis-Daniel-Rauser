import os
import mlflow
import pickle
import joblib
import logging

from Functions.Models.Plants.Solar import generate_solar_predictions
from Functions.Models.Plants.Wind import generate_wind_predictions
from Functions.Models.Plants.Hydro import generate_hydro_forecast, generate_hydro_forecast_ml
from Functions.Models.Consumption.forecast_consumption import forecast_consumption
from Functions.Apply.generate_analysis import generate_analysis_data

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

    final_year = params["forecast"].get("final_year")

    solar_preprocessor = load_preprocessor(run_id)
    solar_model = load_model(run_id)

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


    # --- 2) Generate Consumption Forecast

    forecast_consumption(redes_data_path=redes_path,
                         input_path=input_path,
                         final_year=final_year,
                         interval=interval,
                         output_path=output_path)

    ## 2.2 Hydro

    generate_hydro_forecast_ml(redes_data_path=redes_path,
                               final_year=final_year,
                               output_path=output_path,
                               input_path=input_path)

    # --- 3) Generate Reanalysis & Forecasting Data ---
    generate_analysis_data(input_path=input_path,
                           redes_data_path=redes_path,
                           output_path=output_path,
                           interval=interval
                        )






