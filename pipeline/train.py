import mlflow
import pandas as pd
import logging
import os
import joblib
from datetime import datetime
import tempfile
import pickle

from Functions.Preprocessing.preprocessor import Preprocessor, PreprocessorInferenceWrapper
from Functions.Optimizers.optimizers import Optimizer
from Functions.Models.customlightgbm import CustomLightGBM

from Functions.Evaluation.solar_eval import ModelEvaluator

logger = logging.getLogger(__name__)


def save_model(preprocessor, params, results):
    """
    Trains (if needed) and saves the final model using training data from the preprocessor.
    For neural networks, the model is assumed to be already trained and is extracted from results.
    For other models (e.g., LightGBM), a new model is instantiated and trained.
    """

    X_train, y_train = preprocessor.train_data
    normalize_power_output = params.get("normalize_power_output", True)
    # For non-neural network models, we drop the "capacity_MW" column if normalization is used.
    X_train_model = X_train.drop(columns=["capacity_MW"], errors="ignore") if normalize_power_output else X_train

    model_type = params.get("model")

    if model_type == "neural_network":
        # For neural networks, extract the already-trained model from results.
        final_model = results.get("model")
        if final_model is None:
            raise ValueError("Neural network model not found in results dictionary.")
    else:
        # For other models, use the best parameters from results to train a new model.
        best_params = results.get("best_params")
        final_model = CustomLightGBM(best_params=best_params, random_state=params["random_state"])
        final_model.fit(X_train_model, y_train)

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = params.get("output_path", ".")
    os.makedirs(output_path, exist_ok=True)
    model_filename = f"final_model_{current_date}.pkl"
    model_save_path = os.path.join(output_path, model_filename)
    joblib.dump(final_model, model_save_path)
    logger.info("Final model saved to: %s", model_save_path)
    return final_model, model_save_path

def log_preprocessor(preprocessor):
    # Create a temporary directory to save the pickled preprocessor
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "preprocessor.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(preprocessor, f)
        # Log the file as an MLflow artifact under a folder "preprocessor"
        mlflow.log_artifact(save_path, artifact_path="preprocessor")
        logger.info("Preprocessor saved to MLflow artifacts.")

def train_task(params: dict):
    logger.info("Conducting training & tuning of solar model")
    preprocessor = Preprocessor(
        target_variable=params["target_variable"],
        log_target_variable=params["log_target_variable"],
        predictors=params["predictors"],
        test_size=params["test_size"],
        group_variable = params["group_variable"],
        apply_filter=params["filter"],
        filter_variable=params["filter_variable"],
        lag_features=params["lag_features"],
        interaction_terms=params["interaction_terms"],
        normalize_power_output=params["normalize_power_output"],
        cv_folds=params["folds"],
        ml_model=params["model"],
        standardize=params["standardize"],
        random_state=params["random_state"]
    )

    logger.info("Importing Solar Train Data")
    input_path = params.get("input_path", ".")
    solar_data = pd.read_parquet(input_path + '/solar_data.parquet', engine='fastparquet')

    logger.info("Conducting Preprocessing")
    preprocessor.fit(solar_data)

    del solar_data

    inference_wrapper = PreprocessorInferenceWrapper(
                 scaler=preprocessor.scaler,
                 predictors=preprocessor.predictors,
                 standardize=preprocessor.standardize,
                 normalize_power_output=preprocessor.normalize_power_output,
                 log_target_variable=preprocessor.log_target_variable,
                 scaler_features=preprocessor.scaler_features,
                 lag_features=preprocessor.lag_features,
                 interaction_terms=preprocessor.interaction_terms
           )
    log_preprocessor(inference_wrapper)

    optimizer = Optimizer(config=params["optimizer"], preprocessor=preprocessor)
    results = optimizer.fit_transform()

    trained_model, model_path = save_model(preprocessor, params, results)

    mlflow.log_artifact(model_path, artifact_path="model")

    trained_model_summary = {
        "model_type": params.get("model"),
        "saved_model_path": model_path,
        "normalized RMSE": results["metrics"].get("norm_rmse"),
        "best_params": results["best_params"]
    }

    logger.info("Model training completed. Trained model summary:")
    for key, value in trained_model_summary.items():
        logger.info("%s: %s", key, value)

    evaluator = ModelEvaluator(model=trained_model, preprocessor=preprocessor)
    evaluation_metrics = evaluator.evaluate()

    # Build and log a summary
    trained_model_summary = {
        "rmse": evaluation_metrics["rmse"],
        "mae": evaluation_metrics["mae"],
        "r2": evaluation_metrics["r2"],
        "explained_variance": evaluation_metrics["explained_variance"],
        "normalized RMSE": evaluation_metrics["norm_rmse"],
        "normalized MAE": evaluation_metrics["norm_mae"],
    }

    logger.info("Model evaluation completed. Trained model summary on test set:")
    for key, value in trained_model_summary.items():
        logger.info("%s: %s", key, value)

    return trained_model