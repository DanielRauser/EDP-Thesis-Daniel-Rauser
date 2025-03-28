import logging
import mlflow
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from hyperopt import Trials, tpe, atpe, hp, fmin

from Functions.Models.customlightgbm import CustomLightGBM
from Functions.Models.dense import CustomNeuralNetwork

logger = logging.getLogger(__name__)

from tensorflow.keras import callbacks as keras_callbacks

class MLflowLoggingCallback(keras_callbacks.Callback):
    """
    A custom Keras callback to log training progress to MLflow at the end of each epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Log each metric with the epoch number as the step.
            for metric, value in logs.items():
                mlflow.log_metric(metric, value, step=epoch)

#########################################
# Early Stopping with Epsilon Callback
#########################################
def early_stopping_with_epsilon(stopping_rounds, epsilon=0.001, verbose=False):
    """
    Custom early stopping callback that requires an improvement greater than epsilon.
    """
    best_score = None
    best_iter = 0
    wait_count = 0
    is_higher_better = None  # Will be set on the first evaluation result

    def callback(env):
        nonlocal best_score, best_iter, wait_count, is_higher_better
        # Monitor the first metric in evaluation_result_list.
        data_name, eval_name, current_score, current_is_higher_better = env.evaluation_result_list[0]
        if is_higher_better is None:
            is_higher_better = current_is_higher_better

        # Initialize best_score on the first call.
        if best_score is None:
            best_score = current_score
            best_iter = env.iteration
        else:
            if (not is_higher_better and (best_score - current_score) > epsilon) or \
               (is_higher_better and (current_score - best_score) > epsilon):
                best_score = current_score
                best_iter = env.iteration
                wait_count = 0  # Reset counter on significant improvement.
            else:
                wait_count += 1

        if wait_count >= stopping_rounds:
            if verbose:
                print(f"Early stopping triggered. Best iteration is {best_iter} with {eval_name}: {best_score:.5f}")
            env.model.stop_training = True

    callback.order = 10  # Ensure our callback runs in the proper order.
    return callback

#########################################
# Fold Evaluation Function
#########################################
def evaluate_fold(fold, params, normalize_power_output=True, log_target=False, epsilon=1e-6):
    """
    Train a model on a CV fold and compute regression metrics.

    For normalized data:
      - If log_target is True:
           * y_val and y_pred are in log-space.
           * 'norm_log_rmse' and 'norm_log_mae' are computed directly on the log-transformed values.
           * The inverse transformation (using np.exp and multiplying by capacity) is applied to compute
             the true errors on the original scale, returned as 'norm_rmse' and 'norm_mae'. Also, these true errors
             are duplicated as 'rmse' and 'mae' for consistency.
      - If log_target is False:
           * y_val and y_pred are normalized (i.e. capacity factors).
           * The inverse transformation (multiplying by capacity) is applied to compute the true errors,
             returned as 'norm_rmse' and 'norm_mae' (and also as 'rmse' and 'mae').

    For non-normalized data, the raw errors are returned as 'rmse', 'mae', and 'ev'.
    """
    norm_log_rmse, norm_log_mae = None, None
    X_train, X_val, y_train, y_val = fold

    # For normalized data, drop "capacity_MW" for modeling.
    if normalize_power_output:
        X_train_model = X_train.drop(columns=["capacity_MW"], errors="ignore")
        X_val_model = X_val.drop(columns=["capacity_MW"], errors="ignore")
    else:
        X_train_model = X_train
        X_val_model = X_val

    # Train using the LightGBM model.
    model = CustomLightGBM(best_params=params, random_state=params['random_state'], early_stopping_rounds=60)
    model.fit(X_train_model, y_train, eval_set=[(X_val_model, y_val)])
    y_pred = model.predict(X_val_model)

    # Compute metrics on the transformed scale (these may be in normalized or log space).
    base_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    base_mae = mean_absolute_error(y_val, y_pred)
    base_ev = explained_variance_score(y_val, y_pred)

    if normalize_power_output:
        capacity = X_val["capacity_MW"]
        if log_target:
            # When log-transformed, y_val and y_pred are in log-space.
            norm_log_rmse = base_rmse
            norm_log_mae = base_mae
            # Inverse-transform: original power = (exp(value) - epsilon) * capacity.
            y_val_orig = (np.exp(y_val) - epsilon) * capacity
            y_pred_orig = (np.exp(y_pred) - epsilon) * capacity
        else:
            # When not log-transformed, y_val and y_pred are normalized (capacity factors).
            y_val_orig = y_val * capacity
            y_pred_orig = y_pred * capacity

        true_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        true_mae = mean_absolute_error(y_val_orig, y_pred_orig)
        true_ev = explained_variance_score(y_val_orig, y_pred_orig)

        if log_target:
            metrics = {
                "norm_log_rmse": base_rmse,
                "norm_log_mae": base_mae,
                "rmse": true_rmse,
                "mae": true_mae,
                "ev": true_ev,
            }
        else:
            metrics = {
                "norm_rmse": base_rmse,
                "norm_mae": base_mae,
                "rmse": true_rmse,
                "mae": true_mae,
                "ev": true_ev,
            }
    else:
        # No normalization: use raw values.
        metrics = {
            "rmse": base_rmse,
            "mae": base_mae,
            "ev": base_ev
        }

    return metrics


class Optimizer:
    def __init__(self, config, objective=None, preprocessor=None):
        """
        Parameters:
            config (dict): Must include hyperparameters, algorithm identifier, eval count, etc.
            objective (callable, optional): Custom objective accepting (params, data).
            preprocessor (Preprocessor, optional): An already-fitted preprocessor instance.
        """
        self.config = config
        self.optimizer_algo = self._get_optimizer_algo(config.get("algo", "atpe.suggest"))
        self.max_evals = config.get("evals", 10)
        self.random_seed = config.get("random_state")
        self.preprocessor = preprocessor
        self.trials = Trials()
        self.objective = objective if objective is not None else self._default_objective
        self.metrics = None
        self.best_params = None
        self.best_loss = None  # For example, best normalized RMSE
        self.best_mae = None

    @staticmethod
    def _get_optimizer_algo(algo):
        if isinstance(algo, str):
            mapping = {
                "tpe.suggest": tpe.suggest,
                "atpe.suggest": atpe.suggest,
            }
            return mapping.get(algo, atpe.suggest)
        return algo

    def configure_parameter_space(self):
        hyperparams = self.config.get("hyperparams", {})
        space = {}
        for param, details in hyperparams.items():
            param_type = details.get("type", "float")
            if param_type == "int":
                space[param] = hp.quniform(param, details["min"], details["max"], 1)
            elif param_type == "float":
                space[param] = hp.uniform(param, details["min"], details["max"])
            elif param_type == "categorical":
                if "values" not in details:
                    raise ValueError(f"Categorical parameter '{param}' must have a 'values' list.")
                space[param] = hp.choice(param, details["values"])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type} for parameter '{param}'.")
        return space

    def _default_objective(self, params, data):
        # Prepare parameters for LightGBM.
        params['n_estimators'] = int(params.get('n_estimators', 100))
        params['max_depth'] = int(params.get('max_depth', -1))
        params['num_leaves'] = int(params.get('num_leaves', 31))
        params['min_child_samples'] = int(params.get('min_child_samples', 20))
        params.pop('gamma', None)
        params['random_state'] = self.random_seed if self.random_seed is not None else 42
        params['bagging_fraction'] = float(params.get('bagging_fraction', 1.0))
        params['bagging_freq'] = int(params.get('bagging_freq', 1))
        params['min_split_gain'] = float(params.get('min_split_gain', 0.0))

        folds = data.get("folds")
        if not folds:
            raise ValueError("CV folds not found in preprocessed data.")

        # Determine job settings based on data size.
        total_rows = self.preprocessor.train_data[0].shape[0]
        if total_rows < 30e6:
            backend = "threading"
            n_jobs = min(len(folds), 4)
        else:
            backend = "loky"
            n_jobs = 1

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(evaluate_fold)(
                fold,
                params,
                self.preprocessor.normalize_power_output,
                self.preprocessor.log_target_variable
            ) for fold in folds
        )

        # Aggregate metrics over folds.
        metric_keys = results[0].keys()
        aggregated = {key: np.mean([fold_result[key] for fold_result in results]) for key in metric_keys}

        hyperopt_step = len(self.trials)
        logger.info("Aggregated CV metrics: %s with params: %s", aggregated, params)

        # Log basic metrics.
        mlflow.log_metric("mean_cv_rmse", aggregated["rmse"], step=hyperopt_step)
        mlflow.log_metric("mean_cv_mae", aggregated["mae"], step=hyperopt_step)
        mlflow.log_metric("mean_cv_ev", aggregated["ev"], step=hyperopt_step)

        if self.preprocessor.normalize_power_output:
            if self.preprocessor.log_target_variable:
                # Log log-space metrics.
                mlflow.log_metric("mean_cv_norm_log_rmse", aggregated["norm_log_rmse"], step=hyperopt_step)
                mlflow.log_metric("mean_cv_norm_log_mae", aggregated["norm_log_mae"], step=hyperopt_step)
                # Return loss based on the log-space error.
                return {'loss': aggregated["norm_log_rmse"], 'metrics': aggregated, 'status': 'ok'}
            else:
                return {'loss': aggregated["norm_rmse"], 'metrics': aggregated, 'status': 'ok'}
        else:
            return {'loss': aggregated["rmse"], 'metrics': aggregated, 'status': 'ok'}


    def fit(self, objective=None):
        """
        For neural networks, train once on the full training set (with an internal validation split)
        and log per-epoch metrics. For other models (e.g., LightGBM), run hyperparameter optimization using CV.
        """
        if self.preprocessor is None or self.preprocessor.train_data is None:
            raise ValueError("Preprocessor has not been fitted. Please run preprocessor.fit() first.")

        # Branch for neural network: avoid duplicate training if already done.
        if getattr(self.preprocessor, "ml_model", None) == "neural_network":
            if hasattr(self.preprocessor, "ml_trained_model") and self.preprocessor.ml_trained_model is not None:
                logger.info("Neural network already trained, skipping duplicate training.")
                return self

            logger.info("Training neural network on the full training set (no folds, no hyperopt).")
            # Expect train_data as a tuple: (X_train, y_train)
            X_train, y_train = self.preprocessor.train_data

            # Get neural network hyperparameters.
            nn_params = self.config.get("neural_network_params", {})
            nn_params['random_state'] = self.random_seed if self.random_seed is not None else 42

            # Add gradient clipping parameters to mitigate exploding gradients.
            # For example, setting a default clipnorm value.
            nn_params.setdefault("clipnorm", 1.0)
            # Optionally, you can also add a clipvalue parameter:
            # nn_params.setdefault("clipvalue", 1.0)

            # Create and train the neural network.
            model = CustomNeuralNetwork(best_params=nn_params, random_state=nn_params['random_state'])
            # Define default callbacks with MLflow logging.
            default_callbacks = [
                keras_callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    min_delta=0.0001
                ),
                keras_callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=2,
                    min_lr=1e-4
                ),
                # Optionally, remove redundant ReduceLROnPlateau callbacks to simplify training.
                MLflowLoggingCallback()
            ]
            model.fit(X_train, y_train, validation_split=0.2, callbacks=default_callbacks)
            y_pred = model.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            mae = mean_absolute_error(y_train, y_pred)
            ev = explained_variance_score(y_train, y_pred)
            metrics = {"rmse": rmse, "mae": mae, "ev": ev}

            logger.info("Neural network training completed with metrics: %s", metrics)
            mlflow.log_metric("nn_rmse", rmse)
            mlflow.log_metric("nn_mae", mae)
            mlflow.log_metric("nn_ev", ev)

            # Save the trained model so that later calls do not retrain it.
            self.preprocessor.ml_trained_model = model
            self.best_params = nn_params
            self.best_loss = rmse
            self.best_mae = mae
            self.metrics = metrics
            return self

        # Otherwise, proceed with hyperparameter optimization using CV folds (for LightGBM).
        data = {
            "train": self.preprocessor.train_data,
            "test": self.preprocessor.test_data,
            "folds": self.preprocessor.get_cv_folds(),
            "scaler": self.preprocessor.scaler
        }
        obj_func = objective if objective is not None else self.objective

        def wrapped_objective(params):
            return obj_func(params, data)

        rstate = np.random.default_rng(self.random_seed) if self.random_seed is not None else None
        space = self.configure_parameter_space()

        algo_used = self.optimizer_algo
        if algo_used is atpe.suggest:
            logger.info("Using atpe.suggest")
        elif algo_used is tpe.suggest:
            logger.info("Using tpe.suggest")

        import warnings as std_warnings
        original_np_warnings = getattr(np, "warnings", None)
        np.warnings = std_warnings

        try:
            self.best_params = fmin(
                fn=wrapped_objective,
                space=space,
                algo=self.optimizer_algo,
                max_evals=self.max_evals,
                trials=self.trials,
                rstate=rstate
            )
        finally:
            if original_np_warnings is None:
                delattr(np, "warnings")
            else:
                np.warnings = original_np_warnings

        best_trial = min(self.trials.trials, key=lambda trial: trial["result"]["loss"])
        self.best_loss = best_trial["result"]["loss"]
        self.best_mae = best_trial["result"]["metrics"].get("mae")
        logger.info("Best hyperparameters from hyperopt: %s", self.best_params)
        mlflow.log_params(self.best_params)
        mlflow.log_metric("best_cv_loss", self.best_loss)
        mlflow.log_metric("best_cv_mae", self.best_mae)
        return self

    def transform(self):
        if self.best_params is None:
            raise ValueError("Call fit before calling transform.")
        if getattr(self.preprocessor, "ml_model", None) == "neural_network":
            return {
                "best_params": self.best_params,
                "model": self.preprocessor.ml_trained_model,
                "metrics": self.metrics
            }
        best_trial = min(self.trials.trials, key=lambda trial: trial["result"]["loss"])
        best_metrics = best_trial["result"].get("metrics", {})
        return {
            "best_params": self.best_params,
            "metrics": best_metrics
        }

    def fit_transform(self, objective=None):
        self.fit(objective)
        return self.transform()