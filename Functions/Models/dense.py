import numpy as np
import mlflow

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score
)

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import Callback

import logging

logger = logging.getLogger(__name__)

@register_keras_serializable()
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

class MLflowLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            mlflow.log_metric(metric, value, step=epoch)

class CustomNeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, best_params, random_state=None, **kwargs):
        """
        A custom feed-forward neural network regressor.
        """
        self.best_params = best_params
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

    def build_model(self, input_shape):
        hidden_layers = self.best_params.get("hidden_layers", [1400, 900, 700, 700, 350, 350])
        activation = self.best_params.get("activation", "relu")
        dropout = self.best_params.get("dropout", 0.05)
        learning_rate = self.best_params.get("learning_rate", 0.001)

        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            # Adding LeakyReLU after the Dense layers
            model.add(layers.LeakyReLU(negative_slope=0.01))
            if dropout > 0.0:
                model.add(layers.Dropout(dropout))
        # Output layer for regression.
        model.add(layers.Dense(1, activation="linear"))

        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.5),
            loss=rmse_loss,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                "mae",
                "mse",
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="msle")
            ]
        )
        return model

    def fit(self, X, y, **fit_params):
        X_np = X.values if hasattr(X, "values") else X
        y_np = y.values if hasattr(y, "values") else y
        input_shape = (X_np.shape[1],)
        self.model_ = self.build_model(input_shape)

        epochs = self.best_params.get("epochs", 5)
        batch_size = self.best_params.get("batch_size", 512)

        # Retrieve user-specified callbacks, if any.
        callbacks_list = fit_params.get("callbacks", [])
        # Ensure MLflow logging is included.
        callbacks_list.append(MLflowLoggingCallback())
        validation_split = fit_params.get("validation_split", 0.2)

        logger.info("Training neural network for %d epochs with batch size %d", epochs, batch_size)
        self.model_.fit(
            X_np,
            y_np,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else X
        preds = self.model_.predict(X_np)
        return preds.flatten()

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        ev = explained_variance_score(y, y_pred)
        r2 = r2_score(y, y_pred)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "explained_variance": ev,
            "r2": r2
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics