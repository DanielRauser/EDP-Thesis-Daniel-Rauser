import lightgbm as lgb
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap  # Ensure you have installed shap: pip install shap

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


class CustomLightGBM(BaseEstimator, RegressorMixin):
    def __init__(self, best_params, random_state=None, **kwargs):
        """
        Initialize the custom LightGBM regressor.

        Parameters:
            best_params (dict): Dictionary of best hyperparameters.
            random_state (int): Random seed.
            **kwargs: Additional keyword arguments to pass to LGBMRegressor.
        """
        self.best_params = best_params
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y, **fit_params):
        """
        Fit the LightGBM model for regression.

        Parameters:
            X: Features.
            y: Target values.
            **fit_params: Additional parameters for the underlying fit method.
                         Expected to include (if needed) eval_set and callbacks.
        Returns:
            self
        """
        # Copy and adjust best_params.
        params = self.best_params.copy()
        params['n_estimators'] = int(params.get('n_estimators', 100))
        params['max_depth'] = int(params.get('max_depth', -1))
        params['num_leaves'] = int(params.get('num_leaves', 31))
        params['min_child_samples'] = int(params.get('min_child_samples', 20))
        params['bagging_fraction'] = float(params.get('bagging_fraction', 1.0))
        params['bagging_freq'] = int(params.get('bagging_freq', 1))
        params['min_split_gain'] = float(params.get('min_split_gain', 0.0))
        params['random_state'] = self.random_state

        logger.info("Initializing LightGBM regressor with parameters: %s", params)

        self.kwargs.setdefault("verbosity", -1)
        self.kwargs.setdefault("verbose", -1)

        # Initialize LGBMRegressor.
        self.model_ = lgb.LGBMRegressor(**params, **self.kwargs)

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """
        Predict target values for samples in X.
        """
        if self.model_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.model_.predict(X)

    def transform(self, X):
        """
        For compatibility with scikit-learn pipelines, returns the predictions.
        """
        return self.predict(X)

    def fit_transform(self, X, y, **fit_params):
        """
        Fit the model and return the transform output.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def get_feature_importance(self, importance_type='split'):
        """
        Returns the feature importance of the underlying LightGBM model as a DataFrame.

        Parameters:
            importance_type (str): Not used directly here (LightGBM's regressor returns the same by default),
                                   but you can extend this to choose between 'split' and 'gain'.

        Returns:
            A pandas DataFrame with features and their importances.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        # LightGBM feature importance can be retrieved from self.model_.feature_importances_
        importance = self.model_.feature_importances_
        # Try to get feature names from the underlying booster.
        if hasattr(self.model_, "booster_"):
            feature_names = self.model_.booster_.feature_name()
        else:
            feature_names = [f"f{i}" for i in range(len(importance))]
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        return df_importance.sort_values(by="importance", ascending=False)

    def plot_feature_importance(self, importance_type='split', max_features=20):
        """
        Plots the feature importance.

        Parameters:
            importance_type (str): 'split' or 'gain' (not fully implemented separately).
            max_features (int): Number of top features to display.

        Returns:
            The matplotlib figure object.
        """
        df_importance = self.get_feature_importance(importance_type=importance_type)
        df_top = df_importance.head(max_features)
        plt.figure(figsize=(10, 6))
        plt.barh(df_top["feature"][::-1], df_top["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        fig = plt.gcf()
        return fig

    def get_shap_summary(self, X, max_samples=1000):
        """
        Computes SHAP values and returns the SHAP summary plot.

        Parameters:
            X: Features as a DataFrame.
            max_samples (int): Maximum number of samples to use for SHAP summary.

        Returns:
            The SHAP summary plot figure.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        # Use underlying booster if available.
        model_to_explain = self.model_.booster_ if hasattr(self.model_,
                                                           "booster_") and self.model_.booster_ is not None else self.model_
        # Sample data if too many rows.
        if X.shape[0] > max_samples:
            X_sample = X.sample(n=max_samples, random_state=self.random_state)
        else:
            X_sample = X
        explainer = shap.TreeExplainer(model_to_explain)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        fig = plt.gcf()
        return fig

    def plot_shap_summary(self, X, max_samples=1000):
        """
        Generates and returns the SHAP summary plot figure.
        """
        try:
            fig = self.get_shap_summary(X, max_samples=max_samples)
            return fig
        except Exception as e:
            logger.error("Error generating SHAP summary plot: %s", e)
            return None


class CalibratedCustomLightGBM(CustomLightGBM):
    def __init__(self, best_params, random_state=None, calibrate=True, **kwargs):
        """
        Initialize a calibrated version of the CustomLightGBM model.

        Parameters:
            best_params (dict): Dictionary of best hyperparameters.
            random_state (int): Random seed.
            calibrate (bool): Whether to perform calibration after fitting.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(best_params, random_state=random_state, **kwargs)
        self.calibrate_model = calibrate
        self.calibrator_ = None

    def fit(self, X, y, **fit_params):
        """
        Fit the LightGBM model and, if calibrate=True, fit a calibration model
        on the training predictions.
        """
        super().fit(X, y, **fit_params)
        if self.calibrate_model:
            # Get raw predictions on training data.
            preds = super().predict(X)
            # Reshape predictions and fit calibrator.
            self.calibrator_ = LinearRegression().fit(preds.reshape(-1, 1), y)
            logger.info("Calibrator fitted. Intercept: %s, Coefficient: %s",
                        self.calibrator_.intercept_, self.calibrator_.coef_)
        return self

    def predict(self, X):
        """
        Predict target values. If a calibrator was fitted, apply it to the raw predictions.
        """
        preds = super().predict(X)
        if self.calibrator_ is not None:
            preds = self.calibrator_.predict(np.array(preds).reshape(-1, 1)).flatten()
        return preds


class QuantileCalibratedCustomLightGBM(CustomLightGBM):
    def __init__(self, best_params, random_state=None, calibrate=True, quantile=0.5, **kwargs):
        """
        Initialize a quantile-calibrated version of the CustomLightGBM model.

        Parameters:
            best_params (dict): Dictionary of best hyperparameters.
            random_state (int): Random seed.
            calibrate (bool): Whether to perform calibration after fitting.
            quantile (float): The quantile to estimate (default 0.5 for median).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(best_params, random_state=random_state, **kwargs)
        self.calibrate_model = calibrate
        self.calibration_quantile = quantile
        self.calibrator_ = None

    def fit(self, X, y, **fit_params):
        """
        Fit the LightGBM model and, if calibrate=True, fit a quantile regression calibrator
        on the training predictions.
        """
        # Fit the underlying LightGBM model
        super().fit(X, y, **fit_params)

        if self.calibrate_model:
            # Get raw predictions on training data
            preds = super().predict(X)
            # Fit a quantile regression calibrator.
            # Here, setting alpha=0 for an unregularized fit.
            self.calibrator_ = QuantileRegressor(quantile=self.calibration_quantile, alpha=0)
            self.calibrator_.fit(preds.reshape(-1, 1), y)

        return self

    def predict(self, X):
        """
        Predict target values. If a quantile calibrator was fitted, apply it to the raw predictions.
        """
        # Get raw predictions from the underlying model
        preds = super().predict(X)

        # If calibration was performed, adjust predictions via quantile regression
        if self.calibrator_ is not None:
            preds = self.calibrator_.predict(np.array(preds).reshape(-1, 1))
        return preds

