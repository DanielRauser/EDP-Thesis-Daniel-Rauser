import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from mlflow import log_metrics

from Functions.Evaluation.helpers import log_fig

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained (and already calibrated) model on test data.
    Computes standard and normalized regression metrics and generates diagnostic plots.
    Diagnostic plots are also generated per capacity quantile.

    Improvements include:
      - Removing reliance on MAPE (due to near-zero actual values) and using SMAPE instead.
      - Adding median absolute error (MedAE) and normalized RMSE (NRMSE relative to the target range).
      - Introducing tolerance-based accuracy metrics (both relative and fixed thresholds).
      - Enhancing diagnostics to assess performance across different capacities.
      - Adapting feature importance extraction for neural networks using permutation importance.
      - Updated SHAP integration to support all activation functions (e.g., ReLU, LeakyReLU)
        by directly using the op_handler update as suggested in GitHub Issue #1463.
    """

    def __init__(self, model, preprocessor):
        """
        Parameters:
            model: A trained model with a .predict() method. Assumed to be calibrated.
            preprocessor: Provides test_data, scaler, and normalize_power_output.
        """
        self.model = model
        self.X_test, self.y_test = preprocessor.test_data
        self.scaler = preprocessor.scaler
        self.normalize_power_output = preprocessor.normalize_power_output
        self.log_target_variable = preprocessor.log_target_variable

        self.trained_features_ = None
        if hasattr(model, "model_") and hasattr(model.model_, "booster_"):
            try:
                self.trained_features_ = model.model_.booster_.feature_name()
            except Exception as e:
                logger.warning("Could not retrieve feature names from booster: %s", e)

    def evaluate(self):
        """
        Evaluates the model:
          1. Prepares test features.
          2. Computes predictions (already calibrated).
          3. Computes overall metrics including additional robust metrics.
          4. Logs overall and quantile-specific diagnostic plots.
          5. Logs feature importance or permutation importance (for neural networks).
        Returns:
            A dictionary with evaluation metrics.
        """
        X_test, y_test = self.X_test, self.y_test

        # Prepare test features: if normalized by capacity, drop "capacity_MW" for prediction.
        X_test_model = (
            X_test.drop(["capacity_MW"], axis=1)
            if self.normalize_power_output and "capacity_MW" in X_test.columns
            else X_test
        )
        predictions = self.model.predict(X_test_model)

        # Convert predictions to actual MW values if model operates on capacity-factor scale.
        if self.normalize_power_output and "capacity_MW" in X_test.columns:
            capacities = X_test["capacity_MW"]
            if self.log_target_variable:
                epsilon = 1e-6
                # Inverse-transform predictions and targets (both are in log(normalized) space)
                predictions_norm = np.exp(predictions) - epsilon
                y_test_norm = np.exp(y_test) - epsilon
            else:
                predictions_norm = predictions
                y_test_norm = y_test

            # Convert normalized values (capacity factors) to actual MW.
            predictions_actual = predictions_norm * capacities
            y_test_actual = y_test_norm * capacities

            # Clip predictions to be within a valid range: between 0 and the capacity.
            predictions_actual = np.clip(predictions_actual, 0, capacities)
        else:
            predictions_actual = predictions
            y_test_actual = y_test

        # Calculate residuals.
        errors = y_test_actual - predictions_actual

        # Create metrics as a dictionary literal.
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test_actual, predictions_actual)),
            "mae": mean_absolute_error(y_test_actual, predictions_actual),
            "medae": median_absolute_error(y_test_actual, predictions_actual),
            "r2": r2_score(y_test_actual, predictions_actual),
            "explained_variance": explained_variance_score(y_test_actual, predictions_actual),
            "smape": 100 * np.mean(
                np.abs(y_test_actual - predictions_actual) /
                ((np.abs(y_test_actual) + np.abs(predictions_actual)) / 2)
            )
        }
        y_range = y_test_actual.max() - y_test_actual.min()
        metrics["nrmse"] = metrics["rmse"] / y_range if y_range != 0 else np.nan

        # Tolerance-based accuracy (e.g., predictions within 5% of actual).
        relative_tolerance = 0.05
        metrics["accuracy_within_5pct"] = np.mean(np.abs(errors) <= relative_tolerance * np.abs(y_test_actual)) * 100

        # Normalized metrics.
        if self.normalize_power_output:
            metrics["norm_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics["norm_mae"] = mean_absolute_error(y_test, predictions)
        else:
            capacities = X_test["capacity_MW"] if "capacity_MW" in X_test.columns else 1.0
            metrics["norm_rmse"] = np.sqrt(np.mean(((y_test_actual - predictions_actual) / capacities) ** 2))
            metrics["norm_mae"] = np.mean(np.abs((y_test_actual - predictions_actual) / capacities))

        log_metrics(metrics)

        # ------------------
        # Overall Diagnostic Plots
        # ------------------

        # Predicted vs Actual plot.
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_actual, predictions_actual, alpha=0.5)
        plt.xlabel("Actual (MW)")
        plt.ylabel("Predicted (MW)")
        plt.title("Predicted vs Actual")
        min_val = min(y_test_actual.min(), predictions_actual.min())
        max_val = max(y_test_actual.max(), predictions_actual.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        log_fig("predicted_vs_actual.png")

        # Residual plot.
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions_actual, errors, alpha=0.5)
        plt.xlabel("Predicted (MW)")
        plt.ylabel("Residuals (MW)")
        plt.title("Residual Plot")
        plt.axhline(0, color="red", linestyle="--")
        log_fig("residual_plot.png")

        # Histogram of residuals.
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel("Residuals (MW)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")
        log_fig("residual_histogram.png")

        # Boxplot of residuals.
        plt.figure(figsize=(8, 6))
        plt.boxplot(errors, vert=False)
        plt.xlabel("Residual (MW)")
        plt.title("Boxplot of Residuals")
        log_fig("boxplot_residuals.png")

        # CDF of absolute errors.
        abs_errors = np.abs(errors)
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_errors, cdf, marker=".", linestyle="none")
        plt.xlabel("Absolute Error (MW)")
        plt.ylabel("CDF")
        plt.title("Cumulative Distribution of Absolute Errors")
        log_fig("cdf_absolute_errors.png")

        # -------------------------
        # Diagnostics by Capacity
        # -------------------------
        if "capacity_MW" in X_test.columns:
            capacities = X_test["capacity_MW"]
            # Residuals vs Capacity plot.
            plt.figure(figsize=(8, 6))
            plt.scatter(capacities, errors, alpha=0.5)
            plt.xlabel("Capacity (MW)")
            plt.ylabel("Residuals (MW)")
            plt.title("Residuals vs Capacity")
            plt.axhline(0, color="red", linestyle="--")
            log_fig("residuals_vs_capacity.png")

            # Group residuals by capacity bins.
            capacity_bins = pd.qcut(capacities, q=10, duplicates="drop")
            df_group = pd.DataFrame({"capacity": capacities, "residual": errors})
            grouped = df_group.groupby(capacity_bins, observed=False).agg(
                {"capacity": "mean", "residual": ["mean", "std"]}
            )
            grouped.columns = ["mean_capacity", "mean_residual", "std_residual"]
            grouped = grouped.reset_index()

            plt.figure(figsize=(8, 6))
            plt.errorbar(
                grouped["mean_capacity"],
                grouped["mean_residual"],
                yerr=grouped["std_residual"],
                fmt="o",
                capsize=5,
            )
            plt.xlabel("Mean Capacity (MW)")
            plt.ylabel("Mean Residual (MW)")
            plt.title("Grouped Residuals by Capacity Bins")
            log_fig("grouped_residuals_by_capacity.png")

            # Analysis per capacity quantile.
            df_quant = pd.DataFrame({
                "capacity": capacities,
                "predicted": predictions_actual,
                "actual": y_test_actual,
                "error": errors,
                "abs_error": np.abs(errors),
            })
            df_quant["capacity_bin"] = pd.qcut(df_quant["capacity"], q=10, duplicates="drop")
            quantile_metrics = df_quant.groupby("capacity_bin", observed=True).agg(
                rmse=("error", lambda arr: np.sqrt(np.mean(arr ** 2))),
                mae=("error", lambda arr: np.mean(np.abs(arr))),
                medae=("error", lambda arr: np.median(np.abs(arr))),
                count=("error", "count")
            ).reset_index()
            quantile_metrics["bin_label"] = quantile_metrics["capacity_bin"].apply(
                lambda interval: f"{interval.left:.1f}–{interval.right:.1f} MW"
            )

            # Grouped bar chart for RMSE and MAE.
            ind = np.arange(len(quantile_metrics))
            width = 0.4
            plt.figure(figsize=(10, 6))
            plt.bar(ind - width / 2, quantile_metrics["rmse"], width, label="RMSE")
            plt.bar(ind + width / 2, quantile_metrics["mae"], width, label="MAE")
            plt.xticks(ind, quantile_metrics["bin_label"], rotation=45, ha="right")
            plt.xlabel("Capacity Range (MW)")
            plt.ylabel("Error (MW)")
            plt.title("RMSE and MAE by Capacity Quantile")
            plt.legend()
            plt.tight_layout()
            log_fig("rmse_mae_by_capacity_quantile.png")
            plt.close()

            # Quantile-specific diagnostic plots.
            cutoff_table_data = []
            quantile_count = 0
            for quant, group in df_quant.groupby("capacity_bin", observed=False):
                quantile_count += 1
                quant_str = (
                    str(quant)
                    .replace(" ", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "_")
                )

                # Predicted vs Actual for quantile.
                plt.figure(figsize=(8, 6))
                plt.scatter(group["actual"], group["predicted"], alpha=0.5)
                plt.xlabel("Actual (MW)")
                plt.ylabel("Predicted (MW)")
                plt.title(f"Predicted vs Actual for Quantile {quant_str} MW")
                q_min = group["actual"].min()
                q_max = group["actual"].max()
                plt.plot([q_min, q_max], [q_min, q_max], "r--")
                log_fig(f"{quantile_count}_{quant_str}/predicted_vs_actual.png")

                # Residual plot for quantile.
                plt.figure(figsize=(8, 6))
                plt.scatter(group["predicted"], group["error"], alpha=0.5)
                plt.xlabel("Predicted (MW)")
                plt.ylabel("Residual (MW)")
                plt.title(f"Residual Plot for Quantile {quant_str} MW")
                plt.axhline(0, color="red", linestyle="--")
                log_fig(f"{quantile_count}_{quant_str}/residual_plot.png")

                # Histogram of residuals for quantile.
                plt.figure(figsize=(8, 6))
                plt.hist(group["error"], bins=30, alpha=0.7)
                plt.xlabel("Residual (MW)")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of Residuals for Quantile {quant_str} MW")
                log_fig(f"{quantile_count}_{quant_str}/residual_histogram.png")

                # Boxplot of residuals for quantile.
                plt.figure(figsize=(8, 6))
                plt.boxplot(group["error"], vert=False)
                plt.xlabel("Residual (MW)")
                plt.title(f"Boxplot of Residuals for Quantile {quant_str} MW")
                log_fig(f"{quantile_count}_{quant_str}/boxplot_residuals.png")

                # CDF of absolute errors for quantile.
                sorted_abs = np.sort(group["abs_error"])
                cdf_q = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
                plt.figure(figsize=(8, 6))
                plt.plot(sorted_abs, cdf_q, marker=".", linestyle="none")
                plt.xlabel("Absolute Error (MW)")
                plt.ylabel("CDF")
                plt.title(f"CDF of Absolute Errors for Quantile {quant_str} MW")
                log_fig(f"{quantile_count}_{quant_str}/cdf_absolute_errors.png")

                # Compute absolute error cutoffs.
                if len(sorted_abs) > 0:
                    abs_80 = np.percentile(sorted_abs, 80)
                    abs_90 = np.percentile(sorted_abs, 90)
                    abs_99 = np.percentile(sorted_abs, 99)
                else:
                    abs_80, abs_90, abs_99 = np.nan, np.nan, np.nan

                cutoff_table_data.append({
                    "Quantile": quant_str,
                    "80% AE": abs_80,
                    "90% AE": abs_90,
                    "99% AE": abs_99,
                })

            df_table = pd.DataFrame(cutoff_table_data)
            fig, ax = plt.subplots(figsize=(8, df_table.shape[0] * 0.5 + 1))
            ax.axis("tight")
            ax.axis("off")
            column_labels = ["Capacity Quantile", "80% AE", "90% AE", "99% AE"]
            ax.table(cellText=df_table.values, colLabels=column_labels, loc="center")
            plt.title("Absolute Error Cutoffs by Capacity Quantile")
            log_fig("absolute_error_cutoffs_table.png")
            plt.close()

            # ------------------------
            # Feature Importance / SHAP Summaries
            # ------------------------
            try:
                if self.scaler is not None:
                    if hasattr(self.model, "scaler_features"):
                        scaled_cols = self.model.scaler_features
                    else:
                        scaled_cols = X_test_model.columns.tolist()
                    X_test_to_inverse = X_test_model[scaled_cols]
                    inv = self.scaler.inverse_transform(X_test_to_inverse)
                    X_test_original = pd.DataFrame(inv, columns=scaled_cols, index=X_test_to_inverse.index)
                    # Always add capacity_MW back for diagnostic plots.
                    if self.normalize_power_output and "capacity_MW" in X_test.columns:
                        X_test_original["capacity_MW"] = X_test["capacity_MW"]
                else:
                    X_test_original = X_test_model.copy()
            except Exception as e:
                logger.error("Error during inverse_transform: %s", e)
                X_test_original = X_test_model.copy()

            # For SHAP, if the model is a neural network (trained without capacity_MW), drop that column:
            X_test_shap = X_test_original.copy()
            if self.model.__class__.__name__ == "CustomNeuralNetwork":
                X_test_shap = X_test_shap.drop(columns=["capacity_MW"], errors="ignore")

            if self.trained_features_ is not None:
                final_cols = [c for c in self.trained_features_ if c in X_test_original.columns]
                X_test_original = X_test_original[final_cols]

            # Prefer built-in feature importance if available.
            if hasattr(self.model, "plot_feature_importance") and callable(self.model.plot_feature_importance):
                fi_fig = self.model.plot_feature_importance(max_features=20)
                mlflow.log_figure(fi_fig, "feature_importance_trained_model_custom.png")
                plt.close(fi_fig)
            elif hasattr(self.model, "feature_importances_"):
                feat_importances = self.model.model_.feature_importances_
                features = self.trained_features_ if self.trained_features_ is not None else X_test_original.columns
                indices = np.argsort(feat_importances)[::-1]
                plt.figure(figsize=(10, 6))
                plt.title("Feature Importance - Trained Model")
                plt.bar(range(len(features)), feat_importances[indices], align="center")
                plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha="right")
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "feature_importance_trained_model.png")
                plt.close()
            # Now generate a SHAP summary plot.
            try:
                # If the model provides its own SHAP plot, use that.
                if hasattr(self.model, "plot_shap_summary") and callable(self.model.plot_shap_summary):
                    shap_fig = self.model.plot_shap_summary(X_test_original, max_samples=1000)
                    if shap_fig is not None:
                        mlflow.log_figure(shap_fig, "shap_summary_trained_model_custom.png")
                        plt.close(shap_fig)
                # If the model is tree-based, use TreeExplainer.
                elif hasattr(self.model, "model_") and hasattr(self.model.model_, "booster_"):
                    explainer = shap.TreeExplainer(self.model.model_.booster_)
                    shap_values = explainer.shap_values(X_test_original)
                    plt.figure()
                    shap.summary_plot(shap_values, X_test_original, show=False)
                    mlflow.log_figure(plt.gcf(), "shap_summary_trained_model.png")
                    plt.close()
                # If the model is a neural network, use DeepExplainer with the updated op_handler. -> Not supported currently
               # elif self.model.__class__.__name__ == "CustomNeuralNetwork":
               #     sample_data = X_test_original.iloc[:min(100, len(X_test_original))]
               #     explainer = shap.DeepExplainer(self.model.predict, sample_data)
               #     shap_values = explainer.shap_values(X_test_original)
               #     plt.figure()
               #     shap.summary_plot(shap_values, X_test_original, show=False)
               #      mlflow.log_figure(plt.gcf(), "shap_summary_trained_model.png")
               #      plt.close()
            except Exception as e:
                logger.warning("SHAP summary plot could not be generated: %s", e)

            return metrics