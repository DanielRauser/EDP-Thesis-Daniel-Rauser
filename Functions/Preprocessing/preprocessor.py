import logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def filter_suspicious_zero_output_dynamic(
        X: pd.DataFrame,
        base_solar_threshold=10,
        solar_amplitude=5,
        phase_shift=80,
        feature_quantile_bounds=(0.01, 0.99),
        power_threshold_fraction=0.05,
        absolute_power_threshold=0.1,
        use_dynamic_solar_elevation=True,
) -> pd.DataFrame:
    """
    Filters out rows where power output is 0 (or below expected levels) when environmental conditions suggest generation should be occurring.

    This function:
      - Adjusts the solar elevation threshold dynamically based on the day of year.
      - Uses several environmental features (if available) to define "normal" conditions based on daytime nonzero output.
      - Adjusts the minimum acceptable power output using system capacity.

    Parameters:
        X (pd.DataFrame): DataFrame that must include at least the columns:
            "Power(MW)", "solar_elevation", "day_of_year", and optionally "capacity_MW".
            It can also contain environmental features such as "ssrd", "tcc", "net_radiation",
            "strd", "t2m", "tp", "clear_sky_index", etc.
        base_solar_threshold (float): Base solar elevation threshold (degrees) for daytime.
        solar_amplitude (float): Amplitude for the seasonal (sinusoidal) adjustment.
        phase_shift (float): Phase shift for the sinusoidal function (to align the seasonal curve).
        feature_quantile_bounds (tuple): Lower and upper quantiles used to define normal ranges for the environmental features.
        power_threshold_fraction (float): Fraction of capacity used to compute a dynamic power threshold.
        absolute_power_threshold (float): Minimum absolute power output threshold.
        use_dynamic_solar_elevation (bool): If True, computes a dynamic solar threshold based on season.

    Returns:
        pd.DataFrame: A filtered DataFrame where suspicious zero (or near-zero) power outputs are removed.
    """
    X_ = X.copy()

    # Calculate dynamic solar threshold if required.
    if use_dynamic_solar_elevation:
        if "day_of_year" not in X_.columns:
            raise ValueError("DataFrame must contain 'day_of_year' column for dynamic solar threshold.")
        # Create a dynamic threshold using a sinusoidal pattern
        X_["solar_threshold"] = base_solar_threshold + solar_amplitude * np.sin(
            2 * np.pi * (X_["day_of_year"] - phase_shift) / 365)
    else:
        X_["solar_threshold"] = base_solar_threshold

    # Define daytime rows based on the (possibly dynamic) solar threshold.
    daytime_mask = X_["solar_elevation"] > X_["solar_threshold"]

    # Use daytime data with nonzero power to establish normal environmental conditions.
    normal_gen = X_.loc[daytime_mask & (X_["Power(MW)"] > 0)]

    # Define environmental features to assess.
    # You can adjust this list based on what you believe influences generation.
    env_features = ["ssrd", "tcc", "net_radiation", "strd", "t2m", "tp", "clear_sky_index"]
    # Ensure we only use features present in the DataFrame.
    env_features = [feat for feat in env_features if feat in X_.columns]

    q_low, q_high = feature_quantile_bounds
    feature_bounds = {
        feat: (normal_gen[feat].quantile(q_low), normal_gen[feat].quantile(q_high))
        for feat in env_features
    }

    # Calculate an expected power threshold.
    # If "capacity_MW" exists, use the larger of an absolute threshold or a fraction of capacity.
    if "capacity_MW" in X_.columns:
        expected_power_threshold = np.maximum(absolute_power_threshold, X_["capacity_MW"] * power_threshold_fraction)
    else:
        expected_power_threshold = absolute_power_threshold

    # Create a vectorized mask for suspicious rows:
    # 1. It is daytime (solar_elevation > dynamic threshold).
    # 2. Power output is below the expected threshold.
    suspicious_mask = (X_["solar_elevation"] > X_["solar_threshold"]) & (X_["Power(MW)"] < expected_power_threshold)

    # Add conditions that each environmental feature falls within its normal range.
    for feat in env_features:
        low, high = feature_bounds[feat]
        suspicious_mask &= X_[feat].between(low, high)

    filtered_X = X_.loc[~suspicious_mask].drop(columns=["solar_threshold"])

    logger.info(f"Filtered {suspicious_mask.sum()} suspicious 0-output rows out of {len(X)} total.")

    return filtered_X

class BasePreprocessor:
    def __init__(self, predictors=None, group_variable=None, filter_variable=None,
                 lag_features=None, interaction_terms=None, standardize = None, scaler_features = None):
        self.predictors = predictors if predictors is not None else []
        self.group_variable = group_variable
        self.filter_variable = filter_variable
        self.lag_features = lag_features if lag_features is not None else []
        self.interaction_terms = interaction_terms if interaction_terms is not None else {}
        self.standardize = standardize if standardize is not None else []
        self.scaler_features = scaler_features if scaler_features is not None else []
    def _add_features(self, X):
        """
        Add engineered features: capacity bin, nighttime dummy,
        interaction features, and (optionally) lag features.
        """
        X = X.copy()

        # Capacity bin feature
        if "capacity_MW" in X.columns:
            bins = [0, 20, 30, 47, 52, 69, 78, 98, 122, 150, 290]
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            X["capacity_bin"] = pd.cut(
                X["capacity_MW"],
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=False
            )
            X["capacity_bin"] = X["capacity_bin"].cat.add_categories([-1]).fillna(-1).astype("int")
            if "capacity_bin" not in self.predictors:
                self.predictors.append("capacity_bin")

        # Nighttime dummy
        if "solar_elevation" in X.columns:
            X["night"] = (X["solar_elevation"] < 0).astype(int)
            if "night" not in self.predictors:
                self.predictors.append("night")

        interaction_terms = {
            "solar_x_tcc": {
                "features": ["solar_elevation", "tcc"],
                "operation": "multiply"
            },
            "solar_x_csi": {
                "features": ["solar_elevation", "clear_sky_index"],
                "operation": "multiply"
            },
            "solar_x_netrad": {
                "features": ["solar_elevation", "net_radiation"],
                "operation": "multiply"
            },
            "t2m_x_tp": {
                "features": ["t2m", "tp"],
                "operation": "multiply"
            },
            "solar_x_hour": {
                "features": ["solar_elevation", "hour"],
                "operation": "multiply"
            },
            "tcc_x_ssrd": {
                "features": ["tcc", "ssrd"],
                "operation": "multiply"
            },
            "t2m_x_netrad": {
                "features": ["t2m", "net_radiation"],
                "operation": "multiply"
            },
            "csi_x_netrad": {
                "features": ["clear_sky_index", "net_radiation"],
                "operation": "multiply"
            }
        }

        for feat_name, details in interaction_terms.items(): #self.interaction_terms.items()
            base_features = details['features']
            operation = details['operation']

            if all(col in X.columns for col in base_features):
                if operation == 'multiply':
                    X[feat_name] = X[base_features[0]] * X[base_features[1]]

                if feat_name not in self.predictors:
                    self.predictors.append(feat_name)

        # Lag features (only if a timestamp column and lag features are provided)
        if self.filter_variable and self.lag_features:
            X = X.sort_values(by=[self.group_variable, self.filter_variable])

            # Calculate median frequency in minutes across groups
            time_diffs = (
                    X.groupby(self.group_variable)[self.filter_variable]
                    .diff()
                    .dropna()
                    .dt.total_seconds() / 60  # convert seconds to minutes
            )
            median_freq = time_diffs.median()
            logger.info("Median frequency (minutes): %s", median_freq)

            desired_lag_minutes = [60, 120, 240]

            for feature in self.lag_features:
                if feature in X.columns:
                    for lag_minutes in desired_lag_minutes:
                        # Determine how many rows to shift based on the frequency
                        steps = int(round(lag_minutes / median_freq))
                        lag_name = f"{feature}_lag_{lag_minutes}min"
                        X[lag_name] = X.groupby(self.group_variable)[feature].shift(steps).bfill()
                        if lag_name not in self.predictors:
                            self.predictors.append(lag_name)
                else:
                    logger.warning("Lag feature '%s' not found in data.", feature)
        return X

    def _apply_standardization(self, X, fit=False):
        """
        Standardize the features defined in self.scaler_features.
        """
        if self.standardize:
            if fit:
                self.scaler = StandardScaler()
                X_std = pd.DataFrame(
                    self.scaler.fit_transform(X[self.scaler_features]),
                    columns=self.scaler_features,
                    index=X.index
                )
            else:
                if self.scaler is None:
                    raise ValueError("Scaler has not been fitted. Please call fit() first.")
                X_std = pd.DataFrame(
                    self.scaler.transform(X[self.scaler_features]),
                    columns=self.scaler_features,
                    index=X.index
                )
            return X_std
        return X


class Preprocessor(BasePreprocessor):
    def __init__(
            self,
            target_variable,
            predictors,
            test_size,
            cv_folds=5,
            ml_model="lightgbm",
            standardize=True,
            random_state=42,
            group_variable=None,
            split_type="cv",
            filter_variable=None,
            lag_features=None,
            interaction_terms=None,
            apply_filter=False,
            normalize_power_output=True,
            log_target_variable=True
    ):
        super().__init__(predictors, group_variable, filter_variable, lag_features=lag_features,interaction_terms=interaction_terms)
        if group_variable is None:
            raise ValueError("A group variable must be provided for group-based splitting.")
        self.target_variable = target_variable
        self.predictors = predictors.copy()
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.ml_model = ml_model
        self.standardize = standardize
        self.random_state = random_state
        self.group_variable = group_variable
        self.split_type = split_type
        self.filter_variable = filter_variable
        self.apply_filter = apply_filter
        self.lag_features = lag_features if lag_features is not None else []
        self.interaction_terms= interaction_terms if interaction_terms is not None else {}
        self.normalize_power_output = normalize_power_output
        self.log_target_variable = log_target_variable

        self.scaler = None
        self.scaler_features = None
        self.train_data = None  # Tuple: (X_train_proc, y_train)
        self.test_data = None  # Tuple: (X_test_proc, y_test)
        self.cv_fold_indices = None  # List of (train_idx, val_idx)

    def fit(self, X, y=None):
        logger.info("Preprocessing Dataset with originally: %s rows", len(X))

        # --- 1) Filtering ---
        if self.apply_filter and self.filter_variable:
            if self.filter_variable in X.columns:
                logger.info("Filtering on exact-hour timestamps")
                if not pd.api.types.is_datetime64_any_dtype(X[self.filter_variable]):
                    X[self.filter_variable] = pd.to_datetime(X[self.filter_variable], errors='coerce')
                mask = (
                        X[self.filter_variable].dt.minute.eq(0) &
                        X[self.filter_variable].dt.second.eq(0) &
                        X[self.filter_variable].dt.microsecond.eq(0)
                )
                X = X[mask]
                logger.info("Dataset filtered to hourly timestamps. Remaining rows: %s", len(X))
            else:
                raise ValueError(f"Filter variable '{self.filter_variable}' not found in DataFrame.")

        if "dist_in_km" in X.columns:
            max_distance = 16
            X = X[X["dist_in_km"] <= max_distance]
            logger.info(f"Filtered dataset to dist_in_km <= {max_distance}. Remaining rows: {len(X)}")
        else:
            raise ValueError("Column 'dist_in_km' not found in DataFrame.")

        # --- 1.1) Filtering on exceptional high capacity factors
        max_capacity_factor = 0.99
        X = X[X["Power(MW)"] <= (max_capacity_factor * X["capacity_MW"])]
        logger.info("Filtered dataset on outliers. Remaining rows: %s", len(X))

        # --- 1.2) Additional filtering for anomalous zero-power records
        X = filter_suspicious_zero_output_dynamic(X)
        logger.info("Filtered anomalous zero-power records with positive solar conditions. Remaining rows: %s", len(X))

        # --- 2) Prepare target and features ---
        if y is None:
            if self.target_variable not in X.columns:
                raise ValueError(f"Target variable '{self.target_variable}' not found in DataFrame.")
            y = X[self.target_variable]

        # Normalize target by capacity_MW (and optionally log transform)
        if self.normalize_power_output:
            if "capacity_MW" not in X.columns:
                raise ValueError("Cannot normalize power output because 'capacity_MW' column is missing.")
            y = y / X["capacity_MW"]
            logger.info("Normalized target variable '%s' by capacity_MW.", self.target_variable)
            if self.log_target_variable:
                epsilon = 1e-6
                y = np.log(y + epsilon)
                logger.info("Applied log transformation to normalized target variable '%s'.", self.target_variable)

        # --- 2.2) Feature Engineering ---
        X_data = self._add_features(X.copy())

        # Temporarily add group variable if not in self.predictors (needed for splitting)
        temp_added_group = False
        if self.group_variable not in self.predictors:
            if self.group_variable in X.columns:
                X_data[self.group_variable] = X[self.group_variable]
                temp_added_group = True
            else:
                raise ValueError(f"Group variable '{self.group_variable}' not found in DataFrame.")

        # --- 2.3) Filter on predictors ---
        predictors_for_split = self.predictors.copy()
        # Ensure group variable is included in X_data for splitting
        if self.group_variable not in predictors_for_split and self.group_variable in X_data.columns:
            predictors_for_split.append(self.group_variable)
        X_data = X_data[predictors_for_split]

        # --- 3) Group-based Splitting ---
        logger.info("Splitting data into train and test sets using group-based splitting")
        if self.group_variable not in X.columns:
            raise ValueError(f"Group variable '{self.group_variable}' not found in DataFrame.")

        unique_groups = X[self.group_variable].unique()

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_groups_idx, test_groups_idx = next(gss.split(unique_groups, groups=unique_groups))
        train_groups = unique_groups[train_groups_idx]
        test_groups = unique_groups[test_groups_idx]

        # Use X_data (which now includes the group variable) to compute masks
        train_mask = X_data[self.group_variable].isin(train_groups)
        test_mask = X_data[self.group_variable].isin(test_groups)

        X_train = X_data[train_mask]
        y_train = y[train_mask]
        X_test = X_data[test_mask]
        y_test = y[test_mask]

        del X_data

        # If the group variable was added only for splitting, drop it from the final data
        if temp_added_group:
            X_train = X_train.drop(columns=[self.group_variable], errors="ignore")
            X_test = X_test.drop(columns=[self.group_variable], errors="ignore")

        logger.info(
            "Train data shape: %s, Test data shape: %s",
            X_train.shape, X_test.shape
        )
        mlflow.log_metric("test_samples", X_test.shape[0])

        # Save the original capacity values from X_test (and X_train if needed)
        X_train_capacity = X_train["capacity_MW"].copy() if "capacity_MW" in X_train.columns else None
        X_test_capacity = X_test["capacity_MW"].copy() if "capacity_MW" in X_test.columns else None

        # --- 4) Standardization ---
        if self.standardize:
            logger.info("Standardizing training data")
            self.scaler_features = (
                [col for col in X_train.columns if col != "capacity_MW"]
                if self.normalize_power_output
                else list(X_train.columns)
            )
            X_train_proc = self._apply_standardization(X_train, fit=True)
            X_test_proc = self._apply_standardization(X_test, fit=False)
            mlflow.log_param("standardize", True)
        else:
            X_train_proc, X_test_proc = X_train, X_test
            mlflow.log_param("standardize", False)

        # Reattach the capacity_MW column to X_test_proc for evaluation purposes
        if X_train_capacity is not None and self.ml_model != "neural_network":
            X_train_proc["capacity_MW"] = X_train_capacity
        if X_test_capacity is not None:
            X_test_proc["capacity_MW"] = X_test_capacity

        self.train_data = (X_train_proc, y_train)
        self.test_data = (X_test_proc, y_test)

        # --- 5) Compute Cross-Validation Folds if we use lightgbm---
        if self.ml_model=="lightgbm":
            self.cv_fold_indices = []
            logger.info("Creating CV fold indices")
            cv_splitter = GroupKFold(n_splits=self.cv_folds)
            groups = X.loc[X_train.index, self.group_variable]
            for train_idx, val_idx in cv_splitter.split(X_train_proc, y_train, groups=groups):
                self.cv_fold_indices.append((train_idx, val_idx))

        return self

    def transform(self, X):
        X_trans = self._add_features(X)
        # Use only the predictors (drop capacity_MW if not needed)
        X_proc = X_trans[self.predictors].drop(columns=["capacity_MW"], errors="ignore")
        if self.standardize:
            X_proc = self._apply_standardization(X_trans, fit=False)
        return X_proc

    def inverse_transform_predictions(self, predictions, X):
        """
        Reverse normalization and log transformation.
        """
        if self.normalize_power_output:
            if "capacity_MW" not in X.columns:
                raise ValueError("Column 'capacity_MW' is required to inverse transform predictions.")
            if self.log_target_variable:
                epsilon = 1e-6
                predictions = np.exp(predictions) - epsilon
            predictions = predictions * X["capacity_MW"]
        return predictions

    def get_cv_folds(self):
        if self.train_data is None or self.cv_fold_indices is None:
            raise ValueError("The preprocessor has not been fitted yet.")
        X_train_proc, y_train = self.train_data
        folds = []
        for train_idx, val_idx in self.cv_fold_indices:
            folds.append((
                X_train_proc.iloc[train_idx],
                X_train_proc.iloc[val_idx],
                y_train.iloc[train_idx],
                y_train.iloc[val_idx]
            ))
        return folds

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self


class PreprocessorInferenceWrapper(BasePreprocessor):
    def __init__(self, scaler, predictors, standardize,
                 normalize_power_output=True, log_target_variable=True,
                 scaler_features=None, group_variable=None, filter_variable=None,
                 lag_features=None, interaction_terms=None):
        """
        A lightweight wrapper for inference. It expects the fitted scaler, predictors,
        and other transformation parameters from training.

        Parameters:
            scaler: The fitted StandardScaler.
            predictors (list): List of predictor column names.
            standardize (bool): Whether standardization is applied.
            normalize_power_output (bool): If True, the target was normalized during training.
            log_target_variable (bool): If True, the target was log-transformed during training.
            scaler_features (list): The list of features used in scaling.
            group_variable (str): The column name used for group-based features.
            filter_variable (str): The timestamp column name for filtering/lag features.
            lag_features (list): List of feature names for which lags were computed.
        """
        super().__init__(predictors, group_variable, filter_variable, lag_features, interaction_terms=interaction_terms)
        self.scaler = scaler
        self.predictors = predictors.copy()
        self.standardize = standardize
        self.normalize_power_output = normalize_power_output
        self.log_target_variable = log_target_variable
        self.scaler_features = scaler_features
        self.group_variable = group_variable
        self.filter_variable = filter_variable
        # Set lag features as provided (if any) to ensure they are computed during inference.
        self.lag_features = lag_features if lag_features is not None else []
        self.interaction_terms = interaction_terms if interaction_terms is not None else {}

    def transform(self, X):
        X_trans = self._add_features(X)
        # Ensure to select only the predictors (dropping capacity_MW if not needed)
        X_proc = X_trans[self.predictors].drop(columns=["capacity_MW"], errors="ignore")
        if self.standardize:
            X_proc = self._apply_standardization(X_trans, fit=False)
        return X_proc

    def inverse_transform_predictions(self, predictions, X):
        if self.normalize_power_output:
            if "capacity_MW" not in X.columns:
                raise ValueError("Column 'capacity_MW' required for inverse transformation.")
            if self.log_target_variable:
                epsilon = 1e-6
                predictions = np.exp(predictions) - epsilon
            predictions = predictions * X["capacity_MW"]
        return predictions