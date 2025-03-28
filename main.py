import mlflow
import logging
import sys
import yaml
import argparse

from pipeline.dataprep import dataprep_task
from pipeline.train import train_task
from pipeline.apply import apply_task
from pipeline.evaluate import evaluate_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_pipeline(config: dict, experiment_name: str = None):
    """
    Runs the full ML pipeline (dataprep, tune, train, evaluate).

    For each step, the configuration can specify:
      - use_existing: A boolean flag indicating whether to use an existing run.
      - run_id: The MLflow run ID to use if reusing an existing step.

    If use_existing is True, the step does not execute a new run but instead
    sets a corresponding tag with the provided run ID. Otherwise, a new nested
    MLflow run is created to execute the task.
    """
    # Determine experiment name (command-line arg takes precedence over config)
    if experiment_name is None:
        experiment_name = config.get("experiment", {}).get("name", "Default_Experiment")

    # Set the MLflow experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Failed to set or retrieve experiment '{experiment_name}'.")
        sys.exit(1)
    experiment_id = experiment.experiment_id

    # Start the main MLflow run for the entire pipeline
    with mlflow.start_run(run_name="pipeline") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run with Experiment ID: {experiment_id}, Run ID: {run_id}")

        # Set experiment metadata as tags
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("experiment_name", experiment_name)

        # Data Preparation Step
        if "dataprep" in config and config["dataprep"]["run"]:
            dp_config = config["dataprep"]
            if dp_config.get("use_existing", False):
                existing_run_id = dp_config.get("run_id")
                logger.info(f"Using existing dataprep run id: {existing_run_id}")
                mlflow.set_tag("dataprep_run_id", existing_run_id)
            else:
                with mlflow.start_run(nested=True, run_name="dataprep"):
                    logger.info("Starting new dataprep step.")
                    dataprep_task(params=dp_config)
                    mlflow.log_param("step", "dataprep")

        # Training Step
        if "train" in config and config["train"]["run"]:
            train_config = config["train"]
            if train_config.get("use_existing", False):
                existing_run_id = train_config.get("run_id")
                logger.info(f"Using existing train run id: {existing_run_id}")
                mlflow.set_tag("train_run_id", existing_run_id)
            else:
                with mlflow.start_run(nested=True, run_name="train"):
                    logger.info("Starting new train step.")
                    train_task(params=train_config)
                    mlflow.log_param("step", "train")

        if "apply" in config and config["apply"]["run"]:
            apply_config = config["apply"]
            if apply_config.get("use_existing", False):
                existing_run_id = apply_config.get("run_id")
                logger.info(f"Using existing apply run id: {existing_run_id}")
                mlflow.set_tag("apply_run_id", existing_run_id)
            else:
                with mlflow.start_run(nested=True, run_name="apply"):
                    logger.info("Starting new apply step.")
                    apply_task(params=apply_config)
                    mlflow.log_param("step", "apply")

        # Evaluation Step
        if "evaluate" in config and config["evaluate"]["run"]:
            eval_config = config["evaluate"]
            if eval_config.get("use_existing", False):
                existing_run_id = eval_config.get("run_id")
                logger.info(f"Using existing evaluate run id: {existing_run_id}")
                mlflow.set_tag("evaluate_run_id", existing_run_id)
            else:
                with mlflow.start_run(nested=True, run_name="evaluate"):
                    logger.info("Starting new evaluate step.")
                    evaluate_task(
                        params_data_prep=eval_config.get("params_data_prep", {}),
                        params=eval_config.get("params", {})
                    )
                    mlflow.log_param("step", "evaluate")

        logger.info("All pipeline steps executed. MLflow pipeline run completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full ML pipeline (dataprep, tune, train, evaluate) based on a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Name of the MLflow experiment. If not provided, it will be taken from the config file."
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Loaded configuration: %s", config)

        experiment_name = args.experiment or config.get("experiment", {}).get("name", "Default_Experiment")
        if args.experiment:
            logger.info(f"Using experiment name from command-line argument: {experiment_name}")
        else:
            logger.info(f"Using experiment name from config: {experiment_name}")

        run_pipeline(config=config, experiment_name=experiment_name)
        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", e)
        sys.exit(1)