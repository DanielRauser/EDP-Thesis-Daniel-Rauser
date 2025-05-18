import logging

from Functions.Dataprep.redes_dataprep import prepare_redes_data
from Functions.Dataprep.climate_dataprep import fetch_climate_data
from Functions.Dataprep.solar_dataprep import prepare_solar_train_data
from Functions.Dataprep.plants_dataprep import prepare_plant_era5_data, prepare_plant_cordex_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def dataprep_task(params: dict):
    logger.info("Starting data preparation")
    input_path = params.get("input_path", ".")
    output_path = params.get("output_path", ".")

    prepare_redes_data(output_path=output_path, update=False)

    if params.get("fetch_climate_data"):
        climate_config = params['climate']
        fetch_climate_data(output_path, climate_config)

    prepare_solar_train_data(input_path=input_path,
                             output_path=output_path,
                             era5_file="era5_us_2006.nc",
                             batch_size=250)

    plant_config = params['plants']

    prepare_plant_era5_data(
                            file_name="Plants_EDP.xlsx",
                            sheet_names=plant_config["names"],
                            era5_file="era5_portugal_baseline.nc",
                            output_path=output_path
                            )

    prepare_plant_cordex_data(
                              file_name="Plants_EDP.xlsx",
                              sheet_names=plant_config["names"],
                              output_path=output_path
                             )