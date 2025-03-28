import os
import logging
import requests
import holidays
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_redes_data():
    """
    Downloads multiple Excel files, reads them into DataFrames, and deletes the files.

    Returns:
        dict: A dictionary where keys are dataset names and values are pandas DataFrames.
    """
    urls = [
        "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets/energia-produzida-total-nacional/exports/xlsx?lang=en&timezone=Europe%2FLisbon&use_labels=true",
        "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets/energia-injetada-na-rede-de-distribuicao/exports/xlsx?lang=en&timezone=Europe%2FLisbon&use_labels=true",
        "https://e-redes.opendatasoft.com/api/explore/v2.1/catalog/datasets/consumo-total-nacional/exports/xlsx?lang=en&timezone=Europe%2FLisbon&use_labels=true"
    ]

    file_names = [
        "produzida-total-nacional.xlsx",
        "energia-injetada-na-rede-de-distribuicao.xlsx",
        "consumo-total-nacional.xlsx"
    ]

    dataframes = {}

    for url, file_name in zip(urls, file_names):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info(f"Downloading {file_name}")
                with open(file_name, "wb") as file:
                    file.write(response.content)

                df = pd.read_excel(file_name)
                dataframes[file_name] = df

                os.remove(file_name)
            else:
                logger.error(f"Failed to download {file_name}. HTTP Status: {response.status_code}")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")

    return dataframes


def fetch_local_data():
    """
    Reads local CSV and Excel files from the 'Data' project folder and loads them into pandas DataFrames.

    Returns:
        dict: A dictionary where keys are dataset names and values are pandas DataFrames.
    """
    data_folder = "Data"

    # Dictionary containing file information: filename, delimiter (for CSV), and rows to skip
    file_info = {
        "Day-ahead Market Prices_20230101_20250304.csv": {"delimiter": ";", "skiprows": 2},
        "ML_Consumption_Data.csv": {"delimiter": ",", "skiprows": 0},
        "Breakdown of Production_20230101_20250326.xlsx": {"sheet_name": 0, "skiprows": 2},
        "icap-graph-price-data-2020-01-01-2025-02-13.csv": {"delimiter": ",", "skiprows": 1},
    }

    dataframes = {}

    for file_name, params in file_info.items():
        file_path = os.path.join(data_folder, file_name)

        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path, sep=params["delimiter"], skiprows=params["skiprows"])
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path, sheet_name=params["sheet_name"], skiprows=params["skiprows"])
            else:
                logger.warning(f"Unsupported file format: {file_name}")
                continue

            dataframes[file_name] = df
            logger.info(f"Successfully read {file_name}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_name}")
        except Exception as e:
            logger.error(f"Error reading {file_name}: {str(e)}")

    return dataframes


def prepare_redes_data(output_path, update=False):
    """
    Loads remote and local datasets, processes and merges them and
    writes the final merged DataFrame to 'redes_data.xlsx'.
    """
    output_dir = os.path.join(output_path, "e-redes-data")
    output_file = os.path.join(output_dir, "redes_data.xlsx")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # If update is False and the output file exists, load and return it immediately.
    if not update and os.path.exists(output_file):
        return pd.read_excel(output_file)

    # Fetch data
    remote_data = fetch_redes_data()
    local_data = fetch_local_data()

    try:
        ms_edp = local_data.get("ML_Consumption_Data.csv").copy()
        energy_prices = local_data.get("Day-ahead Market Prices_20230101_20250304.csv").copy()
        ren_data = local_data.get("Breakdown of Production_20230101_20250326.xlsx").copy()
        ets_df = local_data.get("icap-graph-price-data-2020-01-01-2025-02-13.csv").copy()

        con_data = remote_data.get("consumo-total-nacional.xlsx").copy()
        prod_data = remote_data.get("energia-injetada-na-rede-de-distribuicao.xlsx").copy()
        prod_data_alt = remote_data.get("produzida-total-nacional.xlsx").copy()

    except AttributeError as e:
        logger.error("One or more datasets could not be loaded properly: " + str(e))
        return None

    prod_data_alt.rename(columns={'Total (kWh)': 'Total Prod (kWh)'}, inplace=True)
    con_data.rename(columns={'Total (kWh)': 'Total Con (kWh)'}, inplace=True)

    # Process energy_prices
    energy_prices['Hour'] = energy_prices['Hour'].replace({24: 0, 25: 1})
    energy_prices['Hour'] = energy_prices['Hour'].astype(str).str.zfill(2)
    # Create timestamp column with correct formatting
    energy_prices['timestamp'] = energy_prices['Date'] + 'T' + energy_prices['Hour'] + ':00:00+00:00'
    energy_prices['Date/Time'] = pd.to_datetime(energy_prices['timestamp'], format='%Y-%m-%dT%H:%M:%S%z')
    energy_prices = energy_prices[['Date/Time', 'Portugal', 'Spain']]

    # Rename columns to reflect price per MW in Euros
    energy_prices = energy_prices.rename(columns={
        'Portugal': 'Portugal Price (€/MW)',
        'Spain': 'Spain Price (€/MW)'
    })

    # Merge consumption and production datasets
    merge_keys = ['Date/Time', 'Day', 'Month', 'Year', 'Date', 'Hour']
    merged_data = pd.merge(con_data, prod_data, on=merge_keys, how='outer')
    merged_data = pd.merge(merged_data, prod_data_alt, on=merge_keys, how='outer')
    merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'], utc=True)

    # Merge with prices and directly interpolate for not hourly
    merged_data = merged_data.sort_values('Date/Time')
    energy_prices = energy_prices.sort_values('Date/Time')
    merged_data = pd.merge_asof(merged_data, energy_prices, on='Date/Time', direction='backward')

    # Process monthly data (ms_edp)
    month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    ms_edp['Month_Num'] = ms_edp['Month'].str[:3].map(month_mapping)
    ms_edp['Year'] = ms_edp['Month'].str[-4:].astype(int)
    ms_edp.drop(columns=['Month', '∆ to month of previous year (%)', 'ML Consumo (GWh)'], inplace=True)

    # Merge monthly data
    df = pd.merge(merged_data, ms_edp, left_on=['Month', 'Year'],
                  right_on=['Month_Num', 'Year'], how='outer')

    df.drop(columns=['Month_Num'], inplace=True)

    del merged_data, energy_prices, con_data, prod_data, prod_data_alt

    # -------------------------
    # Process REN data (local breakdown of production)
    # -------------------------
    if ren_data is not None:
        # Convert 'Date and Time' to datetime and drop the original column
        ren_data['Date/Time'] = pd.to_datetime(ren_data['Date and Time']) \
            .dt.tz_localize('Europe/Lisbon', ambiguous='infer') \
            .dt.tz_convert('UTC')
        ren_data.drop(columns=['Date and Time'], inplace=True)
        # Convert all numeric columns from MW to kWh (multiply by 1000 and 0.25 for 15min interval)
        numeric_cols = ren_data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            ren_data[col] = ren_data[col] * 1000 * 0.25
        # Rename columns to overwrite corresponding energy values in df
        ren_data.rename(columns={
            'Wind': 'Wind (kWh)',
            'Hydro': 'Hydro (kWh)',
            'Pumping': 'Pumping (kWh)',
            'Solar': 'Photovoltaics (kWh)',
            'Battery Injection': 'Battery Injection (kWh)',
            'Battery Consumption': 'Battery Consumption (kWh)',
            'Biomass': 'Biomass (kWh)',
            'Wave': 'Wave (kWh)',
            'Natural Gas': 'Natural Gas (kWh)',
            'Coal': 'Coal (kWh)',
            'Oil': 'Oil (kWh)',
            'Other Thermal': 'Other Thermal (kWh)',
            'Imports': 'Imports (kWh)',
            'Exports': 'Exports (kWh)',
            'Consumption': 'Consumption (kWh)'
        }, inplace=True)

        # Merge REN data with df using merge_asof (assuming hourly data with up to 30 minutes difference)
        df = df.sort_values('Date/Time')
        ren_data = ren_data.sort_values('Date/Time')
        df = pd.merge_asof(df, ren_data, on='Date/Time', direction='nearest',
                           tolerance=pd.Timedelta("30min"), suffixes=('', '_ren'))
        # Overwrite main columns with REN values where available and drop the extra ones
        for col in ['Wind (kWh)', 'Hydro (kWh)', 'Photovoltaics (kWh)']:
            if col + '_ren' in df.columns:
                df[col] = df[col + '_ren'].combine_first(df[col])
                df.drop(columns=[col + '_ren'], inplace=True)

    # -------------------------
    # Process ETS prices from local data
    # -------------------------
    if ets_df is not None:
        ets_df = ets_df.copy()
        ets_df['Date/Time'] = pd.to_datetime(ets_df['Date'], utc=True)
        # Rename the price column – here we use the 'Primary Market' column as the ETS price
        ets_df.rename(columns={"Primary Market": "ETS price"}, inplace=True)
        # Merge ETS prices with the main dataframe via merge_asof
        df = df.sort_values('Date/Time')
        ets_df = ets_df.sort_values('Date/Time')
        df = pd.merge_asof(df, ets_df[['Date/Time', 'ETS price']], on='Date/Time', direction='backward')

    # -------------------------
    # Additional processing
    # -------------------------
    portugal_holidays = holidays.country_holidays('PT')
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], utc=True)
    df['Holiday'] = df['Date/Time'].dt.date.apply(lambda x: 1 if x in portugal_holidays else 0)
    df['Month'] = df['Date/Time'].dt.month
    df['Weekday'] = df['Date/Time'].dt.day_name()

    season_mapping = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['Season'] = df['Month'].map(season_mapping)

    # Convert energy values from kWh to GWh
    conversion_columns = {
        'Total Con (kWh)': 'Total Con (GWh)',
        'Total Prod (kWh)': 'Total Prod (GWh)',
        'Cogeneration (kWh)': 'Cogeneration (GWh)',
        'Wind (kWh)': 'Wind (GWh)',
        'Photovoltaics (kWh)': 'Photovoltaics (GWh)',
        'Hydro (kWh)': 'Hydro (GWh)',
        'Other Technologies (kWh)': 'Other Technologies (GWh)',
        ' Distribution Network (kWh)': 'Distribution Network (GWh)'
    }
    for old_col, new_col in conversion_columns.items():
        df[new_col] = df[old_col] / 1_000_000

    df['Residual Load (GWh)'] = df['Total Con (GWh)'] - (df['Distribution Network (GWh)'] - df['Other Technologies (GWh)'])
    df['Total Con (GWh) EDP'] = df['Total Con (GWh)'] * (df['Overall EDP (%)'] / 100)

    df.drop(columns=['Low Voltage (kWh)', 'Medium Voltage (kWh)', 'High Voltage (kWh)', 'Very High Voltage (kWh)'], inplace=True)

    # Save to Excel
    df['Date/Time'] = df['Date/Time'].dt.tz_localize(None)
    df.to_excel(output_file, index=False)
    logger.info("redes_data.xlsx has been created successfully.")