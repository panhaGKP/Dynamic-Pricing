import streamlit as st
import os
import pandas as pd
import numpy as np
import cachetools
import json

from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor # type: ignore
from joblib import dump
from statsmodels.tsa.stattools import adfuller

# Import snowflake packages
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session


st.set_page_config(page_title="Product Training", page_icon="😊")
# Title for the app
st.title('Product Training Portal')


# ============ function Definitions part ==============
# Refresh Snowflake session after 60 minites
@st.cache_resource(ttl=3600)
def get_active_session():
    #get account credentials from
    load_dotenv()
    connection_parameters = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "query_tag": "streamlit-app product-training"
    }
    return Session.builder.configs(connection_parameters).create()

def convert_list_to_df(data, columns):
    return pd.DataFrame(data, columns=columns)

def treat_outliers(df, column, outlier_percentages):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # if any of the numerical columns have outliers that under 5%, we can keep them, otherwise we treat theme with mean imputation
    if outlier_percentages[column] > 5:
        df[column] = np.where(df[column] < lower_bound, df[column].mean(), df[column])
        df[column] = np.where(df[column] > upper_bound, df[column].mean(), df[column])
    return df

def create_table_if_not_exists(table_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        DATE_ORDINAL INT,
        UNITS_SOLD INT,
        CUSTOMER_SEGMENT INT,
        AVG_DISCOUNT_RATE FLOAT,
        PRODUCT_SKU STRING
    );
    """
    session.sql(create_table_query).collect()  # Execute the query

def insert_pandas_dataframe_to_snowflake(session, pandas_df, table_name, schema):
    # Convert pandas DataFrame to Snowpark DataFrame
    snowpark_df = session.create_dataframe(pandas_df, schema=schema)
    product_sku = pandas_df['PRODUCT_SKU'][0]
    # Write the Snowpark DataFrame into Snowflake table
    session.sql(f"DELETE FROM {table_name} WHERE PRODUCT_SKU = '{product_sku}'").collect()
    snowpark_df.write.mode("append").save_as_table(f"{table_name}")
    return snowpark_df

def get_modile_file_name_from_stage(session, stage_name, extension):
    files = session.sql(f"LIST @{stage_name}").collect()
    model_files = [file['name'] for file in files if file['name'].endswith(extension)]

    #insert file name without prefix dynamic_pricing_model_stage/
    model_files = [file.replace(f"{stage_name.lower()}/", "") for file in model_files]
    return model_files # list data type

@cachetools.cached(cache={})
def load_model(filename):

    # Import packages
    import sys
    import os
    import joblib

    # Get the import directory where the model file is stored
    import_dir = sys._xoptions.get("snowflake_import_directory")

    # Get the import directory where the model file is stored
    if import_dir:
        with open(os.path.join(import_dir, filename), "rb") as f:
            model = joblib.load(f)
            return model

# Function to predict the average discount rate
def avg_discount_rate_prediction(date_ordinal: int, units_sold: int, customer_segment: int, model_name: str) -> float:
    import pandas as pd
    # Create a DataFrame from the input values (this assumes feature_cols)
    feature_cols = ['DATE_ORDINAL', 'UNITS_SOLD', 'CUSTOMER_SEGMENT']
    X = pd.DataFrame([[date_ordinal, units_sold, customer_segment]], columns=feature_cols)

    # Load the model using the model name
    model = load_model(model_name)

    # Get predictions
    predictions = model.predict(X)

    # Return the predicted value (assuming a regression model returning a single value)
    return predictions[0]

def insert_trained_result_to_snowflake(session, product_sku, df_model, model_name, model_file_name, stg_dir, evaluation_metrics, hyperparameters):
    model_path = f"{stg_dir}/{model_file_name}"
    product_trained = session.sql(f"""
    SELECT * FROM LU_PRD_PRODUCT_TRAINED WHERE PRODUCT_SKU = '{product_sku}' AND MODEL_NAME = '{model_name}'
    """).collect()

    if len(product_trained) > 0:
        print("Product trained, now updated")
        session.sql(f"""
            UPDATE LU_PRD_PRODUCT_TRAINED 
            SET 
                PRODUCT_SKU = '{product_sku}',
                MIN_VOLUMN_SOLD = {df_model['UNITS_SOLD'].min()},
                MAX_VOLUMN_SOLD = {df_model['UNITS_SOLD'].max()},
                DATE_MODIFIED = CURRENT_TIMESTAMP(),
                MODEL_PATH = '{model_path}',
                MODEL_NAME = '{model_name}',
                EVALUATION = PARSE_JSON('{json.dumps(evaluation_metrics)}'),
                HYPER_PARAMS = PARSE_JSON('{json.dumps(hyperparameters)}')
            WHERE PRODUCT_SKU = '{product_sku}' AND MODEL_NAME = '{model_name}'
        """).collect()

    else:
        print("Product not trained")
        session.sql(f"""
            INSERT INTO LU_PRD_PRODUCT_TRAINED (PRODUCT_SKU, MIN_VOLUMN_SOLD, MAX_VOLUMN_SOLD, DATE_CREATED, DATE_MODIFIED, MODEL_PATH, MODEL_NAME, EVALUATION, HYPER_PARAMS)
            SELECT 
                '{product_sku}',
                {df_model['UNITS_SOLD'].min()},
                {df_model['UNITS_SOLD'].max()}, 
                CURRENT_TIMESTAMP(), 
                CURRENT_TIMESTAMP(), 
                '{model_path}',
                '{model_name}',
                PARSE_JSON('{json.dumps(evaluation_metrics)}'),
                PARSE_JSON('{json.dumps(hyperparameters)}')
        """).collect()

def insert_best_trained_result_to_snowflake(session, table_name, product_sku, df_model, model_name, model_file_name, stg_dir, evaluation_metrics, hyperparameters):
    model_path = f"{stg_dir}/{model_file_name}"
    product_trained = session.sql(f"""
    SELECT * FROM {table_name} WHERE PRODUCT_SKU = '{product_sku}' AND MODEL_NAME = '{model_name}'
    """).collect()

    if len(product_trained) > 0:
        print("Product trained, now updated")
        session.sql(f"""
            UPDATE {table_name} 
            SET 
                PRODUCT_SKU = '{product_sku}',
                MIN_VOLUMN_SOLD = {df_model['UNITS_SOLD'].min()},
                MAX_VOLUMN_SOLD = {df_model['UNITS_SOLD'].max()},
                DATE_MODIFIED = CURRENT_TIMESTAMP(),
                MODEL_PATH = '{model_path}',
                MODEL_NAME = '{model_name}',
                EVALUATION = PARSE_JSON('{json.dumps(evaluation_metrics)}'),
                HYPER_PARAMS = PARSE_JSON('{json.dumps(hyperparameters)}')
            WHERE PRODUCT_SKU = '{product_sku}' AND MODEL_NAME = '{model_name}'
        """).collect()

    else:
        print("Product not trained")
        session.sql(f"""
            INSERT INTO {table_name} (PRODUCT_SKU, MIN_VOLUMN_SOLD, MAX_VOLUMN_SOLD, DATE_CREATED, DATE_MODIFIED, MODEL_PATH, MODEL_NAME, EVALUATION, HYPER_PARAMS)
            SELECT 
                '{product_sku}',
                {df_model['UNITS_SOLD'].min()},
                {df_model['UNITS_SOLD'].max()}, 
                CURRENT_TIMESTAMP(), 
                CURRENT_TIMESTAMP(), 
                '{model_path}',
                '{model_name}',
                PARSE_JSON('{json.dumps(evaluation_metrics)}'),
                PARSE_JSON('{json.dumps(hyperparameters)}')
        """).collect()

# query and filling data with 0 is done in Snowflake
def get_demand_data(product_sku, customer_segment):
    data = session.sql(f"""
    WITH sales_data AS (
    SELECT 
        FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE AS "DATE",
        LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE AS "CUSTOMER_SEGMENT",
        SUM(FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY) AS "UNITS_SOLD"
    FROM 
        FACT_ORDER_LINE_ITEM_BASE
    LEFT JOIN 
        LU_PRD_PRODUCT
    ON 
        FACT_ORDER_LINE_ITEM_BASE.PRODUCT_CODE = LU_PRD_PRODUCT.PRODUCT_CODE
    LEFT JOIN 
        LU_PRD_PRODUCT_SKU
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT.PRODUCT_SKU
    LEFT JOIN 
        LU_CUS_CUSTOMER 
    ON 
        FACT_ORDER_LINE_ITEM_BASE.CUSTOMER_CODE = LU_CUS_CUSTOMER.CUSTOMER_CODE
    LEFT JOIN 
        LU_CUS_RFM_SEGMENT
    ON 
        LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE = LU_CUS_CUSTOMER.RFM_SEGMENT_CODE
    WHERE 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN ('{product_sku}')
        AND LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE = {customer_segment}
    GROUP BY 
        FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE, LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE
),

date_range AS (
    SELECT 
        MIN(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) AS MIN_DATE,
        MAX(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) AS MAX_DATE
    FROM 
        FACT_ORDER_LINE_ITEM_BASE
    LEFT JOIN 
        LU_PRD_PRODUCT
    ON 
        FACT_ORDER_LINE_ITEM_BASE.PRODUCT_CODE = LU_PRD_PRODUCT.PRODUCT_CODE
    LEFT JOIN
        LU_PRD_PRODUCT_SKU
    ON 
        LU_PRD_PRODUCT.PRODUCT_SKU = LU_PRD_PRODUCT_SKU.PRODUCT_SKU
    WHERE 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN ('{product_sku}')
)

SELECT 
    LU_CAL_DATE.DATE AS "DATE",
    COALESCE(sales_data.CUSTOMER_SEGMENT, {customer_segment}) AS "CUSTOMER_SEGMENT",
    COALESCE(sales_data.UNITS_SOLD, 0) AS "UNITS_SOLD"
FROM 
    LU_CAL_DATE
LEFT JOIN 
    sales_data
ON 
    LU_CAL_DATE.DATE = sales_data.DATE
JOIN 
    date_range
ON 
    LU_CAL_DATE.DATE BETWEEN date_range.MIN_DATE AND date_range.MAX_DATE
WHERE 
    YEAR(LU_CAL_DATE.DATE) BETWEEN 2021 AND 2023
ORDER BY 
    LU_CAL_DATE.DATE ASC;
    """).collect()
    return data

def treat_outliers_on_demand_forecast(data):
    # Ensure that UNITS_SOLD is numeric (converting all types to float)
    data['UNITS_SOLD'] = pd.to_numeric(data['UNITS_SOLD'], errors='coerce')
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(data['UNITS_SOLD']))
    
    # Find outliers where the z-score is greater than 3
    outliers = np.where(z_scores > 3)[0]
    
    # Replace outliers with the mean value of UNITS_SOLD
    mean_value = data['UNITS_SOLD'].mean()
    data.loc[outliers, 'UNITS_SOLD'] = mean_value
    
    return data

# Function to create the table if it doesn't exist
def create_table_demand_forcast_cleaned_if_not_exists(table_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        DATE DATE,
        CUSTOMER_SEGMENT INT,
        UNITS_SOLD INT,
        PRODUCT_SKU STRING
    );
    """
    session.sql(create_table_query).collect()  # Execute the query

def demand_forecast_prediction(
    model_file_name: str,
    n_periods: int
)-> T.Variant:
    # Import packages
    import json
    # Load the model
    model = load_model(model_file_name)

    # Make the prediction
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

    result = {
        'forecast': forecast.tolist(),
        'confidence_interval': conf_int.tolist()
    }
    # result string
    result_str = json.dumps(result)
    return result_str

def get_avg_discount_rate_data(product_sku):
    raw_data = session.sql(f"""
    SELECT 
        FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE AS "DATE",
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE AS "RETAIL_RPICE",
        LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE AS "CUSTOMER_SEGMENT",
        COUNT(DISTINCT(FACT_ORDER_LINE_ITEM_BASE.ORDER_CODE)) AS "NUM_ORDERS",
        SUM(FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY) AS "UNITS_SOLD",
        AVG(FACT_ORDER_LINE_ITEM_BASE.DISCOUNT_RATE) AS "AVG_DISCOUNT_RATE",
        SUM(FACT_ORDER_LINE_ITEM_BASE.SUPPLY_UNIT_COST) AS "COGS",
    FROM 
        LU_PRD_PRODUCT_SKU
    JOIN 
        LU_PRD_PRODUCT
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT.PRODUCT_SKU
    JOIN 
        FACT_ORDER_LINE_ITEM_BASE 
    ON 
        FACT_ORDER_LINE_ITEM_BASE.PRODUCT_CODE = LU_PRD_PRODUCT.PRODUCT_CODE
    JOIN 
        LU_CUS_CUSTOMER 
    ON
        LU_CUS_CUSTOMER.CUSTOMER_CODE = FACT_ORDER_LINE_ITEM_BASE.CUSTOMER_CODE
    JOIN 
        LU_CUS_RFM_SEGMENT
    ON
        LU_CUS_RFM_SEGMENT.RFM_SEGMENT_CODE = LU_CUS_CUSTOMER.RFM_SEGMENT_CODE
    WHERE
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product_sku}')
        AND (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
    GROUP BY
        DATE, RETAIL_RPICE, CUSTOMER_SEGMENT
    ORDER BY
        DATE ASC
    """).collect()
    return raw_data
# ============ End of function Definitions =========
# ============ Global Variables =========
training_table = "FACT_DYNAMIC_PRICING_CLEANED"
demand_forecast_training_table = "FACT_DEMAND_FORECASTING_CLEANED"
stage_name = "DYNAMIC_PRICING_MODEL_STAGE"
target_col = 'UNITS_SOLD'
session = get_active_session()
customer_segement = [1, 2, 3, 4, 5, 6, 7, 8, 9]
numerical_columns = ['NUM_ORDERS', 'UNITS_SOLD', 'AVG_DISCOUNT_RATE', 'COGS']
feature_cols = ['DATE_ORDINAL', 'UNITS_SOLD', 'CUSTOMER_SEGMENT']

udf_avg_discount_rate_prediction_name = "udf_avg_discount_rate_prediction_v2"
#============= End of Global Variables ==
# ============ Main Code ==============


gs_psku_list = session.sql("""
    SELECT
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU AS "PRD_PRODUCT_SKU",
        SUM(FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY * FACT_ORDER_LINE_ITEM_BASE.RETAIL_PRICE) AS "GS"
    FROM
        FACT_ORDER_LINE_ITEM_BASE
    JOIN 
        LU_PRD_PRODUCT
    ON
        LU_PRD_PRODUCT.PRODUCT_CODE = FACT_ORDER_LINE_ITEM_BASE.PRODUCT_CODE
    JOIN 
        LU_PRD_PRODUCT_SKU 
    ON
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT.PRODUCT_SKU
    WHERE
        (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
        AND LU_PRD_PRODUCT_SKU.PRODUCT_SKU NOT IN (SELECT PRODUCT_SKU FROM LU_PRD_PRODUCT_SKU_PREDICTED)
        
    GROUP BY PRD_PRODUCT_SKU
    ORDER BY GS DESC;
""").collect()

gs_psku_df = convert_list_to_df(gs_psku_list, ["PRODUCT_SKU", "GROSS_SALES"])

products_trained_options = gs_psku_df["PRODUCT_SKU"].tolist()
#products = ['B15_S1', 'B13_S1', 'B31_S32', 'B44_S4', 'B15_S47']
# Product input - String list
product_sku_to_train_selected = st.selectbox(
    'Choose a Product',
    products_trained_options  # Add or change the product names as needed
)

weekly_units_sold_by_product = session.sql(f"""
    SELECT 
        DATE_TRUNC('week', FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) AS "WEEK",
        SUM(FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY) AS "UNITS_SOLD",  
    FROM 
        LU_PRD_PRODUCT_SKU
    JOIN 
        LU_PRD_PRODUCT
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT.PRODUCT_SKU
    JOIN 
        FACT_ORDER_LINE_ITEM_BASE 
    ON 
        FACT_ORDER_LINE_ITEM_BASE.PRODUCT_CODE = LU_PRD_PRODUCT.PRODUCT_CODE
    WHERE
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product_sku_to_train_selected}')
        AND (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
    GROUP BY
        WEEK
    ORDER BY
        WEEK ASC
""").collect()
columns = ["WEEK", "UNITS_SOLD"]
weekly_units_sold_by_product = convert_list_to_df(weekly_units_sold_by_product, columns)
weekly_units_sold_by_product['WEEK'] = pd.to_datetime(weekly_units_sold_by_product['WEEK'])
weekly_units_sold_by_product['UNITS_SOLD'] = weekly_units_sold_by_product['UNITS_SOLD'].astype('int')
#  ====== Data Exploration Part =========
# plot the area chart
st.area_chart(weekly_units_sold_by_product, x="WEEK", y="UNITS_SOLD", color=["#00CCDD"])
# !will add more visual analysis here


# ===== End of Data Exploration Part =====
# find min and max volumn of select product
if st.button("Train Model for this Product"):
    # ============ Data collection Part ==============
    st.markdown('---')
    st.subheader(f"Training Model for Product {product_sku_to_train_selected}")
    st.markdown("Demand forecasting with ```ARIMA``` model")

    with st.status("Data Collection on Product Demand Trend...", expanded=True) as status:
        data_collection = {}
        for segment in customer_segement:
            data = pd.DataFrame(get_demand_data(product_sku_to_train_selected, segment))
            data_collection[segment] = data
            st.write(f"Data collection for Customer Segment {segment} complete!")

        status.update(
            label="Data Collection on Product Demand Trend, complete!", state="complete", expanded=False
        )

    # ============ End of Data collection Part ==============
    # ============ Data Preprocessing Part ==============
    with st.status("Data Preprocessing...", expanded=True) as status:
        for segment in data_collection:
            data_collection[segment] = treat_outliers_on_demand_forecast(data_collection[segment])
            st.write(f"Outliers treatment for Customer Segment {segment} complete!")
        status.update(
            label="Data Preprocessing complete!", state="complete", expanded=False
        )
    # ============ End of Data Preprocessing Part ==============
    combined_data = pd.concat(data_collection.values(), ignore_index=True)
    # Display the combined DataFrame
    combined_data['PRODUCT_SKU'] = product_sku_to_train_selected
    combined_data['UNITS_SOLD'] = combined_data['UNITS_SOLD'].astype(int)
    demand_forecast_data_schema = StructType([
        StructField("DATE", DateType()),
        StructField("CUSTOMER_SEMENT", IntegerType()),
        StructField("UNITS_SOLD", IntegerType()),
        StructField("PRODUCT_SKU", StringType())
    ])

    with st.status("Insert Cleaned Data to snowflake...", expanded=True) as status:                
        create_table_demand_forcast_cleaned_if_not_exists(training_table)
        demand_forecast_snowpark_df = insert_pandas_dataframe_to_snowflake(session, combined_data, demand_forecast_training_table, demand_forecast_data_schema)
        st.write("Create table if not exist for demand forecast cleaned data complete!")
        session.sql(f"CREATE STAGE IF NOT EXISTS {stage_name}").collect()   
        st.write("Create stage if not exist for model training complete!")
        status.update(
            label="Insert Cleaned Data to snowflake complete!", state="complete", expanded=False
        )


    with st.status("Train the model with Clean Data for Demand Forecasting...") as status4:
        # ====== Statistical Check part (Stationarity) ======
        stationarity_results = {}
        for segment, data in data_collection.items():
            if data['UNITS_SOLD'].nunique() == 1:
                # Skip this segment as UNIT_SOLD is constant
                continue
            result = adfuller(data['UNITS_SOLD'])
            stationarity_results[segment] = {
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Stationary': result[1] <= 0.05
            }
            # Unit Solds is all Zero, continue to next segment
            if result[1] > 0.05:
                continue
        stationary_num = sum([1 for segment, result in stationarity_results.items() if result['Stationary']])
        st.write(f"Stationarity Check for each segment complete!, There are {stationary_num} segments are stationary")
        # ====== End of Statistical Check part (Stationarity) ======

        # ====== Model Training Part ======
        training_results = {}
        for segment, result in stationarity_results.items():
            if result['Stationary']:
                # Fit the auto_arima model
                model_file_demand_forecast = f"{product_sku_to_train_selected}_segment_{segment}.pkl"
                stage_dir =  f"{stage_name}/demand_forecast_models/{product_sku_to_train_selected}"
                # calling the stored procedure to train and save the model
                result = session.call(
                    "sproc_train_arima_model_v1",
                    demand_forecast_training_table,
                    product_sku_to_train_selected,
                    target_col, # Target Variable
                    model_file_demand_forecast,
                    segment, # customer segment
                    stage_dir
                )
                # convert string to dictionary
                result_set = json.loads(result)
                training_results[segment] = result_set
                st.write(f"Model training for Customer Segment {segment} complete!")
        
        # Save trained result to Snowflake
        for segment, result in training_results.items():
            model_name = 'ARIMA'
            model_file_name = f"{product_sku_to_train_selected}_segment_{segment}.pkl" # i.e B33_S3_segment_1.pkl
            model_path = f"{stage_name}/demand_forecast_models/{product_sku_to_train_selected}/{model_file_name}" 
            hyper_params = {
                'best_params' : result['best_params'],
                'best_order' : result['best_order']
            }
            evaluation = {
                'aic': result['aic'],
                'bic': result['bic']
            }
            product_trained = session.sql(f"""
                SELECT * FROM FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED WHERE PRODUCT_SKU = '{product_sku_to_train_selected}' AND CUSTOMER_SEGMENT = {segment}
            """).collect()

            if len(product_trained) > 0:
                print("Product Trained, Now Updated")
                session.sql(f"""
                    UPDATE FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED
                    SET 
                        DATE_MODIFIED = CURRENT_TIMESTAMP(),
                        MODEL_PATH = '{model_path}',
                        MODEL_NAME = '{model_name}',
                        HYPER_PARAMS = PARSE_JSON('{json.dumps(hyper_params)}'),
                        EVALUATION = PARSE_JSON('{json.dumps(evaluation)}')
                    WHERE 
                        PRODUCT_SKU = '{product_sku_to_train_selected}' AND CUSTOMER_SEGMENT = {segment}
                """).collect()
            else:
                session.sql(f"""
                INSERT INTO FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED (PRODUCT_SKU, CUSTOMER_SEGMENT, DATE_CREATED, DATE_MODIFIED, MODEL_PATH, MODEL_NAME, HYPER_PARAMS, EVALUATION)
                SELECT 
                    '{product_sku_to_train_selected}',
                    {segment},
                    CURRENT_TIMESTAMP(),
                    CURRENT_TIMESTAMP(),
                    '{model_path}',
                    '{model_name}',
                    PARSE_JSON('{json.dumps(hyper_params)}'),
                    PARSE_JSON('{json.dumps(evaluation)}')
                """).collect()
        #st.write("Save trained result to Snowflake complete!")
        
        files = session.sql(f"LIST @{stage_name}").collect()
        model_demand_forecasting_files = [file['name'] for file in files if file['name'].endswith('.pkl')]
        #insert file name without prefix dynamic_pricing_model_stage/
        model_demand_forecasting_files = [file.replace(f"{stage_name.lower()}/", "") for file in model_demand_forecasting_files]
        session.udf.register(
            func=demand_forecast_prediction,
            name="udf_demand_forecast_prediction",
            stage_location=stage_name,
            input_type=[T.StringType(), T.IntegerType()],  # Three integers and one string
            return_type=T.VariantType(),  # The return type is a variant
            replace=True,
            is_permanent=True,
            # imports=[f"@{stage_name}/{model_file_name}"],  # Model file is imported,
            imports=[f"@{stage_name}/{model}" for model in model_demand_forecasting_files], # multiple model files are imported, 
            packages=["joblib", "cachetools","pmdarima"],  # Required packages for the UDF
        )

        st.write("Register the trained model to UDFs complete!")
        status4.update(
            label="Train the model with Clean Data for Demand Forecasting complete!", state="complete", expanded=False
        )
    # ============ End of Model Training Part of ARIMA Model ==============
    st.markdown("Train data for Average Discount Prediction model with ```XGBoost``` and ```DecisionTree``` model")
    # ============ Data collection Part ==============
    with st.status("Data Collection on Product Demand Trend...", expanded=True) as status:
        avg_discount_rate_raw_data = get_avg_discount_rate_data(product_sku_to_train_selected)
        st.write("Data collection for Customer Segment 1 complete!")
        status.update(
            label="Data Collection on Product Demand Trend, complete!", state="complete", expanded=False
        )
    # ============ End of Data collection Part For Average discount rate prediction ==============
    # ============ Data Preprocessing Part ==============
    with st.status("Data Preprocessing...", expanded=True) as status:
        avg_discount_rate_df_data = pd.DataFrame(avg_discount_rate_raw_data)
        avg_discount_rate_df_data['UNITS_SOLD'] = avg_discount_rate_df_data['UNITS_SOLD'].astype(int)
        avg_discount_rate_df_data['AVG_DISCOUNT_RATE'] = avg_discount_rate_df_data['AVG_DISCOUNT_RATE'].astype(float)
        avg_discount_rate_df_data['COGS'] = avg_discount_rate_df_data['COGS'].astype(float)
        
        outlier_percentages = {}
        for column in numerical_columns:
            Q1 = avg_discount_rate_df_data[column].quantile(0.25)
            Q3 = avg_discount_rate_df_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = avg_discount_rate_df_data[(avg_discount_rate_df_data[column] < lower_bound) | (avg_discount_rate_df_data[column] > upper_bound)]
            outlier_percentage = (len(outliers) / len(avg_discount_rate_df_data)) * 100
            outlier_percentages[column] = outlier_percentage

        df_raw_data = treat_outliers(avg_discount_rate_df_data, 'AVG_DISCOUNT_RATE', outlier_percentages)
        df_raw_data = treat_outliers(avg_discount_rate_df_data, 'COGS', outlier_percentages)
        st.write("Outliers treatment complete!")
        #Convert Customer Segment to Integer
        avg_discount_rate_df_data['CUSTOMER_SEGMENT'] = avg_discount_rate_df_data['CUSTOMER_SEGMENT'].astype('int')
        avg_discount_rate_df_data['DATE'] = pd.to_datetime(avg_discount_rate_df_data['DATE'])
        avg_discount_rate_df_data['DATE_ORDINAL'] = avg_discount_rate_df_data['DATE'].apply(lambda x: x.toordinal())
        st.write("Type conversion complete!")
        df_selected = avg_discount_rate_df_data.drop(columns=['RETAIL_RPICE', 'DATE'])
        df_model = avg_discount_rate_df_data[['DATE_ORDINAL', 'UNITS_SOLD', 'CUSTOMER_SEGMENT', 'AVG_DISCOUNT_RATE']].copy()
        df_model['PRODUCT_SKU'] = product_sku_to_train_selected
        st.write("Feature Selection complete!")
        status.update(
            label="Data Preprocessing complete!", state="complete", expanded=False
        )
    # ============ End of Data Preprocessing Part ==============
    # ============= Insert Cleaned Data to Snowflake ==============
    with st.status("Insert Cleaned Data to snowflake...", expanded=True) as status:
        avg_discount_rate_cleaned_date_schema = StructType([
            StructField("DATE_ORDINAL", IntegerType()),
            StructField("UNITS_SOLD", IntegerType()),
            StructField("CUSTOMER_SEGMENT", IntegerType()),
            StructField("AVG_DISCOUNT_RATE", FloatType()),
            StructField("PRODUCT_SKU", StringType())
        ])
        create_table_if_not_exists(training_table)
        avg_discount_rate_prediction_snowpark_df = insert_pandas_dataframe_to_snowflake(session, df_model, training_table, avg_discount_rate_cleaned_date_schema)
        #st.write("Insert Cleaned Data to snowflake complete!")
        status.update(
            label="Insert Cleaned Data to snowflake complete!", state="complete", expanded=False
        )
    # ============= End of Insert Cleaned Data to Snowflake ==============
    # ============= Model Training Part with XGBoost ==============
    stage_avg_discount_prediction_model_dir =  f"{stage_name}/avg_discount_rate_prediction_models"
    with st.status("Train cleaned data with XGBoost model...") as status:
        model_xgb_file_name = f"{product_sku_to_train_selected}_xgb_model.sav"
        target_col_for_avg_discount_rate_pred = 'AVG_DISCOUNT_RATE'
        # can only put 2 hyperparameters for each, since snowflake COMPUTE_WH is limited
        xgb_params_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Call the training store procedure by feeding the required parameters
        xgb_train_result = session.call(
            "sproc_train_xgb_model_v2",
            training_table,
            feature_cols, 
            target_col_for_avg_discount_rate_pred, 
            model_xgb_file_name,
            stage_avg_discount_prediction_model_dir, # where model file will be stored,
            xgb_params_grid
        )
        result_xgboost_model = json.loads(xgb_train_result)
        # Evaluation Metrics
        evaluation_metrics = {
            "mse_test_set": result_xgboost_model['mse_test_set'],
            "mse_train_set": result_xgboost_model['mse_train_set'],
            "r_squared": result_xgboost_model['r_squared']
        }

        hyperparameters = {
            "best_params": result_xgboost_model['best_params']
        }
        st.write(f"Result of XGBoost model training: {result_xgboost_model}") 

        # Save trained result to Snowflake
        insert_trained_result_to_snowflake(
            session, 
            product_sku_to_train_selected, 
            df_model, 
            'xgboost',
            model_xgb_file_name, 
            stage_avg_discount_prediction_model_dir, 
            evaluation_metrics, 
            hyperparameters
        )
        st.write("Save trained result to Snowflake complete!")
        status.update(
            label="Train cleaned data with XGBoost model complete!", state="complete", expanded=False
        )

    with st.status("Train cleaned data with DecisionTree model...") as status:
        model_dt_file_name = f"{product_sku_to_train_selected}_dt_model.sav"
        # can only put 2 hyperparameters for each, since snowflake COMPUTE_WH is limited
        dt_params_grid = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }
        # Call the training store procedure by feeding the required parameters
        dt_train_result = session.call(
            "sproc_train_decision_tree_model_v1",
            training_table,
            feature_cols, 
            target_col, 
            model_dt_file_name,
            stage_avg_discount_prediction_model_dir, # where model file will be stored,
            dt_params_grid
        )
        result_dt_model = json.loads(dt_train_result)
        dt_evaluation_metrics = {
            "mse_test_set": result_dt_model['mse_test_set'],
            "mse_train_set": result_dt_model['mse_train_set'],
            "r_squared": result_dt_model['r_squared']
        }
        dt_hyperparameters = {
            "best_params": result_dt_model['best_params']
        }
        st.write(f"Result of DecisionTree model training: {result_dt_model}") 

        # Save trained result to Snowflake
        insert_trained_result_to_snowflake(
            session, 
            product_sku_to_train_selected, 
            df_model,
            'decision_tree', 
            model_dt_file_name, 
            stage_avg_discount_prediction_model_dir, 
            dt_evaluation_metrics, 
            dt_hyperparameters
        )
        st.write("Save trained result to Snowflake complete!")
        status.update(
            label="Train cleaned data with DecisionTree model complete!", state="complete", expanded=False
        )
    
    all_model_result = {
        'xgboost': {
            'model_file_name': model_xgb_file_name,
            'evaluation_metrics': evaluation_metrics,
            'hyperparameters': hyperparameters,
        },
        'decision_tree': {
            'model_file_name': model_dt_file_name,
            'evaluation_metrics': dt_evaluation_metrics,
            'hyperparameters': dt_hyperparameters,
        }
    }

    with st.status("Average Discount Rate Model Selection...") as status:
        xgboost_mse_test = result_xgboost_model['mse_test_set']
        decision_tree_mse_test = result_dt_model['mse_test_set']

        st.write(f"XGBoost Model Test MSE: {xgboost_mse_test}")
        st.write(f"DecisionTree Model Test MSE: {decision_tree_mse_test}")
        best_model_str = ""
        if xgboost_mse_test < decision_tree_mse_test:
            best_model_str = "xgboost"
            st.write("XGBoost model performs better based on MSE.")
        else:
            best_model_str = "decision_tree"
            st.write("DecisionTree model performs better based on MSE.") 
        lu_prd_product_sku_trained = 'LU_PRD_PRODUCT_SKU_PREDICTED'
        insert_best_trained_result_to_snowflake(
            session,
            lu_prd_product_sku_trained,
            product_sku_to_train_selected, 
            df_model,
            best_model_str,
            all_model_result[best_model_str]['model_file_name'], 
            stage_avg_discount_prediction_model_dir, 
            all_model_result[best_model_str]['evaluation_metrics'], 
            all_model_result[best_model_str]['hyperparameters']
        )
        st.write("Save best trained result to Snowflake (LU_PRD_PRODUCT_SKU_PREDICTED) complete!")

        status.update(
            label="Average Discount Rate Model Selection complete!", state="complete", expanded=False
        )
    # ============= End of Model Training Part with Decision Tree ==============

    # ============= Register model to UDFs ==============
   
    with st.status("Register the trained model to UDFs...") as status:
        avg_discount_rate_pred_model_files = get_modile_file_name_from_stage(session, stage_avg_discount_prediction_model_dir,'.sav')
        session.udf.register(
            func=avg_discount_rate_prediction,
            name=udf_avg_discount_rate_prediction_name,
            stage_location=stage_avg_discount_prediction_model_dir,
            input_type=[T.IntegerType(), T.IntegerType(), T.IntegerType(), T.StringType()],  # Three integers and one string
            return_type=T.FloatType(),  # The return type is a float
            replace=True,
            is_permanent=True,
            imports=[f"@{stage_avg_discount_prediction_model_dir}/{model}" for model in avg_discount_rate_pred_model_files], # multiple model files are imported, 
            packages=["joblib", "cachetools", "xgboost","pandas", "scikit-learn"],  # Required packages for the UDF
        )
        st.write("Register the trained model to UDFs complete!")
        status.update(
            label="Register the trained model to UDFs complete!", state="complete", expanded=False
        )

