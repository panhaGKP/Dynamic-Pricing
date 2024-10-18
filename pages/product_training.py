import streamlit as st
import os
import pandas as pd
import numpy as np
import cachetools
import json

import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T

from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor # type: ignore
from joblib import dump
from snowflake.snowpark.types import StructType, StructField, IntegerType, FloatType, StringType


# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session


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

def get_modile_file_name_from_stage(session, stage_name):
    files = session.sql(f"LIST @{stage_name}").collect()
    model_files = [file['name'] for file in files if file['name'].endswith('.sav')]

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

def insert_trained_result_to_snowflake(session, product_sku, df_model, model_file_name, evaluation_metrics, hyperparameters):
    product_trained = session.sql(f"""
        SELECT * FROM LU_PRD_PRODUCT_TRAINED WHERE PRODUCT_SKU = '{product_sku}'
    """).collect()

    if len(product_trained) > 0:
        st.write("Product used to be trained, and being updated")
        session.sql(f"""
            UPDATE LU_PRD_PRODUCT_TRAINED 
            SET 
                PRODUCT_SKU = '{product_sku}',
                MIN_VOLUMN_SOLD = {df_model['UNITS_SOLD'].min()},
                MAX_VOLUMN_SOLD = {df_model['UNITS_SOLD'].max()},
                DATE_MODIFIED = CURRENT_TIMESTAMP(),
                MODEL_PATH = '{model_file_name}',
                MODEL_NAME = 'xgboost',
                EVALUATION = PARSE_JSON('{json.dumps(evaluation_metrics)}'),
                HYPER_PARAMS = PARSE_JSON('{json.dumps(hyperparameters)}')
            WHERE PRODUCT_SKU = '{product_sku}'
        """).collect()

    else:
        print("Product is new to be trained")
        session.sql(f"""
            INSERT INTO LU_PRD_PRODUCT_TRAINED (PRODUCT_SKU, MIN_VOLUMN_SOLD, MAX_VOLUMN_SOLD, DATE_CREATED, DATE_MODIFIED, MODEL_PATH, MODEL_NAME, EVALUATION, HYPER_PARAMS)
            SELECT '{product_sku}', {df_model['UNITS_SOLD'].min()}, {df_model['UNITS_SOLD'].max()}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), '{model_file_name}', 'xgboost', PARSE_JSON('{json.dumps(evaluation_metrics)}'), PARSE_JSON('{json.dumps(hyperparameters)}')
        """).collect()

# ============ End of function Definitions =========
# ============ Main Code ==============
session = get_active_session()

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
        AND LU_PRD_PRODUCT_SKU.PRODUCT_SKU NOT IN (SELECT PRODUCT_SKU FROM LU_PRD_PRODUCT_TRAINED)
        
    GROUP BY PRD_PRODUCT_SKU
    ORDER BY GS DESC;
""").collect()

gs_psku_df = convert_list_to_df(gs_psku_list, ["PRODUCT_SKU", "GROSS_SALES"])

products_trained = gs_psku_df["PRODUCT_SKU"].tolist()
#products = ['B15_S1', 'B13_S1', 'B31_S32', 'B44_S4', 'B15_S47']
# Product input - String list
product_sku_trained_selected = st.selectbox(
    'Choose a Product',
    products_trained  # Add or change the product names as needed
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
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product_sku_trained_selected}')
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
    st.write(f"Training model for product: {product_sku_trained_selected}") 
    st.write(f"Data Collection")
    raw_data = session.sql(f"""
        SELECT 
            FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE AS "DATE",
            LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE AS "RETAIL_RPICE",
            REL_CUS_SEGMENTATION_RFM.CLUSTER AS "CUSTOMER_SEGMENT",
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
            REL_CUS_SEGMENTATION_RFM
        ON
            REL_CUS_SEGMENTATION_RFM.CUSTOMER_CODE = FACT_ORDER_LINE_ITEM_BASE.CUSTOMER_CODE
        WHERE
            LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product_sku_trained_selected}')
            AND (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
        GROUP BY
            DATE, RETAIL_RPICE, CUSTOMER_SEGMENT
        ORDER BY
            DATE ASC
    """).collect()

    columns = ["DATE", "RETAIL_RPICE", "CUSTOMER_SEGMENT", "NUM_ORDERS", "UNITS_SOLD", "AVG_DISCOUNT_RATE", "COGS"]
    df_raw_data = pd.DataFrame(raw_data, columns=columns)

    # ============ Data preprocessing part ==============
    st.write(f"Data Preprocessing")
    # Change the data types of the columns
    df_raw_data['UNITS_SOLD'] = df_raw_data['UNITS_SOLD'].astype(int)
    df_raw_data['AVG_DISCOUNT_RATE'] = df_raw_data['AVG_DISCOUNT_RATE'].astype(float)
    df_raw_data['COGS'] = df_raw_data['COGS'].astype(float)

    # Outlier treatment
    outlier_percentages = {}
    numerical_columns = ['NUM_ORDERS', 'UNITS_SOLD', 'AVG_DISCOUNT_RATE', 'COGS']
    for column in numerical_columns:
        Q1 = df_raw_data[column].quantile(0.25)
        Q3 = df_raw_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_raw_data[(df_raw_data[column] < lower_bound) | (df_raw_data[column] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df_raw_data)) * 100
        outlier_percentages[column] = outlier_percentage
    df_raw_data = treat_outliers(df_raw_data, 'NUM_ORDERS', outlier_percentages)
    df_raw_data = treat_outliers(df_raw_data, 'UNITS_SOLD', outlier_percentages)
    df_raw_data = treat_outliers(df_raw_data, 'AVG_DISCOUNT_RATE', outlier_percentages)
    df_raw_data = treat_outliers(df_raw_data, 'COGS', outlier_percentages)

    # Drop missing values
    df_raw_data.dropna(inplace=True)
    # Drop the duplicate rows
    df_raw_data.drop_duplicates(inplace=True)

    # Convert the DATE column to datetime
    df_raw_data['DATE'] = pd.to_datetime(df_raw_data['DATE'])
    df_raw_data['DATE_ORDINAL'] = df_raw_data['DATE'].apply(lambda x: x.toordinal())

    retail_price = df_raw_data['RETAIL_RPICE'][0]
    # ============== End of Data Preprocessing Part ==============
    # ============== Feature Engineering Part ==============
    st.write(f"Feature Engineering")
    # feature selection
    df_model = df_raw_data[['DATE_ORDINAL', 'UNITS_SOLD', 'CUSTOMER_SEGMENT', 'AVG_DISCOUNT_RATE']]
    df_model['CUSTOMER_SEGMENT'] = df_model['CUSTOMER_SEGMENT'].astype('int')
    # ============== End of Feature Engineering Part ==============

    # =============== Data Splitting Part ==============
    st.write(f"Data Splitting")
    train_set, test_set = train_test_split(df_model, test_size=0.2, random_state=42)
    df_model['PRODUCT_SKU'] = product_sku_trained_selected
    # =============== End of Data Splitting Part ==============

    # ============== Store preprocessed data to snowflake ==============
    st.write(f"Insert Clean Data to snowflake!")
    training_table = "FACT_DYNAMIC_PRICING_CLEANED"
    # schema structure for pandas DataFrame convert to Snowpark DataFrame
    schema = StructType([
            StructField("DATE_ORDINAL", IntegerType()),
            StructField("UNITS_SOLD", IntegerType()),
            StructField("CUSTOMER_SEGMENT", IntegerType()),
            StructField("AVG_DISCOUNT_RATE", FloatType()),
            StructField("PRODUCT_SKU", StringType())
        ])
    create_table_if_not_exists(training_table)
    snowpark_df = insert_pandas_dataframe_to_snowflake(session, df_model, training_table, schema)
    # ============== End of storing preprocessed data to snowflake ==============

    # =============== Create a stage for the model ==============
    st.write(f"Stage Creation if not exists")
    stage_name = "DYNAMIC_PRICING_MODEL_STAGE"
    session.sql(f"CREATE STAGE IF NOT EXISTS {stage_name}").collect()

    # =============== Model Training Part ==============
    st.write(f"Model Training start.........")
    training_table = "FACT_DYNAMIC_PRICING_CLEANED"
    model_file_name = f"{product_sku_trained_selected}_xgb_model.sav"
    feature_cols = ['DATE_ORDINAL','UNITS_SOLD','CUSTOMER_SEGMENT']
    target_col = 'AVG_DISCOUNT_RATE'
    params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    # calling the stored procedure to train and save the model
    result = session.call(
        "sproc_train_xgb_model1",
        training_table,
        feature_cols,
        target_col,
        model_file_name,
        stage_name,
        params
    )
    # =============== Model Evaluation Part ==============
    st.write(f"Model Evaluation")
    st.write(result)
    # ----- Write code where all models are compared and best model is selected ----

    # =============== End of Model Evaluation Part ==============
    model_files = get_modile_file_name_from_stage(session, stage_name)

    # re-defined udf_avg_discount_rate_prediction with new import model list
    session.udf.register(
        func=avg_discount_rate_prediction,
        name="udf_avg_discount_rate_prediction",
        stage_location=stage_name,
        input_type=[T.IntegerType(), T.IntegerType(), T.IntegerType(), T.StringType()],  # Three integers and one string
        return_type=T.FloatType(),  # The return type is a float
        replace=True,
        is_permanent=True,
        # imports=[f"@{stage_name}/{model_file_name}"],  # Model file is imported,
        imports=[f"@{stage_name}/{model}" for model in model_files], # multiple model files are imported, 
        packages=["joblib", "cachetools", "xgboost","pandas"],  # Required packages for the UDF
    )

    # =============== End of Model Training Part ==============
    
    # =============== Store result to snowflake =================
    st.write(f"Store result to snowflake!")

    parsed_result = json.loads(result)
    evaluation_metrics = {
        "mse": parsed_result['mse']
    }
    hyperparameters = {
        "best_params": parsed_result['best_params']
    }
    # push trained product to snowflake

    insert_trained_result_to_snowflake(session, product_sku_trained_selected, df_model, model_file_name, evaluation_metrics, hyperparameters)
    # =============== End of storing result to snowflake ==============
    st.write("Model training process completed successfully")                         