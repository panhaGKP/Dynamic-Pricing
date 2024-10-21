import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import modules for machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor # type: ignore
from joblib import dump

#import snowflake package
import snowflake.snowpark.functions as F
from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Price Recommendation", page_icon="🏷️")

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

def get_list_models_path_demand_forecasting(session, product_sku):
    model_path_query = f"""
    SELECT
        MODEL_PATH,
        CUSTOMER_SEGMENT
    FROM 
        FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED 
    WHERE 
        PRODUCT_SKU = '{product_sku}' ORDER BY CUSTOMER_SEGMENT
    """

    # Execute the query and collect the results
    model_paths_result = session.sql(model_path_query).collect()

    # Extract the model names from the result
    model_paths = [row['MODEL_PATH'] for row in model_paths_result]

    # Print the list of model names
    return model_paths

# ============ End of function Definitions =========
# ============ Main Code ==============

session = get_active_session()

st.title('Pricing Recommendation App')

# read list of products from snowflake LU_PRD_PRODUCT_SKU_PREDICTED
product_model_table = "LU_PRD_PRODUCT_SKU_PREDICTED"
demand_forecasting_model_table = "FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED"
product_list = session.sql(f"""
    SELECT  
        PRODUCT_SKU
    FROM 
        {product_model_table}
    ORDER BY 
        DATE_MODIFIED DESC
""").collect()

product = [product[0] for product in product_list]
#product = ['B33_S3','B15_S1', 'B13_S1', 'B31_S32', 'B44_S4', 'B15_S47']
#print(product)

product = st.selectbox(
    'Choose a Product',
    product
)
selected_product_retail_price = session.sql(f"""
    SELECT 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE 
    FROM 
        {product_model_table}                                    
    JOIN 
        LU_PRD_PRODUCT_SKU 
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = {product_model_table}.PRODUCT_SKU 
    WHERE 
        {product_model_table}.PRODUCT_SKU = '{product}'
""").collect()
#print(type(selected_product_retail_price))
#print(selected_product_retail_price)
product_retail_price_display = selected_product_retail_price[0]['PRODUCT_SKU_RETAIL_PRICE']
# print(product_retail_price)
st.caption(f"{product} base price: \\${product_retail_price_display:.2f}")

# find min and max volumn of select product
product_info = session.sql(f"""
    SELECT 
        {product_model_table}.MIN_VOLUMN_SOLD AS "MIN_VOLUMN_SOLD", 
        {product_model_table}.MAX_VOLUMN_SOLD AS "MAX_VOLUMN_SOLD",
        {product_model_table}.MODEL_PATH AS "MODEL_PATH",
        {product_model_table}.MODEL_NAME AS "MODEL_NAME",
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE "RETAIL_PRICE",
        LU_PRD_PRODUCT_SKU.LATEST_DATE AS "LATEST_DATE_SOLD"
    FROM 
        {product_model_table} 
    JOIN 
        LU_PRD_PRODUCT_SKU 
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = {product_model_table}.PRODUCT_SKU 
    WHERE 
        {product_model_table}.PRODUCT_SKU = '{product}'
""").collect()

lastest_sold = product_info[0]['LATEST_DATE_SOLD'] + timedelta(days=1)
after_lastest_sold = product_info[0]['LATEST_DATE_SOLD'] + timedelta(days=7)

start_date, end_date = st.date_input(
    "Select Date Range for Forecast",
    value = [lastest_sold, after_lastest_sold],  # Default range: today to one week from today
    min_value = lastest_sold  # Setting the minimum selectable date as today
)

num_days_selected = (end_date - start_date).days + 1
# print(f"Number of days selected: {num_days_selected}")
min_volume = max(product_info[0]['MIN_VOLUMN_SOLD'], num_days_selected)  # Ensure minimum volume is at least number of days
max_volume = product_info[0]['MAX_VOLUMN_SOLD']
retail_price = product_info[0]['RETAIL_PRICE']

volume = st.number_input(
    f"Enter the Expected Volume Sold (Recommended Range: {min_volume} - {max_volume*num_days_selected})",  # Label for the input
    min_value=min_volume,  # Minimum value based on the product
    value=min_volume*100,  # Default value
    step=1 # Step for incrementing/decrementing
)

# ======Distribute the volume across the selected days by using Weight from ARIMA model =======
# last_date = product_info[0]['LATEST_DATE_SOLD']
# print(f"Last Date Sold: {last_date}")
# regular_date = datetime.fromordinal(last_date)
cluster_members = [1,2, 3, 4, 5, 6, 7, 8, 9]
    # Create a list of cluster samples for each day
cluster_samples = [[clusters]*num_days_selected for clusters in cluster_members]

#===Debugging===
# print("forecast_df_for_each_segment")
# print(forecast_df_for_each_segment)
# print("Volumn Distribution")
# print(volume_distribution)
# print("Cluster Samples")
print(cluster_samples)

# print(f"Volume Distribution: {volume_distribution}")
# Button to Show Images and Prices
if st.button('Pricing Recommendations'):
    st.markdown('---')
    st.subheader('Product Pricing Recommended Range')
    forecast_index = pd.date_range(start=lastest_sold, periods=num_days_selected, freq='D')
    # print(forecast_index)
    model_demand_forecasting_files = get_list_models_path_demand_forecasting(session, product)
    demand_predict_result = {}
    for segment, model_file in enumerate(model_demand_forecasting_files):
        file_name = os.path.basename(model_file)
        result = session.sql(f"""
            SELECT udf_demand_forecast_prediction('{file_name}', {num_days_selected}) as DEMAND_FORECAST
        """).collect()
        json_result = json.loads(result[0]['DEMAND_FORECAST'])
        json_result_dict = json.loads(json_result)
        demand_predict_result[segment] = json_result_dict
        print(f"Segment {segment+1} Forecast: {json_result_dict['forecast']}")

    forecast_df_for_each_segment = {}
    for segment, result in demand_predict_result.items():
        # Calculate the total units sold in the forecast
        forecast_df_for_each_segment[segment] = pd.DataFrame(result['forecast'], index=forecast_index, columns=['UNITS_SOLD'])
        total_units_sold = forecast_df_for_each_segment[segment]['UNITS_SOLD'].sum()
        forecast_df_for_each_segment[segment]['WEIGHT_PERCENTAGE'] = (forecast_df_for_each_segment[segment]['UNITS_SOLD'] / total_units_sold) * 100

    volume_distribution_for_each_segment = {}
    for segment, df in forecast_df_for_each_segment.items():
        volume_distribution_for_each_segment[segment] = ((df['WEIGHT_PERCENTAGE']/100) * volume).round().astype(int)

    list_of_dates = [start_date + timedelta(days=i) for i in range(num_days_selected)]
    # Three clusters of customers (But there will be more)
    


    each_cluster_average_discount_rate = []
    for cluster_number in cluster_members:
        print(f"Cluster {cluster_number}")
        predict_discount_rate_model_file = os.path.basename(product_info[0]['MODEL_PATH'])
        predict_discount_rate_model_name = product_info[0]['MODEL_NAME']
        volume_sold_by_date = (forecast_df_for_each_segment[cluster_number-1]['WEIGHT_PERCENTAGE'] / 100) * volume
        sample_data = {
            #'UNITS_SOLD': volume_distribution,
            'UNITS_SOLD': volume_sold_by_date.to_list(),
            'DATE': list_of_dates,
            #'CLUSTER': [1,1,1,1,1,1,1,1],
            'CUSTOMER_SEGMENT': cluster_samples[cluster_number],
        }
        df_sample = pd.DataFrame(sample_data)

        # ================= Data Preprocessing Steps =====================
        df_sample['DATE'] = pd.to_datetime(df_sample['DATE'])
        df_sample['DATE_ORDINAL'] = df_sample['DATE'].apply(lambda x: x.toordinal())
        # Drop the original DATE column
        df_sample = df_sample.drop(columns=['DATE'])

        # Predict using the trained model
        # Reorder the columns of df_sample
        df_sample = df_sample[['DATE_ORDINAL', 'UNITS_SOLD', 'CUSTOMER_SEGMENT']]
        # print(df_sample)
        
        snowpark_df_sample = session.create_dataframe(df_sample)

        # ================= Predict the Average Discount Rate from UDFs snowflake===================
        predictions_set = snowpark_df_sample.select(
            F.col("DATE_ORDINAL"),
            F.col("UNITS_SOLD"),
            F.col("CUSTOMER_SEGMENT"),
            F.call_udf("udf_avg_discount_rate_prediction_v2", 
                snowpark_df_sample['DATE_ORDINAL'],
                snowpark_df_sample['UNITS_SOLD'], 
                snowpark_df_sample['CUSTOMER_SEGMENT'], 
                f"{predict_discount_rate_model_file}").alias('PREDICTED_AVG_DISCOUNT_RATE')
        ).collect()
        # print("predictions_set")
        # print(predictions_set)
        # Extract the 'PREDICTED_AVG_DISCOUNT_RATE' column from each row in predictions_set
        predicted_discount_rates = [row['PREDICTED_AVG_DISCOUNT_RATE'] for row in predictions_set]

        # Calculate the minimum value from the 'PREDICTED_AVG_DISCOUNT_RATE' column
        min_discount_rate = min(predicted_discount_rates)
        max_discount_rate = max(predicted_discount_rates)

        # print(f"Cluster {cluster_number} - Min Discount Rate: {min_discount_rate}, Max Discount Rate: {max_discount_rate}")
        # Store min and max discount rate for each cluster
        each_cluster_average_discount_rate.append({
            'min_discount_rate': min_discount_rate,
            'max_discount_rate': max_discount_rate
        })

    print("Each Cluster Average Discount Rate")  
    print(each_cluster_average_discount_rate)

    min_price_cs1 = retail_price * (1 - each_cluster_average_discount_rate[0]['min_discount_rate'])
    max_price_cs1 = retail_price * (1 - each_cluster_average_discount_rate[0]['max_discount_rate'])
    min_price_cs2 = retail_price * (1 - each_cluster_average_discount_rate[1]['min_discount_rate'])
    max_price_cs2 = retail_price * (1 - each_cluster_average_discount_rate[1]['max_discount_rate'])
    min_price_cs3 = retail_price * (1 - each_cluster_average_discount_rate[2]['min_discount_rate'])
    max_price_cs3 = retail_price * (1 - each_cluster_average_discount_rate[2]['max_discount_rate'])

    col1, col2, col3 = st.columns(3)

    
    with col1:
        st.image('assets/CS 1.png', caption='Customer Segment 1') 
        st.write(f"Price Range: \\${min_price_cs1:.2f} - \\${max_price_cs1:.2f}")

    
    with col2:
        st.image('assets/CS 2.png', caption='Customer Segment 2')
        st.write(f"Price Range: \\${min_price_cs2:.2f} - \\${max_price_cs2:.2f}")

    
    with col3:
        st.image('assets/CS 3.png', caption='Product Image 3')
        st.write(f"Price Range: \\${min_price_cs3:.2f} - \\${max_price_cs3:.2f}")

    st.markdown('---')
    st.subheader('Additional Information Range')

    
    st.write(f"Selected Product: {product}")
    st.write(f"Product Base Price: \\${retail_price}")
    st.write(f"Expected Volume Sold: {volume} units")
    st.write(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")