import streamlit as st
import time
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
        "query_tag": "streamlit-app product_price_recommendation"
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
    model_paths_result = session.sql(model_path_query).collect()
    model_paths = [row['MODEL_PATH'] for row in model_paths_result]
    return model_paths

def convert_to_ordinal(dates):
    # Convert each date in the DatetimeIndex to ordinal using the toordinal() method
    return [date.to_pydatetime().toordinal() for date in dates]

# ============ End of function Definitions =========
# ============ Global Variables =========
session = get_active_session()
product_model_table = "LU_PRD_PRODUCT_SKU_PREDICTED"
demand_forecasting_model_table = "FACT_PRODUCT_SKU_CUS_DEMAND_TRAINED"

#============= End of Global Variables ==
# ============ Main Code ==============
st.title('Product Price Recommendation')

# read list of products from snowflake LU_PRD_PRODUCT_SKU_PREDICTED
product_trained_list = session.sql(f"""
    SELECT  
        PRODUCT_SKU
    FROM 
        {product_model_table}
    ORDER BY 
        DATE_MODIFIED DESC
""").collect()

if 'product_list_options' not in st.session_state or st.session_state.product_list_options is None:
    # If not, generate the product list and store it in session state
    product_list_options = [product[0] for product in product_trained_list]
    st.session_state.product_list_options = product_list_options
else:
    # If it's already in session state, use it
    product_list_options = st.session_state.product_list_options

# Initialize session state for product selection
if 'product_selected' not in st.session_state:
    st.session_state.product_selected = None

product_selected = st.selectbox(
    'Choose a Product',
    product_list_options,
    index=product_list_options.index(st.session_state.product_selected) if st.session_state.product_selected else 0
)
st.session_state.product_selected = product_selected
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
        {product_model_table}.PRODUCT_SKU = '{product_selected}'
""").collect()

product_retail_price_display = selected_product_retail_price[0]['PRODUCT_SKU_RETAIL_PRICE']

st.caption(f"{product_selected} base price: \\${product_retail_price_display:.2f}")

# find min and max volumn of select product
product_selected_info = session.sql(f"""
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
        {product_model_table}.PRODUCT_SKU = '{product_selected}'
""").collect()

product_lastest_sold_date = product_selected_info[0]['LATEST_DATE_SOLD'] + timedelta(days=1)
after_lastest_sold = product_selected_info[0]['LATEST_DATE_SOLD'] + timedelta(days=7)

start_date, end_date = st.date_input(
    "Select Date Range for Forecast",
    value = [product_lastest_sold_date, after_lastest_sold],  # Default range: today to one week from today
    min_value = product_lastest_sold_date  # Setting the minimum selectable date as today
)

num_days_selected = (end_date - start_date).days + 1
# print(f"Number of days selected: {num_days_selected}")
product_min_volume_sold = max(product_selected_info[0]['MIN_VOLUMN_SOLD'], num_days_selected)  # Ensure minimum volume is at least number of days
product_max_volume_sold = product_selected_info[0]['MAX_VOLUMN_SOLD']
product_retail_price = product_selected_info[0]['RETAIL_PRICE']
product_avg_discount_predict_model_path = product_selected_info[0]['MODEL_PATH']
product_avg_discount_predict_model_name = product_selected_info[0]['MODEL_NAME']

expected_units_sold = st.number_input(
    f"Enter the Expected Volume Sold (Recommended Range: {product_min_volume_sold} - {product_max_volume_sold*num_days_selected})",  # Label for the input
    min_value=product_min_volume_sold,  # Minimum value based on the product
    value=product_min_volume_sold*100,  # Default value
    step=1 # Step for incrementing/decrementing
)

# ======Distribute the volume across the selected days by using Weight from ARIMA model =======
customer_segment_members = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Create a list of cluster samples for each day
customer_segment_samples = [[customer_segment]*num_days_selected for customer_segment in customer_segment_members]

#===Debugging===
# print("forecast_df_for_each_segment")
# print(forecast_df_for_each_segment)
# print("Volumn Distribution")
# print(volume_distribution)
# print("Customer Segment Samples:")
# print(customer_segment_samples)

# print(f"Number of periods to forecast: {n_periods}")
# print(f"Volume Distribution: {volume_distribution}")
# Button to Show Images and Prices

if st.button('Pricing Recommendations'):
    st.markdown('---')
    st.subheader('Product Pricing Recommended Range')
    # forecast_index = pd.date_range(start=lastest_sold, periods=num_days_selected, freq='D')
    # # print(forecast_index)
    # model_demand_forecasting_files = get_list_models_path_demand_forecasting(session, product)
    # demand_predict_result = {}

    # number of periods to forecast = last_date_of_sales - end_date
    n_periods = (end_date - product_lastest_sold_date).days + 1
    forecast_index = pd.date_range(start=product_lastest_sold_date, periods=n_periods, freq='D')
    # print(forecast_index)
    my_bar = st.progress(0, text=f"Customer Segment 1: Operation in progress. Please wait.")   
    price_ranges = []
    for customer_segment in customer_segment_samples:
        progress_text = f"Customer Segment {customer_segment[0]}: Operation in progress. Please wait."
        percent_complete = customer_segment[0] / len(customer_segment_samples)
        my_bar.progress(percent_complete, text=progress_text)
        # Demand Forecasting for weight percentage of customer segment 
        customer_segment_number = customer_segment[0]
        product_demand_forecast_info = session.sql(f"""
            SELECT * FROM {demand_forecasting_model_table}
            WHERE PRODUCT_SKU = '{product_selected}' AND CUSTOMER_SEGMENT = {customer_segment_number}                                       
        """).collect()
        demand_forecast_model_path = product_demand_forecast_info[0]['MODEL_PATH']
        demand_forecast_model_name = product_demand_forecast_info[0]['MODEL_NAME']
        demand_forecast_file = os.path.basename(demand_forecast_model_path)
        demand_predict_result = session.sql(f"""
            SELECT udf_demand_forecast_prediction('{demand_forecast_file}', {n_periods}) as DEMAND_FORECAST
        """).collect()
        json_result = json.loads(demand_predict_result[0]['DEMAND_FORECAST'])
        json_result_dict = json.loads(json_result)

        forecast_df = pd.DataFrame(json_result_dict['forecast'], index=forecast_index, columns=['UNITS_SOLD_FORECAST'])
        total_units_sold = forecast_df['UNITS_SOLD_FORECAST'].sum()
        forecast_df['WEIGHT_PERCENTAGE'] = (forecast_df['UNITS_SOLD_FORECAST'] / total_units_sold) * 100
        volume_sold_by_date = (forecast_df['WEIGHT_PERCENTAGE'] / 100) * expected_units_sold
        volume_sold_by_date_selected = volume_sold_by_date[-num_days_selected:]

        # Predict Average Discount Rate
        avg_discount_prediction_file = os.path.basename(product_avg_discount_predict_model_path)
        ordinal_dates_selected = convert_to_ordinal(forecast_index[-num_days_selected:])
        avg_discount_predicted_sample = {
            'UNITS_SOLD': volume_sold_by_date_selected.to_list(),
            'DATE_ORDINAL': ordinal_dates_selected,
            'CUSTOMER_SEGMENT': customer_segment
        }
        avg_discount_predicted_sample_snowpark_df = session.create_dataframe(pd.DataFrame(avg_discount_predicted_sample))
        avg_discount_predict_result = avg_discount_predicted_sample_snowpark_df.select(
        "UNITS_SOLD",
        "DATE_ORDINAL",
        "CUSTOMER_SEGMENT",
        F.call_udf("udf_avg_discount_rate_prediction_v2", 
                avg_discount_predicted_sample_snowpark_df['DATE_ORDINAL'],
                avg_discount_predicted_sample_snowpark_df['UNITS_SOLD'], 
                avg_discount_predicted_sample_snowpark_df['CUSTOMER_SEGMENT'], 
                f"{avg_discount_prediction_file}").alias('PREDICTED_AVG_DISCOUNT_RATE')
        )
        min_discount_rate = avg_discount_predict_result.select(F.min(avg_discount_predict_result['PREDICTED_AVG_DISCOUNT_RATE'])).collect()[0][0]
        max_discount_rate = avg_discount_predict_result.select(F.max(avg_discount_predict_result['PREDICTED_AVG_DISCOUNT_RATE'])).collect()[0][0]
        # st.write(f"Customer Segment {customer_segment_number}: Min Discount Rate: {min_discount_rate}, Max Discount Rate: {max_discount_rate}")

        new_price_max_xgb = product_retail_price * (1 - min_discount_rate)
        new_price_min_xgb = product_retail_price * (1 - max_discount_rate)
        
        
        price_ranges.append({
            'Customer Segment': customer_segment_number,
            'Min Price': new_price_min_xgb,
            'Max Price': new_price_max_xgb,
        })
        # st.write(f"Customer Segment {customer_segment_number}, volumn sold by date: {volume_sold_by_date.to_list()}")
        # st.write(f"Customer Segment {customer_segment_number}, Forecast: {json_result_dict['forecast']}")

    my_bar.empty()
    # Display the DataFrame
    price_ranges_df = pd.DataFrame(price_ranges)
    st.dataframe(price_ranges_df)


    # col1, col2, col3 = st.columns(3)
    # min_price_cs1 = 999999
    # max_price_cs1 = 0
    # min_price_cs2 = 999999
    # max_price_cs2 = 0
    # min_price_cs3 = 999999
    # max_price_cs3 = 0
    # with col1:
    #     st.image('assets/CS 1.png', caption='Customer Segment 1') 
    #     st.write(f"Price Range: \\${min_price_cs1:.2f} - \\${max_price_cs1:.2f}")

    
    # with col2:
    #     st.image('assets/CS 2.png', caption='Customer Segment 2')
    #     st.write(f"Price Range: \\${min_price_cs2:.2f} - \\${max_price_cs2:.2f}")

    
    # with col3:
    #     st.image('assets/CS 3.png', caption='Product Image 3')
    #     st.write(f"Price Range: \\${min_price_cs3:.2f} - \\${max_price_cs3:.2f}")



    # ======= Additional Information =====
    st.markdown('---')
    st.subheader('Additional Information Range')
    st.write(f"Selected Product: {product_selected}")
    st.write(f"Product Base Price: \\${product_retail_price:.2f}")
    st.write(f"Expected Volume Sold: {expected_units_sold} units")
    st.write(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")
    # ======= End of Additional Information =====