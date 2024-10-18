import streamlit as st
import pandas as pd
import numpy as np
import os
# import snowflake.connector as sf # type: ignore

# from dotenv import load_dotenv # type: ignore
# from joblib import load
from datetime import datetime, timedelta

#import snowflake model
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor # type: ignore
from joblib import dump
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

# ============ End of function Definitions =========
# ============ Main Code ==============

session = get_active_session()

st.title('Pricing Recommendation App')

# read list of products from snowflake LU_PRD_PRODUCT_TRAINED
product_list = session.sql("""
    SELECT  
        PRODUCT_SKU
    FROM 
        LU_PRD_PRODUCT_TRAINED
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
    FROM LU_PRD_PRODUCT_TRAINED 
        JOIN LU_PRD_PRODUCT_SKU 
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU 
    WHERE 
        LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU = '{product}'
""").collect()
#print(type(selected_product_retail_price))
#print(selected_product_retail_price)
product_retail_price_display = selected_product_retail_price[0]['PRODUCT_SKU_RETAIL_PRICE']
# print(product_retail_price)
st.caption(f"{product} base price: \\${product_retail_price_display:.2f}")

# find min and max volumn of select product
min_max_volume_retail_price_latest_sales = session.sql(f"""
    SELECT 
        LU_PRD_PRODUCT_TRAINED.MIN_VOLUMN_SOLD AS "MIN_VOLUMN_SOLD", 
        LU_PRD_PRODUCT_TRAINED.MAX_VOLUMN_SOLD AS "MAX_VOLUMN_SOLD",
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE "RETAIL_PRICE",
        LU_PRD_PRODUCT_SKU.LATEST_DATE AS "LATEST_DATE_SOLD"
    FROM 
        LU_PRD_PRODUCT_TRAINED 
    JOIN 
        LU_PRD_PRODUCT_SKU 
    ON 
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU 
    WHERE 
        LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU = '{product}'
""").collect()

lastest_sold = min_max_volume_retail_price_latest_sales[0]['LATEST_DATE_SOLD'] + timedelta(days=1)
after_lastest_sold = min_max_volume_retail_price_latest_sales[0]['LATEST_DATE_SOLD'] + timedelta(days=7)

start_date, end_date = st.date_input(
    "Select Date Range for Forecast",
    value = [lastest_sold, after_lastest_sold],  # Default range: today to one week from today
    min_value = lastest_sold  # Setting the minimum selectable date as today
)

num_days_selected = (end_date - start_date).days + 1
# print(f"Number of days selected: {num_days_selected}")
min_volume = max(min_max_volume_retail_price_latest_sales[0]['MIN_VOLUMN_SOLD'], num_days_selected)  # Ensure minimum volume is at least number of days
max_volume = min_max_volume_retail_price_latest_sales[0]['MAX_VOLUMN_SOLD']
retail_price = min_max_volume_retail_price_latest_sales[0]['RETAIL_PRICE']

volume = st.number_input(
    f"Enter the Expected Volume Sold (Recommended Range: {min_volume} - {max_volume*num_days_selected})",  # Label for the input
    min_value=min_volume,  # Minimum value based on the product
    value=min_volume*100,  # Default value
    step=1 # Step for incrementing/decrementing
)

# ======Distribute the volume across the selected days=======
volume_per_day = volume // num_days_selected  # Base volume per day
remaining_volume = volume % num_days_selected  # Remaining volume to distribute
# Create a list to store the volume distribution
volume_distribution = [volume_per_day] * num_days_selected

# Add bias to the first `remaining_volume` days
for i in range(remaining_volume):
    volume_distribution[i] += 1  # Distribute the extra volume to the earlier days

list_of_dates = [start_date + timedelta(days=i) for i in range(num_days_selected)]
# Three clusters of customers (But there will be more)
cluster_members = [0,1,2]
# Create a list of cluster samples for each day
cluster_samples = [[clusters]*num_days_selected for clusters in cluster_members]

#===Debugging===
# print("Volumn Distribution")
# print(volume_distribution)
# print("Cluster Samples")
# print(cluster_samples)

# print(f"Volume Distribution: {volume_distribution}")
# Button to Show Images and Prices
if st.button('Pricing Recommendations'):
    st.markdown('---')
    st.subheader('Product Pricing Recommended Range')

    

    each_cluster_average_discount_rate = []
    for cluster_number in cluster_members:
        sample_data = {
            'UNIT_SOLD': volume_distribution,
            'DATE': list_of_dates,
            #'CLUSTER': [1,1,1,1,1,1,1,1],
            'CLUSTER': cluster_samples[cluster_number],
        }

        df_sample = pd.DataFrame(sample_data)

        # ================= Data Preprocessing Steps =====================
        df_sample['DATE'] = pd.to_datetime(df_sample['DATE'])
        df_sample['DATE_ORDINAL'] = df_sample['DATE'].apply(lambda x: x.toordinal())
        # Drop the original DATE column
        df_sample = df_sample.drop(columns=['DATE'])

        # Predict using the trained model
        # Reorder the columns of df_sample
        df_sample = df_sample[['DATE_ORDINAL', 'UNIT_SOLD', 'CLUSTER']]
        # print(df_sample)
        
        snowpark_df_sample = session.create_dataframe(df_sample)

        # ================= Predict the Average Discount Rate from UDFs snowflake===================
        predictions_set = snowpark_df_sample.select(
            F.col("DATE_ORDINAL"),
            F.col("UNIT_SOLD"),
            F.col("CLUSTER"),
            F.call_udf("udf_avg_discount_rate_prediction", 
                snowpark_df_sample['DATE_ORDINAL'],
                snowpark_df_sample['UNIT_SOLD'], 
                snowpark_df_sample['CLUSTER'], 
                "B33_S3_xgb_model.sav").alias('PREDICTED_AVG_DISCOUNT_RATE')
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