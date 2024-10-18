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


from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor # type: ignore
from joblib import dump

def exec_query(query):
    cs = ctx.cursor()
    try:
        cs.execute(query)
        return cs.fetchall()
    finally:
        cs.close()

load_dotenv()

# Connect to Snowflake
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
ROLE = os.getenv("SNOWFLAKE_ROLE")

ctx = sf.connect(
    user=USER,
    password=PASSWORD,
    account=ACCOUNT,
    warehouse=WAREHOUSE,
    database=DATABASE,
    schema=SCHEMA,
    role=ROLE,
    session_parameters={
        'QUERY_TAG': 'Panha Project Streamlit',
    }
)

st.set_page_config(page_title="Price Recommendation", page_icon="🏷️")


st.title('Pricing Recommendation App')


# read list of products from snowflake LU_PRD_PRODUCT_TRAINED
product_list = exec_query("SELECT *, LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE, LU_PRD_PRODUCT_SKU.LATEST_DATE FROM LU_PRD_PRODUCT_TRAINED JOIN LU_PRD_PRODUCT_SKU ON LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU")

product = [i[0] for i in product_list]
#product = ['B33_S3','B15_S1', 'B13_S1', 'B31_S32', 'B44_S4', 'B15_S47']


product = st.selectbox(
    'Choose a Product',
    product
)
retail_price = exec_query(f"SELECT LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE FROM LU_PRD_PRODUCT_TRAINED JOIN LU_PRD_PRODUCT_SKU ON LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU WHERE LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU = '{product}'")
print(retail_price[0][0])
st.caption(f"{product} base price: \\${retail_price[0][0]:.2f}")

# find min and max volumn of select product
min_max_volume_retail_price_latest_sales = exec_query(f"SELECT LU_PRD_PRODUCT_TRAINED.MIN_VOLUMN_SOLD, LU_PRD_PRODUCT_TRAINED.MAX_VOLUMN_SOLD, LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE, LU_PRD_PRODUCT_SKU.LATEST_DATE,   FROM LU_PRD_PRODUCT_TRAINED JOIN LU_PRD_PRODUCT_SKU ON LU_PRD_PRODUCT_SKU.PRODUCT_SKU = LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU WHERE LU_PRD_PRODUCT_TRAINED.PRODUCT_SKU = '{product}'")

start_date, end_date = st.date_input(
    "Select Date Range for Forecast",
    value=[datetime.today(), datetime.today() + timedelta(days=7)],  # Default range: today to one week from today
    min_value=datetime.today()  # Setting the minimum selectable date as today
)
num_days_selected = (end_date - start_date).days + 1
print(f"Number of days selected: {num_days_selected}")
min_volume = max(min_max_volume_retail_price_latest_sales[0][0], num_days_selected)  # Ensure minimum volume is at least number of days
max_volume = min_max_volume_retail_price_latest_sales[0][1]
retail_price = min_max_volume_retail_price_latest_sales[0][2]

volume = st.number_input(
    f"Enter the Expected Volume Sold (Recommended Range: {min_volume} - {max_volume})",  # Label for the input
    min_value=min_volume,  # Minimum value based on the product
    value=min_volume,  # Default value
    step=1 # Step for incrementing/decrementing
)

volume_per_day = volume // num_days_selected  # Base volume per day
remaining_volume = volume % num_days_selected  # Remaining volume to distribute

# Create a list to store the volume distribution
volume_distribution = [volume_per_day] * num_days_selected

# Add bias to the first `remaining_volume` days
for i in range(remaining_volume):
    volume_distribution[i] += 1  # Distribute the extra volume to the earlier days

list_of_dates = [start_date + timedelta(days=i) for i in range(num_days_selected)]
# print(f"Volume Distribution: {volume_distribution}")
# Button to Show Images and Prices
if st.button('Pricing Recommendations'):
    st.markdown('---')
    st.subheader('Product Pricing Recommended Range')
    cluster_members = [0,1,2]
    cluster_samples = [[clusters]*num_days_selected for clusters in cluster_members]
    print(cluster_samples)
    each_cluster_average_discount_rate = []
    for cluster_number in cluster_members:
        sample_data = {
            'UNIT_SOLD': volume_distribution,
            'DATE': list_of_dates,
            #'CLUSTER': [1,1,1,1,1,1,1,1],
            'CLUSTER': cluster_samples[cluster_number],
        }

        df_sample = pd.DataFrame(sample_data)
        df_sample['DATE'] = pd.to_datetime(df_sample['DATE'])
        df_sample['DATE_ORDINAL'] = df_sample['DATE'].apply(lambda x: x.toordinal())

        # Drop the original DATE column
        df_sample = df_sample.drop(columns=['DATE'])

        # Predict using the trained model
        # Reorder the columns of df_sample
        df_sample = df_sample[['DATE_ORDINAL', 'UNIT_SOLD', 'CLUSTER']]
        print(df_sample)
        # Display the reordered DataFrame
        loaded_model = load(f"Project_Development/models/{product}_xgb_model.joblib")

        # Predict using the loaded model
        sample_prediction_loaded = loaded_model.predict(df_sample)
        sample_prediction = loaded_model.predict(df_sample)
        average_discount_rate = sample_prediction.mean()
        each_cluster_average_discount_rate.append(average_discount_rate)
        print(f"Cluster {cluster_number} Prediction (Average Discount Rate): {average_discount_rate}")
        #print(sample_prediction_loaded)

    min_price_cs1 = retail_price * (1 - each_cluster_average_discount_rate[0])
    max_price_cs1 = retail_price * (1 + each_cluster_average_discount_rate[0])
    min_price_cs2 = retail_price * (1 - each_cluster_average_discount_rate[1])
    max_price_cs2 = retail_price * (1 + each_cluster_average_discount_rate[1])
    min_price_cs3 = retail_price * (1 - each_cluster_average_discount_rate[2])
    max_price_cs3 = retail_price * (1 + each_cluster_average_discount_rate[2])

    col1, col2, col3 = st.columns(3)

    
    with col1:
        st.image('CS 1.png', caption='Customer Segment 1') 
        st.write(f"Price Range: \\${min_price_cs1:.2f} - \\${max_price_cs1:.2f}")

    
    with col2:
        st.image('CS 2.png', caption='Customer Segment 2')
        st.write(f"Price Range: \\${min_price_cs2:.2f} - \\${max_price_cs2:.2f}")

    
    with col3:
        st.image('CS 3.png', caption='Product Image 3')
        st.write(f"Price Range: \\${min_price_cs3:.2f} - \\${max_price_cs3:.2f}")

    st.markdown('---')
    st.subheader('Additional Information Range')

    
    st.write(f"Selected Product: {product}")
    st.write(f"Product Base Price: \\${retail_price}")
    st.write(f"Expected Volume Sold: {volume} units")
    st.write(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")


