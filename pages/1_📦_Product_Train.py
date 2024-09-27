import streamlit as st
import os
import snowflake.connector as sf # type: ignore
import pandas as pd
import numpy as np

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

def convert_list_to_df(data, columns):
    return pd.DataFrame(data, columns=columns)
load_dotenv()

# =======Connect to Snowflake
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
raw_data = exec_query("""
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
""")

st.set_page_config(page_title="Product Training", page_icon="😊")
# Title for the app
st.title('Pricing Recommendation App')

products = [i[0] for i in raw_data]
#products = ['B15_S1', 'B13_S1', 'B31_S32', 'B44_S4', 'B15_S47']
# Product input - String list
product = st.selectbox(
    'Choose a Product',
    products  # Add or change the product names as needed
)

weekly_gross_sales_by_product = exec_query(f"""
    SELECT 
        DATE_TRUNC('week', FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) AS "WEEK",
        SUM(LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE * FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY) AS "GROSS_SALES",  
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
    WHERE
        LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product}')
        AND (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
    GROUP BY
        WEEK
    ORDER BY
        WEEK ASC
""")
columns = ["DATE", "GROSS_SALES"]
weekly_gross_sales_by_product = convert_list_to_df(weekly_gross_sales_by_product, columns)

# plot the area chart
st.area_chart(weekly_gross_sales_by_product, x="DATE", y="GROSS_SALES", color=["#00CCDD"])

# find min and max volumn of select product
if st.button("Train Model for this Product"):
    #data collection
    raw_data = exec_query(f"""
        SELECT 
            FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE AS "DATE",
            LU_PRD_PRODUCT_SKU.PRODUCT_SKU_RETAIL_PRICE AS "RETAIL_RPICE",
            REL_CUS_SEGMENTATION_RFM.CLUSTER AS "CLUSTER",
            COUNT(DISTINCT(FACT_ORDER_LINE_ITEM_BASE.ORDER_CODE)) AS "NUM_ORDERS",
            SUM(FACT_ORDER_LINE_ITEM_BASE.ORDERED_QUANTITY) AS "UNIT_SOLD",
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
            LU_PRD_PRODUCT_SKU.PRODUCT_SKU IN  ('{product}')
            AND (YEAR(FACT_ORDER_LINE_ITEM_BASE.ORDER_DATE) BETWEEN 2021 AND 2023)
        GROUP BY
            DATE, RETAIL_RPICE, CLUSTER
        ORDER BY
            DATE ASC
    """)

    columns = ["DATE", "RETAIL_RPICE", "CLUSTER", "NUM_ORDERS", "UNIT_SOLD", "AVG_DISCOUNT_RATE", "COGS"]
    df_raw_data = pd.DataFrame(raw_data, columns=columns)

    # data preprocessing
    df_raw_data['UNIT_SOLD'] = df_raw_data['UNIT_SOLD'].astype(int)
    df_raw_data['AVG_DISCOUNT_RATE'] = df_raw_data['AVG_DISCOUNT_RATE'].astype(float)
    df_raw_data['COGS'] = df_raw_data['COGS'].astype(float)

    df_raw_data['DATE'] = pd.to_datetime(df_raw_data['DATE'])

    df_raw_data['DATE_ORDINAL'] = df_raw_data['DATE'].apply(lambda x: x.toordinal())

    df_selected = df_raw_data.drop(columns=['RETAIL_RPICE', 'DATE'])

    # feature selection
    df_model = df_raw_data[['DATE_ORDINAL', 'UNIT_SOLD', 'CLUSTER', 'AVG_DISCOUNT_RATE']]

    # split the data
    train_set, test_set = train_test_split(df_model, test_size=0.2, random_state=42)

    
    X_train = train_set.drop(columns=['AVG_DISCOUNT_RATE'])
    y_train = train_set['AVG_DISCOUNT_RATE']
    X_test = test_set.drop(columns=['AVG_DISCOUNT_RATE'])
    y_test = test_set['AVG_DISCOUNT_RATE']

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Initialize the XGBRegressor
    xgb = XGBRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Train the model with the best parameters
    best_xgb = XGBRegressor(**best_params)
    best_xgb.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_xgb.predict(X_test)

    # Calculate the mean squared error
    mse = np.mean((y_pred - y_test) ** 2)
    path = os.path.join(os.getcwd(), f"Project_Development/models/{product}_xgb_model.joblib")
    dump(best_xgb, path)
    print("Mean Squared Error: ", mse)
    print("Training model for product", product)

    # push trained product to snowflake
    exec_query(f"""
        INSERT INTO LU_PRD_PRODUCT_TRAINED (PRODUCT_SKU, MIN_VOLUMN_SOLD, MAX_VOLUMN_SOLD, DATE_CREATED, DATE_MODIFIED)
        VALUES ('{product}', {df_model['UNIT_SOLD'].min()}, {df_model['UNIT_SOLD'].max()}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
    """)
    st.write(f"Training model for product: {product}")                          