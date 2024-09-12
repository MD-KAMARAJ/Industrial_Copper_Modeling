import streamlit as st
from streamlit_option_menu import option_menu
import joblib
from joblib import load
import pandas as pd
import numpy as np
import sklearn

st.set_page_config(layout="wide")

st.title("Industrical copper modeling")
selected = option_menu( 
                        
                        menu_title= None,
                        options=["Home","Classification","Regression"],
                        orientation="horizontal"              
                        )
if selected == "Home":
    st.header("Industrial Copper Modelling - Manufacturing Domain")
    st.write('''The copper industry deals with less complex data related to sales and pricing. 
            However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions.
            Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions and so a machine
            learning regression model is developed to address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging
            algorithms that are robust to skewed and noisy data. After performing all these techniques a prediction model is developed to predict the selling price (Regression Model)  and status (Classification Model) by giving various inputs  
            The inputs are as follows:''')
    st.header('''1. Country''') 
    st.write(''' This information can be useful for understanding the geographic distribution of customers and may have implications for logistics and international sales.''') 
    st.header(''' 2. Item date''') 
    st.write('''This input represents the date when each transaction or item was recorded or occurred. It's important for tracking the timing of business activities.''')
    st.header(''' 3. Quantity tons''') 
    st.write('''This input indicates the quantity of the item in tons, which is essential for inventory management and understanding the volume of products sold or
                produced.''')
    st.header(''' 4. Item type''') 
    st.write('''This column categorizes the type or category of the items being sold or produced. Understanding item types is essential for inventory categorization and
                business reporting. After encoding this types are changed to numerics ranging from 0 to 6. Where,              
                0 represents IPL, 
                1 represents Others, 
                2 represents PL, 
                3 represents S, 
                4 represents SLAWR, 
                5 represents W, 
                6 represents WI.''')
    st.header('''5. Application''') 
    st.write('''The "application" defines the specific use or application of the items. This information can help tailor marketing and product development efforts.''')
    st.header('''6. Thickness''') 
    st.write('''The "thickness" donoted the details about the thickness of the items.''') 
    st.header(''' 7. Width''') 
    st.write('''The "width" specifies the width of the items.''') 
    st.header(''' 8. Delivery date''') 
    st.write('''This input represents the expected or actual delivery date for each item or transaction. It's crucial for managing logistics and ensuring timely delivery to
                customers.''')
    st.header(''' 9. Selling Price''') 
    st.write('''The "selling_price" column represents the price at which the items are sold. This is a critical factor for revenue generation and profitability analysis. This input is required only for predicting the STATUS (Classification model). ''')

if selected == "Classification":
    col1,col2 = st.columns(2)
    col1.markdown("## Inputs")
    df_features = pd.read_csv ("copper.csv")
    country = st.selectbox("country list",df_features['country'].unique())
    application = st.number_input("application")
    width = st.number_input("width")
    thickness = st.number_input("thickness")
    quantity = st.number_input("quantity ")
    item_type= st.selectbox("item type",[0,1,2,3,4,5,6])
    item_date = st.date_input("item date",value=None)
    delivery_date = st.date_input("Delivery date",value=None)
    selling_price = st.number_input("Selling Price")
        
    classification = { 
                        "country":  country,
                        "application": application,
                        "width": width,
                        "thickness": thickness,
                        "quantity tons": quantity,
                        "itemtype_encoded": item_type,
                        "item_date": item_date,
                        "delivery date": delivery_date,
                        "selling_price": selling_price                   
                        }
        
    df_classification = pd.DataFrame([classification]) 

    df_classification["item_date"] = pd.to_datetime(df_classification["item_date"],format='%Y%m%d')
    df_classification["delivery date"] = pd.to_datetime(df_classification["delivery date"],format='%Y%m%d')
    df_classification["quantity_kg"] = df_classification["quantity tons"]*1000
    df_classification["days_taken_for_delivery"] = (df_classification["delivery date"] - df_classification["item_date"]).dt.days
    df_classification['item_date_day'] = df_classification['item_date'].dt.day
    df_classification['item_date_month'] = df_classification['item_date'].dt.month
    df_classification['item_date_year'] = df_classification['item_date'].dt.year
    df_classification['delivery date_day'] = df_classification['delivery date'].dt.day
    df_classification['delivery date_month'] = df_classification['delivery date'].dt.month
    df_classification['delivery date_year'] = df_classification['delivery date'].dt.year
    df_classification["quantity_tons_kg_log"] = np.log(df_classification['quantity_kg'])
    df_classification['selling_price_log'] = np.log(df_classification['selling_price'])
    df_classification['thickness_log'] = np.log(df_classification['thickness'])
    df_classification['width_log'] = np.log(df_classification['width'])

    # droping item date and delivery date columns hence we converted to day,month,year columns separated 
    df1_class = df_classification.copy()
    
    df2_class = df1_class[["country","application","quantity_tons_kg_log","thickness_log","width_log","selling_price_log","days_taken_for_delivery","item_date_day","item_date_month","item_date_year","delivery date_day","delivery date_month","delivery date_year","itemtype_encoded"]]
    # Load the model and scaler
    model = joblib.load("model_class.joblib")
    scaler = joblib.load("standard_scaler_classification.joblib")

    # Check if there are any missing or incorrect values
    #if df3_class.isnull().values.any():
        #col2.container(border=True).markdown("***Please enter the input to predict the status***")
    if np.isinf(df2_class).values.any():
        col2.container(border=True).markdown("***The input contains infinite values. Please provide valid input values.***")    
    elif (df_classification["selling_price"] == 0).any():
        col2.container(border=True).markdown("***Please enter the input to predict the status***")
    else:
        # Scale the data
        scaled_data = scaler.transform(df2_class)
        
        # Predict using the model
        predictions = model.predict(scaled_data)
        
        # Display predictions
        col2.subheader("Predictions")
        col2.container(border=True).text(predictions[0])
        
if selected == "Regression":
    col1,col2 = st.columns(2)
    col1.markdown("## Inputs")
    df_features = pd.read_csv ("copper.csv")
    country = st.selectbox("country list",df_features['country'].unique())
    application = st.number_input("application")
    width = st.number_input("width")
    thickness = st.number_input("thickness")
    quantity = st.number_input("quantity tons")
    item_type= st.selectbox("item type",[0,1,2,3,4,5,6])
    item_date = st.date_input("item date",value=None)
    delivery_date = st.date_input("Delivery date",value=None)
        
    regression = { 
                        "country":  country,
                        "application": application,
                        "width": width,
                        "thickness": thickness,
                        "quantity tons": quantity,
                        "itemtype_encoded": item_type,
                        "item_date": item_date,
                        "delivery date": delivery_date}
        
    df_reg = pd.DataFrame([regression]) 

    df_reg["item_date"] = pd.to_datetime(df_reg["item_date"],format='%Y%m%d')
    df_reg["delivery date"] = pd.to_datetime(df_reg["delivery date"],format='%Y%m%d')
    df_reg["quantity_kg"] = df_reg["quantity tons"]*1000
    df_reg["days_taken_for_delivery"] = (df_reg["delivery date"] - df_reg["item_date"]).dt.days
    df_reg['item_date_day'] = df_reg['item_date'].dt.day
    df_reg['item_date_month'] = df_reg['item_date'].dt.month
    df_reg['item_date_year'] = df_reg['item_date'].dt.year
    df_reg['delivery date_day'] = df_reg['delivery date'].dt.day
    df_reg['delivery date_month'] = df_reg['delivery date'].dt.month
    df_reg['delivery date_year'] = df_reg['delivery date'].dt.year
    df_reg["quantity_tons_kg_log"] = np.log(df_reg['quantity_kg'])
    df_reg['thickness_log'] = np.log(df_reg['thickness'])
    df_reg['width_log'] = np.log(df_reg['width'])

    # droping item date and delivery date columns hence we converted to day,month,year columns separated 
    df1_reg = df_reg.copy()

    df2_reg = df1_reg[["country","application","quantity_tons_kg_log","thickness_log","width_log","days_taken_for_delivery","item_date_day","item_date_month","item_date_year","delivery date_day","delivery date_month","delivery date_year","itemtype_encoded"]]
    
    model = joblib.load("model_regression.joblib", mmap_mode='r')
    scaler_regression = joblib.load("standard_scaler_regression.joblib")

    col2.subheader("Predictions")
    
    if np.isinf(df2_reg).values.any():
        col2.container(border=True).markdown("***The input contains infinite values. Please provide valid input values.***")    
    else:        
        scaled_data = scaler_regression.transform(df2_reg)
        predictions = model.predict(scaled_data)
        predictions_exp_log = np.exp(predictions)
        col2.container(border=True).text(predictions_exp_log[0])