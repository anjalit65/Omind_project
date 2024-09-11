import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# Set up the page
st.title('House Price Predictor and Sales Forecaster')

# House Price Prediction
st.header('House Price Prediction')

house_id = st.number_input('Enter House ID', min_value=1461, max_value=2919, value=1461)

if st.button('Predict'):
    try:
        # Call the FastAPI endpoint
        response = requests.post("http://localhost:8000/house_price/predict/", json={"house_id": house_id})
        response.raise_for_status()  # Raise an error for bad HTTP responses
        
        result = response.json()
        st.success(f"Predicted Sale Price: ${result['predicted_price']:,}")
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")

# Sales Forecasting
st.header('Sales Forecasting')

# Forecast input
periods = st.number_input('Enter number of periods for forecast', min_value=1, max_value=24, value=12)

if st.button('Forecast'):
    try:
        # Call the FastAPI endpoint
        response_for = requests.post("http://localhost:8000/sales_forecast/", json={"periods": periods})
        response_for.raise_for_status()  # Raise an error for bad HTTP responses
        
        result_for = response_for.json()
        
        # Check if the forecast data is available
        if 'forecast' in result_for:
            df = pd.DataFrame(result_for['forecast'])
            
            # Ensure 'index' is a date for plotting
            df['index'] = pd.to_datetime(df['index'], errors='coerce')
            df = df.dropna(subset=['index'])  # Drop rows where conversion failed
            
            # Plotting
            fig, ax = plt.subplots()
            ax.plot(df['index'], df['Forecast'], marker='o')
            ax.set_title('Sales Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Forecasted Value')
            
            # Show plot in Streamlit
            st.pyplot(fig)
        else:
            st.warning("No forecast data available.")
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
