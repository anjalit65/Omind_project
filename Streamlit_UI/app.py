import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
# Set up the page
st.title('House Price Predictor')

# Get user input
house_id = st.number_input('Enter House ID', min_value=1461,max_value=2919)

if st.button('Predict'):
    # Call the FastAPI endpoint
    response = requests.post("http://localhost:8000/house_price/predict/", json={"house_id": house_id})
    result = response.json()
    print(result)
    st.write(f"Predicted Sale Price: ${result['predicted_price']}")


response_for = requests.post("http://localhost:8000/sales_forecast/", params={"periods": 12})
result_for=response_for.json()
print(result_for)


# Example data for forecast plot
df = pd.DataFrame(result_for["forecast"])
df['index'] = pd.to_datetime(df['index'])

# Plotting
fig, ax = plt.subplots()
ax.plot(df['index'], df['Forecast'], marker='o')
ax.set_title('House Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

# Show plot in Streamlit
st.pyplot(fig)
