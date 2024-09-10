import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import joblib
import numpy as np

# Load the training data
file_path = 'path/to/your/train.csv'
train_data = pd.read_csv(file_path)

# Convert 'Order Date' to datetime format
train_data['Order Date'] = pd.to_datetime(train_data['Order Date'], format='%d/%m/%Y')

# Aggregate sales by month
monthly_sales = train_data.groupby(train_data['Order Date'].dt.to_period('M')).sum()['Sales']

# Convert PeriodIndex to DatetimeIndex for modeling
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot the aggregated monthly sales
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Train an ARIMA model for time series forecasting
model = ARIMA(monthly_sales, order=(5, 1, 0))
model_fit = model.fit()

# Save the model to a file
model_dump_path = 'path/to/save/sales_forecasting_model.pkl'
joblib.dump(model_fit, model_dump_path)

