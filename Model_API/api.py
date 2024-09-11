import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model and preprocessor
rf_model_best = joblib.load('../House Price Prediction/random_forest_model_best.pkl')
preprocessor = joblib.load('../House Price Prediction/preprocessor.pkl')

test_df = pd.read_csv('../House Price Prediction/test.csv')
forecast_model_path = '../Sales Forecasting/sales_forecasting_model.pkl'

# Initialize FastAPI app
app = FastAPI()

class IDRequest(BaseModel):
    house_id: int

@app.post("/house_price/predict/")

def predict(request: IDRequest):

    # Fetch the row with the specified ID from test.csv
    row = test_df[test_df['Id'] == request.house_id]
    
    if row.empty:
        raise HTTPException(status_code=404, detail="ID not found")
    
    # Add necessary temporal features
    row['HouseAge'] = row['YrSold'] - row['YearBuilt']
    row['YearsSinceRemodel'] = row['YrSold'] - row['YearRemodAdd']
    

    X_test = row.copy()

    # Preprocess the data
    try:
        X_test_processed = preprocessor.transform(X_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")
    
    # Predict the sale price
    try:
        prediction = rf_model_best.predict(X_test_processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    predicted_price = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    
    # Return the prediction along with the 'Id'
    return {"Id": request.house_id, "predicted_price": round(float(predicted_price), 2)}

class ForecastRequest(BaseModel):
    periods: int

@app.post("/sales_forecast/")
def forecast(request: ForecastRequest):
    try:
        model_fit = joblib.load(forecast_model_path)
        # Generate forecast
        forecast = model_fit.forecast(steps=request.periods)
        
        # Round forecast values to 2 decimal places
        forecast_df = pd.DataFrame({'Forecast': forecast}).round(2)

        # Convert the DataFrame to a dictionary
        forecast_dict = forecast_df.reset_index().to_dict(orient='records')
        return {"forecast": forecast_dict}
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Forecast model file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
