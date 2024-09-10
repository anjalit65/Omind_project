import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
# Load the model and preprocessor
rf_model_best = joblib.load('/Users/anjalitripathi/Downloads/MLOPs_Assignment_Anjali/House Price Prediction/random_forest_model_best.pkl')
preprocessor = joblib.load('/Users/anjalitripathi/Downloads/MLOPs_Assignment_Anjali/House Price Prediction/preprocessor.pkl')

test_df = pd.read_csv('/Users/anjalitripathi/Downloads/MLOPs_Assignment_Anjali/House Price Prediction/test.csv')

# Initialize FastAPI app
app = FastAPI()

class IDRequest(BaseModel):
    house_id: int

@app.post("/predict/")
def predict(request: IDRequest):
    # Fetch the row with the specified ID from test.csv
    row = test_df[test_df['Id'] == request.house_id]
    
    if row.empty:
        raise HTTPException(status_code=404, detail="ID not found in test.csv")
    
    # Add necessary temporal features
    row['HouseAge'] = row['YrSold'] - row['YearBuilt']
    row['YearsSinceRemodel'] = row['YrSold'] - row['YearRemodAdd']
    
    # Drop 'Id' and any other columns not needed for prediction
    X_test = row.copy()
    
    # Preprocess the data
    X_test_processed = preprocessor.transform(X_test)
    
    # Predict the sale price
    prediction = rf_model_best.predict(X_test_processed)
    predicted_price = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    
    # Return the prediction along with the 'Id'
    return {"Id": request.house_id, "predicted_price": predicted_price}
