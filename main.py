import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = "models/lstm_model.keras"
X_SCALER_PATH = "models/X_scaler.joblib"
Y_SCALER_PATH = "models/y_scaler.joblib"

# Global variables to store the loaded artifacts
model = None
X_scaler = None
y_scaler = None

# --- Pydantic Data Validation Model (Input Schema) ---
# This ensures inputs match the 15 features used in training, in the correct order .
class PredictionInput(BaseModel):
    Vehicle_Speed_kmh: float
    Vehicle_Weight_tons: float
    Truck_Count_hr: float
    Silt_Content_pc: float  # Using 'pc' instead of '%' for cleaner Python variable
    Road_Moisture_pc: float # Using 'pc' instead of '%' for cleaner Python variable
    Wind_Speed_mps: float
    Relative_Humidity_pc: float # Using 'pc' instead of '%' for cleaner Python variable
    Rainfall_mm_hr: float
    Surface_Roughness: float
    Temperature_C: float
    Watering_Interval_hr: float
    PM10_lag_1: float
    PM10_lag_2: float
    PM10_lag_3: float

app = FastAPI(
    title="PM10 Dust Prediction Service",
    description="LSTM model prediction for Rampura Agucha Mine dust levels."
)

# --- Startup Event: Load Artifacts into Memory ---
@app.on_event("startup")
async def load_artifacts():
    global model, X_scaler, y_scaler
    try:
        # Load the Keras model (architecture + weights)
        model = load_model(MODEL_PATH) [1]
        # Load the fitted scikit-learn scalers
        X_scaler = joblib.load(X_SCALER_PATH) [3]
        y_scaler = joblib.load(Y_SCALER_PATH) [3]
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # In a production scenario, you might want to stop the service if loading fails
        # raise Exception(f"Failed to load model dependencies: {e}")

# --- Prediction Endpoint ---
@app.post("/api/v1/predict/", summary="Predict PM10 concentration (ug/m3)")
async def predict_pm10(data: PredictionInput):
    # 1. Extract and order the 15 input features based on the Pydantic model
    # Convert Pydantic model fields to a list, maintaining order
    feature_list = list(data.model_dump().values())
    # Convert to NumPy array, required to be 2D for scikit-learn MinMaxScaler (1 sample, 15 features) [5]
    X_input_2D = np.array(feature_list).reshape(1, -1)

    # 2. Feature Scaling
    # Apply the exact scaling used during training
    X_scaled_2D = X_scaler.transform(X_input_2D)

    # 3. Reshape for LSTM
    # Reshape the 2D array to the mandatory 3D tensor: (Samples, Time Steps, Features)
    # Since we predict one time step, the shape is (1, 1, 15) [6, 7]
    X_scaled_3D = X_scaled_2D.reshape(1, 1, X_scaled_2D.shape[8])

    # 4. Inference
    # Model returns a scaled prediction
    scaled_prediction = model.predict(X_scaled_3D)

    # 5. Inverse Transformation (De-normalization)
    # Convert the scaled prediction back into physical units (ug/m3) [9, 10]
    pm10_ugm3_array = y_scaler.inverse_transform(scaled_prediction)

    # 6. Response Handling
    # Extract the scalar float value from the (1, 1) NumPy array for JSON serialization [11]
    pm10_prediction = float(pm10_ugm3_array)

    return {"PM10_ugm3": round(pm10_prediction, 2)}