""" #####################################################################################################################
REMENBER EACH MODEL OF IA TRAINED HAS 
Criteria	Approximate Real-World Prediction Accuracy
Predicting price at same moment (correlation)	✅ 75–85% R² accuracy likely if data is clean and features are informative.
Predicting short-term price movement (few hours ahead)	⚠️ 60–70% confidence at best.
Predicting next-day or future price	⚠️ <50–60% confidence; overfitting risk is high without time-series signals.
HAVE IN MIND THIS BEFORE OF TRAIN A MODEL FOR GIVE A MORE SOLID STRUCTURE """

import hashlib
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np  # IMPORTANT: Import numpy for log transformation
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from utils import database
from services import crypto_service
import logging
import os
import requests 
# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Basic logging configuration. For production, consider more advanced configurations.
router = APIRouter()
###REMENBER THE FIRST IMPORTATIONS ARE FOR HANDLE RANDOM FOREST MODEL AND THE OTHERS FUNCTIONS EXCLUDE ALL OF TF KERASMODEL
######################################################################################################################

######################################################################################################################
## DEEP TRAINIG MODEL START KERAS TF MODEL
# ## IMPORTS
import tensorflow as tf
# File paths

CSV_FILE_PATH = "crypto_data.csv"
KERAS_MODEL_FILE_PATH = "crypto_price_model.keras"
SCALER_FILE_PATH = "scaler.save"
DL_PREDICTIONS_CSV_FILE_PATH = "predicted_price_dl.csv"

def enrich_features(data):
    """Add technical indicators"""
    df = data.copy()

    # Convert price to float if needed
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')

    # Add technical indicators (example with price)
    df['sma_5'] = df['current_price'].rolling(window=5).mean()
    df['sma_10'] = df['current_price'].rolling(window=10).mean()
    df['ema_5'] = df['current_price'].ewm(span=5).mean()
    df['momentum'] = df['current_price'].diff()
    df['volatility'] = df['current_price'].rolling(window=5).std()
    
    # Normalize date
    df['days_since_ath'] = (pd.Timestamp.now() - pd.to_datetime(df['ath_date'], errors='coerce')).dt.days
    df['days_since_atl'] = (pd.Timestamp.now() - pd.to_datetime(df['atl_date'], errors='coerce')).dt.days

    df = df.dropna(subset=['current_price'])

    return df

def train_deep_learning_model(data: pd.DataFrame):
    try:
        data = enrich_features(data)

        drop_cols = ['current_price', 'symbol', 'name', 'id', 'coin_id', 'image', 'ath_date', 'atl_date', 'last_updated', 'timestamp']
        features = [col for col in data.columns if col not in drop_cols]
        numeric_features = data[features].select_dtypes(include=['number']).columns.tolist()

        X = data[numeric_features]
        y = data['current_price']

        valid_rows = X.notna().all(axis=1) & y.notna()
        X = X[valid_rows]
        y = y[valid_rows]

        if X.empty or y.empty:
            raise ValueError("Training data contains only NaNs or no valid rows remain after cleaning.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

        y_pred = model.predict(X_test_scaled).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test MAE: {mae}")

        model.save(KERAS_MODEL_FILE_PATH)
        joblib.dump(scaler, SCALER_FILE_PATH)

    except Exception as e:
        print(f"Error training model: {e}")
        raise

# --- Load Model ---
def load_deep_learning_model():
    try:
        if not os.path.exists(KERAS_MODEL_FILE_PATH) or not os.path.exists(SCALER_FILE_PATH):
            raise FileNotFoundError("Model or scaler file not found.")
        model = tf.keras.models.load_model(KERAS_MODEL_FILE_PATH)
        scaler = joblib.load(SCALER_FILE_PATH)
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None, None

# --- Predict Endpoint ---
@router.get("/crypto/predict_dl/")
async def predict_crypto_prices_dl():
    try:
        if not os.path.exists(CSV_FILE_PATH):
            raise HTTPException(status_code=400, detail="CSV not found. Upload it via /crypto/csv/")

        data = pd.read_csv(CSV_FILE_PATH)

        drop_cols = ['current_price', 'symbol', 'name', 'id', 'coin_id', 'image', 'ath_date', 'atl_date', 'last_updated', 'timestamp']
        features = [col for col in data.columns if col not in drop_cols]
        numeric_features = data[features].select_dtypes(include=['number']).columns.tolist()

        X = data[numeric_features]
        y = data['current_price']

        valid_rows = X.notna().all(axis=1) & y.notna()
        X = X[valid_rows]
        y = y[valid_rows]

        if X.empty:
            raise ValueError("No valid rows after dropping NaNs.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

        model.save(KERAS_MODEL_FILE_PATH)
        joblib.dump(scaler, SCALER_FILE_PATH)

        # Predict for full dataset
        full_input = data[numeric_features].fillna(0)
        full_input = full_input.reindex(columns=scaler.feature_names_in_, fill_value=0)
        input_scaled = scaler.transform(full_input)

        predictions = model.predict(input_scaled).flatten()

        result_df = pd.DataFrame({
            'symbol': data['symbol'],
            'name': data['name'],
            'predicted_price': predictions
        })

        result_df.to_csv(DL_PREDICTIONS_CSV_FILE_PATH, index=False)

        return {
            "message": f"DL Predictions saved to {DL_PREDICTIONS_CSV_FILE_PATH}",
            "sample": result_df.head(5).to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"Error during DL prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crypto/predictions_dl/")
async def get_dl_predictions():
    try:
        if not os.path.exists(DL_PREDICTIONS_CSV_FILE_PATH):
            raise HTTPException(status_code=404, detail="DL Predictions not found")

        df = pd.read_csv(DL_PREDICTIONS_CSV_FILE_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error loading DL predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read DL predictions")


######################################################################################################################

######################################################################################################################
##RADOM FOREST MODEL
""" Criteria	Approximate Real-World Prediction Accuracy
Predicting price at same moment (correlation)	✅ 75–85% R² accuracy likely if data is clean and features are informative.
Predicting short-term price movement (few hours ahead)	⚠️ 60–70% confidence at best.
Predicting next-day or future price	⚠️ <50–60% confidence; overfitting risk is high without time-series signals. """
###80%%% OF PROBABILITY
## FOR USE, REMEMBER FIRST RUN THE CSV ENDPOINT, AND AFTER THAT THE PREDICTION ENDPOINT.
### TRAINED WITH RANDOM FOREST MODEL
# CoinGecko API URL
# DONT ERASE THE URL REMEMBER WHAT IS USED FOR THE CSV METHOD, FOR EXTRACT AND CREATE A CSV FROM THE DATA OF COIN GECKO
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
CSV_FILE_PATH = "crypto_data.csv"
PREDICTIONS_CSV_FILE_PATH = "crypto_predictions.csv"
MODEL_FILE_PATH = "crypto_price_model.joblib"

def train_random_forest_model(csv_file_path: str) -> bool:
    """
    Trains a Random Forest Regressor model from a CSV file, excluding specified columns.
    Applies log transformation to the target variable for better prediction accuracy.
    """
    try:
        data = pd.read_csv(csv_file_path)
        logger.info(f"Data successfully loaded from {csv_file_path}")

        # Data preprocessing: Selecting features and target
        data = data.dropna() # Drop rows with any NaN values

        # Define the target variable
        y = data['current_price']

        # --- KEY CHANGE 1: Apply log transformation to the target variable ---
        # np.log1p(x) calculates log(1+x), which is safer for values close to zero.
        y_log = np.log1p(y)

        # Select all columns except the ones to be dropped.
        # The features you listed earlier (price_change_24h, ath_change_percentage, etc.)
        # are implicitly included here as they are not in the drop list.
        columns_to_drop = [
            'image', 'symbol', 'name', 'id', 'coin_id', # 'id' is likely from DB, 'coin_id' might be redundant
            'last_updated', 'ath_date', 'atl_date', 'current_price'
        ]
        # Use errors='ignore' in case some columns don't exist (e.g., 'id' if not from DB)
        X = data.drop(columns=columns_to_drop, errors='ignore')

        # Data scaling for features (X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- KEY CHANGE 2: Split data using y_log for training the model ---
        X_train, X_test, y_log_train, y_log_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_log_train) # <-- Train on log-transformed y

        # Evaluate the model
        y_log_pred = model.predict(X_test)
        # --- KEY CHANGE 3: Inverse transform predictions and actual test values for evaluation metrics ---
        y_pred_original_scale = np.expm1(y_log_pred)
        y_test_original_scale = np.expm1(y_log_test) # Inverse transform the actual test values too for fair comparison

        mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
        r2 = r2_score(y_test_original_scale, y_pred_original_scale)
        logger.info(f"Model evaluation (original scale) - MSE: {mse:.4f}, R2: {r2:.4f}") # Format for readability

        # --- KEY CHANGE 4: Save a flag indicating log transformation was used ---
        # This flag is crucial for knowing whether to inverse transform during prediction
        joblib.dump((model, scaler, list(X.columns), True), MODEL_FILE_PATH) # Added 'True' flag
        logger.info(f"Model and scaler successfully trained and saved to {MODEL_FILE_PATH}")
        return True

    except FileNotFoundError:
        logger.error(f"File not found: {csv_file_path}")
        return False
    except pd.errors.EmptyDataError:
        logger.error(f"The CSV file is empty: {csv_file_path}")
        return False
    except KeyError as e:
        logger.error(f"Column not found in CSV: {e}")
        return False
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True) # exc_info=True for full traceback
        return False

def load_model():
    """
    Loads the trained Random Forest model, scaler, feature names, and log_transformed flag from a file.
    """
    try:
        # --- KEY CHANGE 5: Load all saved components, including the new log_transformed flag ---
        # We assume the model file will contain 4 elements now.
        model, scaler, feature_names, log_transformed = joblib.load(MODEL_FILE_PATH)
        logger.info(f"Model and scaler successfully loaded from {MODEL_FILE_PATH}")
        return model, scaler, feature_names, log_transformed # Return the new flag
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_FILE_PATH}. Training a new model will be attempted.")
        # Return default values, including False for log_transformed if model not found
        return None, None, None, False
    except Exception as e:
        logger.error(f"Error loading model from {MODEL_FILE_PATH}: {e}", exc_info=True)
        return None, None, None, False

@router.get("/crypto/process_and_predict/")
async def process_and_predict(db: Session = Depends(database.get_db)): # Keep DB dependency if you need it for other parts of the flow
    """
    Fetches crypto data from the CSV file (assumed to be already created),
    trains/loads a model, makes predictions, and generates a CSV file.
    """
    try:
        # Read data from the CSV file
        try:
            data = pd.read_csv(CSV_FILE_PATH)
            logger.info(f"Data successfully loaded from {CSV_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error reading data from CSV: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading data from CSV: {e}")

        # Data preprocessing: Selecting features and target
        data = data.dropna()

        # Train the model. This will now include log transformation on the target.
        # In a real-world application, you typically wouldn't train the model on every request.
        # Consider a scheduled task (e.g., cron job) to run train_random_forest_model periodically.
        if not train_random_forest_model(CSV_FILE_PATH):
            raise HTTPException(status_code=500, detail="Failed to train the Random Forest model.")

        # Load the trained model and scaler, including the log_transformed flag
        model, scaler, feature_names, log_transformed = load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Failed to load the trained model.")

        # Make predictions and store them
        predictions = []
        for index, row in data.iterrows():
            try:
                # Prepare the input data for prediction.
                # Ensure all features used during training are present, even if their values are 0 for this row.
                columns_to_drop_for_prediction = [
                    'image', 'symbol', 'name', 'id', 'coin_id',
                    'last_updated', 'ath_date', 'atl_date', 'current_price'
                ]
                row_data_for_prediction = row.drop(columns=columns_to_drop_for_prediction, errors='ignore')
                input_data = pd.DataFrame([row_data_for_prediction.to_dict()])

                # Reindex to ensure feature order and presence match training data
                # This is critical for StandardScaler to work correctly.
                input_data = input_data.reindex(columns=feature_names, fill_value=0)

                # Scale the input data using the loaded scaler
                input_data_scaled = scaler.transform(input_data)

                # Predict the log-transformed price
                predicted_price_log = model.predict(input_data_scaled)[0]

                # --- KEY CHANGE 6: Inverse transform the predicted price only if log transformation was applied ---
                if log_transformed:
                    predicted_price = np.expm1(predicted_price_log)
                else:
                    predicted_price = predicted_price_log # Fallback (should be True if trained with new logic)

                predictions.append({
                    'symbol': row['symbol'],
                    'predicted_price': round(float(predicted_price), 8) # Round for cleaner output
                })
            except Exception as e:
                logger.error(f"Error making prediction for row {index} (Symbol: {row.get('symbol', 'N/A')}): {e}", exc_info=True)
                # Optionally, you might want to append a prediction of None or a placeholder for failed rows
                predictions.append({
                    'symbol': row.get('symbol', 'ERROR_SYMBOL'),
                    'predicted_price': None # Or float('nan')
                })

        # Create a DataFrame from the predictions
        predictions_df = pd.DataFrame(predictions)
        # Filter out rows where prediction failed (predicted_price is None) if you added None
        predictions_df = predictions_df.dropna(subset=['predicted_price'])

        # Save predictions to a CSV file
        try:
            predictions_df.to_csv(PREDICTIONS_CSV_FILE_PATH, index=False)
            logger.info(f"Predictions saved to {PREDICTIONS_CSV_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error saving predictions to CSV: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error saving predictions to CSV: {e}")

        return {"message": f"Data processed, model trained, predictions made, and saved to {PREDICTIONS_CSV_FILE_PATH}"}

    except HTTPException as http_ex:
        raise http_ex  # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Unexpected error in process_and_predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@router.get("/crypto/csv/")
def csv_endpoint(db: Session = Depends(database.get_db)):
    """
    Endpoint to get crypto data in CSV format from CoinGecko API and save it.
    """
    try:
        crypto_service_instance = crypto_service.CryptoService(db)

        # Fetch data from the API and store it in the database
        try:
            crypto_data = crypto_service_instance.fetch_crypto_data_for_analysis(COINGECKO_API_URL, db)
            if not crypto_data:
                logger.error("Failed to fetch and store data from CoinGecko API.")
                raise HTTPException(status_code=500, detail="Failed to fetch and store data from the API.")
        except requests.exceptions.RequestException as req_e:
            logger.error(f"Network error fetching crypto data from CoinGecko: {req_e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Network error fetching crypto data: {req_e}")
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error fetching crypto data: {e}")

        # Convert data to Pandas DataFrame
        # Ensure 'id' is extracted from the model instance
        df_data = []
        for item in crypto_data:
            item_dict = item.__dict__.copy()
            item_dict['id'] = item.id # Explicitly get the 'id' if it's not directly in __dict__ or for clarity
            df_data.append(item_dict)

        df = pd.DataFrame(df_data)

        # Remove the SQLAlchemy internal attribute '_sa_instance_state' and 'id' if already handled
        df = df.drop(columns=['_sa_instance_state'], errors='ignore')
        # If 'id' is already in your features and you don't want it, remove it here too
        # If your 'id' column is useful for some purpose (e.g., join), don't drop it.
        # The 'id' in crypto_model is probably the primary key, distinct from 'coin_id' from CoinGecko
        # For model training, 'id' (DB primary key) is usually not a feature.

        try:
            df.to_csv(CSV_FILE_PATH, index=False)
            logging.info(f"CSV file successfully created at {CSV_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error saving CSV file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error saving CSV file: {e}")

        return {"message": f"CSV file successfully created at {CSV_FILE_PATH}"}

    except HTTPException as http_ex:
        raise http_ex  # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Unexpected error in csv_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
# --- NEW ENDPOINT: get_crypto_predictions_data (for model predictions) ---
###REMENBER THIS ENDPOINT IS FOR GET THE DATA OF THE PREDICTION OF THE RANDOM FOREST MODEL
##MODEL WHIT A 80%% OF PROBABILITY IN PRICE PREDICTIONS
@router.get("/crypto/predictions-data")
def get_crypto_predictions_data():
    """
    Reads crypto PREDICTIONS data from the generated CSV file (crypto_predictions.csv) and returns it as JSON.
    This is for displaying model predictions in the frontend.
    """
    try:
        file_path = PREDICTIONS_CSV_FILE_PATH # Uses crypto_predictions.csv
        if not os.path.exists(file_path):
            logger.error(f"Crypto predictions CSV file not found at {file_path}")
            raise HTTPException(status_code=404, detail="Crypto predictions data not available. Please run the prediction endpoint first.")

        df = pd.read_csv(file_path)
        df = df.fillna(0) # Replace NaN values (e.g., if a prediction failed or was NaN)

        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading or processing crypto predictions data for frontend: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing crypto predictions data: {e}")
    
    

## REMEMBER WHAT THIS API IS FOR SHOW AND GRAPHIC FROM TEH CSV CRYPTO_DATA, IN THE FRONT END.
# FOR BUTTON Crypto.
@router.get("/crypto/crypto-data")
def get_crypto_data():
    """
    Reads crypto data from the generated CSV file and returns it as JSON.
    """
    try:
        file_path = CSV_FILE_PATH
        if not os.path.exists(file_path):
            logger.error(f"Crypto data CSV file not found at {file_path}")
            raise HTTPException(status_code=404, detail="Crypto data not available. Please run the CSV endpoint first.")

        df = pd.read_csv(file_path)

        # Replace NaN values with a default value (e.g., 0 or "N/A") for frontend display
        df = df.fillna(0) # Or replace with sensible defaults based on column type

        # Convert to JSON records
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading or processing crypto data for frontend: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing crypto data: {e}")
######################################################################################################################



###PAYU!!!
# Removed hashlib and Message, SMTP imports as they were conflicting with other libraries.
# Replaced with standard Python email.mime and smtplib for clarity and consistency.
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib # Re-import smtplib directly

# Function to calculate PayU signature for IPN (CRITICAL!)
# This function MUST match PayU's signature logic.
# The common string for IPN notification signature is:
# MD5(apiKey~merchant_id~referenceCode~value~currency~transactionState)
# PayU often requires the value to be formatted with two decimal places.
def calculate_payu_signature_ipn(api_key: str, merchant_id: str, reference_code: str, amount: str, currency: str, transaction_state: str) -> str:
    """
    Calculates the MD5 hash signature for PayU IPN notification.
    Parameters:
        api_key (str): Your PayU API Key.
        merchant_id (str): Your PayU merchant ID.
        reference_code (str): Transaction reference code.
        amount (str): Total transaction value. Must be a string (e.g., "5000.00").
                      Ensure it has two decimal places if PayU requires it.
        currency (str): Transaction currency (e.g., "COP", "USD").
        transaction_state (str): Transaction state (e.g., "APPROVED", "DECLINED").
    """
    # Ensure amount is formatted as a string with exactly two decimal places for PayU signature
    try:
        amount_float = float(amount)
        formatted_amount = f"{amount_float:.2f}"
    except ValueError:
        logger.error(f"Invalid amount format received: {amount}. Using original string for signature calculation.")
        formatted_amount = amount # Fallback if conversion fails, but this might lead to signature mismatch

    # Important: The order and names of parameters in the string_to_hash MUST match PayU's documentation precisely.
    # Common order: MD5(apiKey~merchant_id~referenceCode~value~currency~transactionState)
    string_to_hash = f"{api_key}~{merchant_id}~{reference_code}~{formatted_amount}~{currency}~{transaction_state}"
    logger.info(f"String to hash for IPN: {string_to_hash}")
    hashed_signature = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()
    return hashed_signature

async def send_prediction_email(email: str, csv_path: str):
    """
    Sends an email with the cryptocurrency predictions CSV attached.
    """
    # Adjust this path if necessary. It assumes 'templates' is two levels up from api/crypto.py
    # Example: If crypto.py is in Backend/api/, templates should be in Backend/templates/
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../templates')
    templates = Jinja2Templates(directory=templates_dir)

    try:
        # Load environment variables for SMTP credentials
        # IMPORTANT: These must be set securely in your environment.
        # You will need to set these environment variables before running your FastAPI app:
        # export SMTP_HOST="your.smtp.host"
        # export SMTP_PORT="587"
        # export SMTP_USERNAME="your_email@example.com"
        # export SMTP_PASSWORD="your_email_password_or_app_specific_password"
        # export SENDER_EMAIL="your_email@example.com"
        SMTP_HOST = os.getenv("SMTP_HOST")
        SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
        SMTP_USERNAME = os.getenv("SMTP_USERNAME")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
        SENDER_EMAIL = os.getenv("SENDER_EMAIL")
        SENDER_NAME = os.getenv("SENDER_NAME", "Crypto Predictor")

        if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
            logger.error("SMTP credentials are not fully configured in environment variables. Email not sent.")
            return

        # Create message container
        msg = MIMEMultipart()
        msg['Subject'] = "Your Cryptocurrency Price Prediction!"
        msg['From'] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg['To'] = email

        # Create the HTML content from a template
        # Make sure you have a file named 'prediction_email.html' in your templates directory
        try:
            template_context = {"receiver_email": email} # You can pass more context if needed
            email_body_html = templates.get_template("prediction_email.html").render(template_context)
            msg.attach(MIMEText(email_body_html, 'html'))
        except Exception as e:
            logger.error(f"Error rendering email template 'prediction_email.html': {e}")
            msg.attach(MIMEText("Please find your cryptocurrency price prediction attached. An error occurred while generating the HTML content.", 'plain'))

        # Attach the CSV file
        try:
            with open(csv_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="csv")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(csv_path))
                msg.attach(attach)
        except FileNotFoundError:
            logger.error(f"Prediction CSV file not found at {csv_path}. Email sent without attachment.")
            msg.attach(MIMEText("\n\nNote: The prediction file could not be attached as it was not found.", 'plain'))
        except Exception as e:
            logger.error(f"Error attaching CSV file {csv_path}: {e}")
            msg.attach(MIMEText("\n\nNote: An error occurred while attaching the prediction file.", 'plain'))

        # Send the email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls() # Enable TLS encryption
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
        logger.info(f"Prediction email sent successfully to {email}")

    except Exception as e:
        logger.error(f"Failed to send prediction email to {email}: {e}", exc_info=True)        # In a real application, you might want to log this error and potentially queue a retry.

@router.post("/payu/notification")
async def payu_notification(request: Request, db: Session = Depends(database.get_db)):
    """
    Handles Instant Payment Notifications (IPN) from PayU.
    Validates the signature and processes approved transactions.
    """
    form_data = await request.form()
    logger.info(f"Notificación IPN de PayU recibida. Form data: {form_data}")

    # Extract required data from PayU IPN notification.
    # Field names MUST match exactly what PayU sends (case-sensitive).
    payu_signature = form_data.get("sign")
    merchant_id_payu = form_data.get("merchant_id")
    reference_code = form_data.get("referenceCode")
    amount = form_data.get("TX_VALUE")
    currency = form_data.get("currency")
    transaction_state = form_data.get("transactionState")

    # IMPORTANT: PayU does NOT send the buyer's email in the IPN.
    # You MUST retrieve the buyer's email from your own database using the `referenceCode`
    # or `orderId` that you passed to PayU when the transaction was initiated.
    # For demonstration, a placeholder is used, but this needs proper implementation.
    # Example:
    # email_comprador = your_db_lookup_function(reference_code)
    # If not found or if the state is not APPROVED, handle appropriately.
    # For now, let's assume you fetch it or have it from a previous step.
    # Replace this with your actual lookup:
    email_comprador = "test_user@example.com" # <-- REPLACE WITH ACTUAL EMAIL LOOKUP BASED ON reference_code
    if not email_comprador:
        logger.error(f"Email for referenceCode {reference_code} not found in your system. Cannot send predictions.")
        return Response(status_code=400, content="Buyer email not found.")

    logger.info(f"Data received for signature verification: merchantId={merchant_id_payu}, referenceCode={reference_code}, amount={amount}, currency={currency}, transactionState={transaction_state}, PayU_Sign={payu_signature}")

    # Your PayU API Key. Load this from environment variables in production.
    # DO NOT hardcode sensitive keys.
    api_key = os.getenv("PAYU_API_KEY", "ehCsEmI12qCYcU6h3RJc9q6tBR") # Default for testing

    # Calculate the signature using the received data and your API Key
    calculated_signature = calculate_payu_signature_ipn(
        api_key=api_key,
        merchant_id=merchant_id_payu,
        reference_code=reference_code,
        amount=amount,
        currency=currency,
        transaction_state=transaction_state
    )

    # Compare the calculated signature with the signature sent by PayU
    if calculated_signature.lower() != payu_signature.lower(): # Compare in lower case to avoid case sensitivity issues
        logger.error(f"Invalid PayU signature. Calculated: {calculated_signature}, Received: {payu_signature}")
        # It's crucial to return a non-200 status code to PayU to indicate validation failure.
        return Response(status_code=400, content="Invalid Signature")

    # Process the transaction if the signature is valid
    if transaction_state == "APPROVED":
        logger.info(f"Payment APPROVED for reference {reference_code}, buyer: {email_comprador}")
        try:
            # 1. Update the crypto_data.csv (fetch fresh data from CoinGecko)
            # This ensures the prediction model uses the most recent data.
            # CONSIDERATION: Calling this on every IPN might be too frequent.
            # For high-traffic apps, schedule this update periodically (e.g., hourly via cron).
            await csv_endpoint(db) # Call the endpoint to update the data CSV.

            # 2. Generate the prediction CSV using the (potentially newly trained) model
            await process_and_predict(db)
            csv_file_path_for_email = PREDICTIONS_CSV_FILE_PATH # Use the defined path

            # 3. Send the prediction email to the buyer
            await send_prediction_email(email_comprador, csv_file_path_for_email)

        except HTTPException as http_ex:
            logger.error(f"Error processing IPN or sending email for {reference_code}: {http_ex.detail}", exc_info=True)
            return Response(status_code=500, content=f"Backend processing error: {http_ex.detail}")
        except Exception as e:
            logger.error(f"Unexpected error during IPN processing for {reference_code}: {e}", exc_info=True)
            return Response(status_code=500, content=f"Unexpected backend error: {e}")

    elif transaction_state == "DECLINED":
        logger.warning(f"Payment DECLINED for reference {reference_code}. No predictions sent.")
        # Optionally: send a notification email for declined payment.
    elif transaction_state == "PENDING":
        logger.info(f"Payment PENDING for reference {reference_code}. No predictions sent yet.")
        # Optionally: update transaction status in your DB, send "payment in process" email.
    else:
        logger.info(f"Unknown transaction state: {transaction_state} for reference {reference_code}.")

    # Always respond with 200 OK to PayU if the notification was successfully received and processed
    # (even if the payment was declined or pending). PayU expects this 200 OK.
    return Response(status_code=200, content="Notification received and processed successfully")
