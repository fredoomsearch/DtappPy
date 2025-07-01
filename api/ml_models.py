from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from Backend.api import events
from Backend.utils import database
from sklearn.linear_model import LinearRegression
import pandas as pd
from Backend.services import event_service, crypto_service
import logging

router = APIRouter()

CSV_FILE = "crypto_data_predictionURL.csv"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"

logger = logging.getLogger(__name__)

def fetch_and_store_data(db: Session = Depends(database.get_db)):
    try:
        crypto_service_instance = crypto_service.CryptoService(db)
        response = crypto_service_instance.fetch_crypto_data_from_api(COINGECKO_API_URL, db)
        data = response

        df = pd.DataFrame(data)
        df = df[['current_price', 'total_volume', 'symbol']]
        df.columns = ['feature1', 'target', 'symbol']
        df.to_csv(CSV_FILE, index=False)
        logger.info(f"Data successfully fetched and stored in {CSV_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error fetching and storing data: {e}")
        return False

def train_model(symbol, db: Session = Depends(database.get_db)):
    """
    Trains a linear regression model from a CSV file for a specific coin.
    Returns the trained model (in-memory).
    """
    try:
        if not fetch_and_store_data(db):
            raise HTTPException(status_code=500, detail="Error fetching and storing data")

        data = pd.read_csv(CSV_FILE)
        logger.info(f"Data successfully loaded from {CSV_FILE}")

        data = data[data['symbol'] == symbol]
        data = data.dropna()
        X = data[['feature1']]
        y = data['target']

        model = LinearRegression()
        model.fit(X, y)
        logger.info(f"Model successfully trained for {symbol}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

@router.get("/predict_price/")
def predict_price(db: Session = Depends(database.get_db)):
    """
    Endpoint to predict the price based on the trained model for each coin.
    """
    try:
        crypto_service_instance = crypto_service.CryptoService(db)
        response = crypto_service_instance.fetch_crypto_data_from_api(COINGECKO_API_URL, db)
        coins_data = response

        predictions = {}
        predictions_list = []
        for coin_data in coins_data:
            symbol = coin_data['symbol']
            current_price = coin_data['current_price']

            # Train model in-memory for this symbol
            model = train_model(symbol, db)

            input_data = pd.DataFrame([[current_price]], columns=['feature1'])
            prediction = model.predict(input_data)[0]

            predictions[symbol] = prediction
            predictions_list.append({'symbol': symbol, 'predicted_price': prediction})

            event_service.create_event(events.EventCreate(
                event_name="PRICE_PREDICTION",
                event_value=prediction,
                event_message=f"Price prediction made for {symbol}",
                event_details={"prediction": prediction, "symbol": symbol}
            ), db)

        # Optionally, save predictions to CSV
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df.to_csv("predictions.csv", index=False)

        return {"predictions": predictions, "message": "Predictions completed for all coins."}
    except HTTPException as e:
        raise e
    except Exception as e:
        event_service.create_event(events.EventCreate(
            event_name="PREDICTION_ERROR",
            event_value=1,
            event_message=f"Error during prediction: {e}",
            event_details={"error": str(e)}
        ), db)
        raise HTTPException(status_code=500, detail="Error during prediction")
    
@router.get("/csv/", summary="Download the prediction CSV")
def download_csv(db: Session = Depends(database.get_db)):
    try:
        from fastapi.responses import FileResponse
        # Serve the predictions CSV, not the raw data CSV
        return FileResponse("predictions.csv", media_type="text/csv", filename="predictions.csv")
    except Exception as e:
        logger.error("Could not serve CSV: %s", e)
        raise



