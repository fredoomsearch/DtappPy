from typing import List
from sqlalchemy.orm import Session
from repositories import crypto_repo
from schemas import CryptoCreate  # Import from schemas.py
from services import event_service
from api import events
from models.crypto_model import CryptoData
from models.event_model import EventData
from dateutil import parser  # Use dateutil.parser for parsing timestamps
import pandas as pd
import logging
from fastapi import HTTPException
import requests
from services.base_service import BaseService
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)

# Configure retry mechanism
retry_strategy = Retry(
    total=3,  # Maximum number of retries
    backoff_factor=1,  # Exponential backoff factor (1 means 1s, 2s, 4s...)
    status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

CSV_FILE = "crypto_data_predictionURL.csv"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"

class CryptoService:
    def __init__(self, db: Session):
        self.db = db
        self.crypto_repo = crypto_repo.CryptoRepository(db)

    def create(self, crypto: CryptoCreate) -> CryptoData:
        return self.crypto_repo.create(obj_in=crypto)

    def fetch_crypto_data_from_api(self, api_url: str, params: dict):
        try:
            response = http.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from {api_url}: {e}")
            event_service.create_event(events.EventCreate(
                event_name="API_ERROR",
                event_value=1,
                event_message=f"Error fetching data from {api_url}: {e}",
                event_details={"api_url": api_url, "error": str(e)}
            ), self.db)
            raise HTTPException(status_code=500, detail=f"API error: {e}")

    def get_crypto_data_from_csv(self):
        """Reads cryptocurrency data from the stored CSV file."""
        try:
            return pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            logging.error(f"CSV file not found: {CSV_FILE}")
            return None
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return None

    def fetch_crypto_data_for_analysis(self, api_url: str, db: Session):
        """Fetch crypto data from an external API and store it in the database."""
        try:
            response = http.get(api_url)  # Use the retrying session
            response.raise_for_status()
            data = response.json()

            crypto_entries = []  # List to hold CryptoData objects

            for item in data:
                timestamp_str = item.get("last_updated")
                try:
                    timestamp = int(parser.parse(timestamp_str).timestamp()) if timestamp_str else None
                    crypto_entry = CryptoData(
                        coin_id=item.get("id"),
                        symbol=item.get("symbol"),
                        name=item.get("name"),
                        image=item.get("image"),
                        current_price=float(item.get("current_price", 0)),
                        market_cap=float(item.get("market_cap", 0)),
                        market_cap_rank=int(item.get("market_cap_rank", 0)),
                        fully_diluted_valuation=float(item.get("fully_diluted_valuation", 0)) if item.get("fully_diluted_valuation") else None,
                        total_volume=float(item.get("total_volume", 0)),
                        high_24h=float(item.get("high_24h", 0)),
                        low_24h=float(item.get("low_24h", 0)),
                        price_change_24h=float(item.get("price_change_24h", 0)),
                        price_change_percentage_24h=float(item.get("price_change_percentage_24h", 0)),
                        market_cap_change_24h=float(item.get("market_cap_change_24h", 0)),
                        market_cap_change_percentage_24h=float(item.get("market_cap_change_percentage_24h", 0)),
                        circulating_supply=float(item.get("circulating_supply", 0)),
                        total_supply=float(item.get("total_supply", 0)),
                        max_supply=float(item.get("max_supply", 0)) if item.get("max_supply") else None,
                        ath=float(item.get("ath", 0)),
                        ath_change_percentage=float(item.get("ath_change_percentage", 0)),
                        ath_date=item.get("ath_date"),
                        atl=float(item.get("atl", 0)),
                        atl_change_percentage=float(item.get("atl_change_percentage", 0)),
                        atl_date=item.get("atl_date"),
                        last_updated=item.get("last_updated"),
                        timestamp=timestamp,
                    )

                    self.db.add(crypto_entry)
                    crypto_entries.append(crypto_entry)  # Append to the list

                except (ValueError, AttributeError) as e:
                    logging.error(f"Error parsing timestamp: {e}")
                    event_service.create_event(events.EventCreate(
                        event_name="ANALYSIS_ERROR",
                        event_value=1,
                        event_message=f"Error parsing timestamp: {e}",
                        event_details={"error": str(e)}
                    ), self.db)
                    continue  # Skip to the next item

            self.db.commit()  # Commit all changes after the loop
            for entry in crypto_entries:
                self.db.refresh(entry)  # Refresh each entry

            return crypto_entries  # Return the list of CryptoData objects

        except requests.exceptions.RequestException as e:
            event_service.create_event(events.EventCreate(
                event_name="API_ERROR",
                event_value=1,
                event_message=f"Error fetching data from {api_url}: {e}",
                event_details={"api_url": api_url, "error": str(e)}
            ), self.db)
            raise HTTPException(status_code=500, detail=f"API error: {e}")

    def analyze_crypto_data(self, data: dict, db: Session):
        event_service.create_event(events.EventCreate(
            event_name="ANALYSIS_STARTED",
            event_value=0,
            event_message="Crypto data analysis started",
            event_details={"data_length": len(data)}
        ), db)

        timestamp_str = data.get("last_updated")
        try:
            timestamp_datetime = parser.parse(timestamp_str)
            timestamp = int(timestamp_datetime.timestamp())

            # Extract all the fields from the data
            coin_id = data.get("id")
            symbol = data.get("symbol")
            name = data.get("name")
            image = data.get("image")
            current_price = float(data.get("current_price"))
            market_cap = float(data.get("market_cap"))
            market_cap_rank = int(data.get("market_cap_rank"))
            fully_diluted_valuation = float(data.get("fully_diluted_valuation")) if data.get("fully_diluted_valuation") else None
            total_volume = float(data.get("total_volume"))
            high_24h = float(data.get("high_24h"))
            low_24h = float(data.get("low_24h"))
            price_change_24h = float(data.get("price_change_24h"))
            price_change_percentage_24h = float(data.get("price_change_percentage_24h"))
            market_cap_change_24h = float(data.get("market_cap_change_24h"))
            market_cap_change_percentage_24h = float(data.get("market_cap_change_percentage_24h"))
            circulating_supply = float(data.get("circulating_supply"))
            total_supply = float(data.get("total_supply"))
            max_supply = float(data.get("max_supply")) if data.get("max_supply") else None
            ath = float(data.get("ath"))
            ath_change_percentage = float(data.get("ath_change_percentage"))
            ath_date = data.get("ath_date")
            atl = float(data.get("atl"))
            atl_change_percentage = float(data.get("atl_change_percentage"))
            atl_date = data.get("atl_date")
            last_updated = data.get("last_updated")

            crypto_entry = CryptoData(
                coin_id=coin_id,
                symbol=symbol,
                name=name,
                image=image,
                current_price=current_price,
                market_cap=market_cap,
                market_cap_rank=market_cap_rank,
                fully_diluted_valuation=fully_diluted_valuation,
                total_volume=total_volume,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change_24h=price_change_24h,
                price_change_percentage_24h=price_change_percentage_24h,
                market_cap_change_24h=market_cap_change_24h,
                market_cap_change_percentage_24h=market_cap_change_percentage_24h,
                circulating_supply=circulating_supply,
                total_supply=total_supply,
                max_supply=max_supply,
                ath=ath,
                ath_change_percentage=ath_change_percentage,
                ath_date=ath_date,
                atl=atl,
                atl_change_percentage=atl_change_percentage,
                atl_date=atl_date,
                last_updated=last_updated,
                timestamp=timestamp
            )

            self.db.add(crypto_entry)
            self.db.commit()
            self.db.refresh(crypto_entry)

        except (ValueError, AttributeError) as e:
            logging.error(f"Error parsing timestamp: {e}")
            event_service.create_event(events.EventCreate(
                event_name="ANALYSIS_ERROR",
                event_value=1,
                event_message=f"Error parsing timestamp: {e}",
                event_details={"error": str(e)}
            ), self.db)
            return data

        analyzed_data = data
        event_service.create_event(events.EventCreate(
            event_name="ANALYSIS_COMPLETED",
            event_value=0,
            event_message="Crypto data analysis completed",
            event_details={"analyzed_data_length": len(analyzed_data)}
        ), self.db)

        return analyzed_data

    def get_crypto_data(self, db: Session, skip: int = 0, limit: int = 100):
        return crypto_repo.get_crypto_data(db, skip, limit)

    def get_crypto_data_by_id(self, crypto_id: int):
        return self.crypto_repo.get_crypto_data_by_id(self.db, crypto_id)

    def create_crypto_data(self, db: Session, crypto: CryptoCreate):
        return crypto_repo.create_crypto_data(db, crypto)
