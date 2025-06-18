from pydantic import BaseModel
from typing import Optional

class CryptoCreate(BaseModel):
    coin_id: str
    symbol: str
    name: str
    image: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    fully_diluted_valuation: Optional[float] = None
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap_change_24h: float
    market_cap_change_percentage_24h: float
    circulating_supply: float
    total_supply: float
    max_supply: Optional[float] = None
    ath: float
    ath_change_percentage: float
    ath_date: str
    atl: float
    atl_change_percentage: float
    atl_date: str
    last_updated: str
    timestamp: int
   
class EventCreate(BaseModel):
    event_name: str
    event_value: float
    event_message: str
    event_details: dict
    
# filepath: /home/j/Documents/PYTHON/APPOFDATAANALYSYS/Backend/schemas.py

class NewsArticle(BaseModel):
    title: str | None = None
    description: str | None = None
    url: str | None = None    