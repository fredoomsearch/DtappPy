from sqlalchemy import Column, Integer, Float, String, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class CryptoData(Base):
    __tablename__ = "crypto_data"

    id = Column(Integer, primary_key=True, index=True)
    coin_id = Column(String, comment="Unique identifier for the coin")
    symbol = Column(String, comment="Symbol of the coin (e.g., BTC)")
    name = Column(String, comment="Name of the coin (e.g., Bitcoin)")
    image = Column(String, comment="URL of the coin's image")
    current_price = Column(Float, comment="Current price of the coin in USD")
    market_cap = Column(Float, comment="Market capitalization of the coin")
    market_cap_rank = Column(Integer, comment="Rank of the coin by market cap")
    fully_diluted_valuation = Column(Float, comment="Fully diluted valuation of the coin")
    total_volume = Column(Float, comment="Total trading volume in the last 24h")
    high_24h = Column(Float, comment="Highest price in the last 24h")
    low_24h = Column(Float, comment="Lowest price in the last 24h")
    price_change_24h = Column(Float, comment="Change in price in the last 24h")
    price_change_percentage_24h = Column(Float, comment="Percentage change in price in the last 24h")
    market_cap_change_24h = Column(Float, comment="Change in market cap in the last 24h")
    market_cap_change_percentage_24h = Column(Float, comment="Percentage change in market cap in the last 24h")
    circulating_supply = Column(Float, comment="Amount of coin circulating in the market")
    total_supply = Column(Float, comment="Total amount of coin in existence")
    max_supply = Column(Float, nullable=True, comment="Maximum amount of coin that will ever exist")
    ath = Column(Float, comment="All-time high price")
    ath_change_percentage = Column(Float, comment="Percentage change from all-time high")
    ath_date = Column(String, comment="Date of all-time high price")
    atl = Column(Float, comment="All-time low price")
    atl_change_percentage = Column(Float, comment="Percentage change from all-time low")
    atl_date = Column(String, comment="Date of all-time low price")
    last_updated = Column(String, comment="Date and time of last update")
    timestamp = Column(Integer, comment="Timestamp of the data") # Keep the timestamp

    def __repr__(self):
        return f"<CryptoData(name='{self.name}', price={self.current_price})>"