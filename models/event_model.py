from sqlalchemy import Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class EventData(Base):
    __tablename__ = "event_data"

    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String)
    event_value = Column(Float)
    event_message = Column(String)
    event_details = Column(JSON)