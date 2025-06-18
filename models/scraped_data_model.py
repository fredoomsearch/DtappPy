from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class ScrapedData(Base):
    __tablename__ = "scraped_data"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    url = Column(String, nullable=True)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)  # Use Text for potentially long descriptions
    content = Column(Text, nullable=True)      # Use Text for potentially long content
    publication_date = Column(DateTime(timezone=True), nullable=True)
    source = Column(String, nullable=True)