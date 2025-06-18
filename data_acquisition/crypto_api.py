import requests
from fastapi import HTTPException
from Backend.services import event_service
from Backend.api import events
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

def fetch_news_data(news_api_url: str, db: Session):
    try:
        response = requests.get(news_api_url)
        response.raise_for_status()
        data = response.json()  # Parse the JSON response
        print(f"NewsAPI Response: {data}")  # Debugging: Check the response
        articles = []
        for item in data["articles"]:
            articles.append({
                "title": item.get("title", "N/A"),
                "description": item.get("description", "N/A"),
                "url": item.get("url", "N/A")
            })
        return articles  # Extract the "articles" field
    except requests.exceptions.RequestException as e:
        event_service.create_event(events.EventCreate(
            event_name="API_ERROR",
            event_value=1,
            event_message=f"Error fetching news data from {news_api_url}: {e}",
            event_details={"api_url": news_api_url, "error": str(e)}
        ), db)
        raise HTTPException(status_code=500, detail=f"API error: {e}")