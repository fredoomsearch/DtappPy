from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from Backend.api import events
from Backend.utils import database
from Backend.data_acquisition import web_scraper
from Backend.services import event_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

#FILES WHAT USE THIS END POINT CRYPTO_API , WEB_SCRAPER

def assign_topic(article):
    """Assign a topic to an article based on its content."""
    if not isinstance(article, dict):
        logger.error(f"Invalid article format: {article}")
        return "Other"  # Default topic for invalid data

    title = (article.get('title') or '').lower()
    description = (article.get('description') or '').lower()
    if any(word in title or word in description for word in ['stock', 'market', 'gdp', 'inflation']):
        return 'Economy'
    elif any(word in title or word in description for word in ['meta', 'microsoft', 'ai', 'tech', 'google', 'apple']):
        return 'Tech'
    elif any(word in title or word in description for word in ['trump', 'gop', 'republican', 'democrat', 'congress']):
        return 'Politics'
    else:
        return 'Other'

def process_news_data(news_data):
    """Process news data and add topics."""
    processed_data = []
    if isinstance(news_data, list):
        for article in news_data:
            if isinstance(article, dict):
                article_copy = article.copy()
                article_copy['topic'] = assign_topic(article)
                processed_data.append(article_copy)
    return processed_data

@router.get("/news/")
def get_news(db: Session = Depends(database.get_db)):
    """
    Fetches news data from NewsAPI and WSJ and returns it as JSON.
    """
    news_url = "https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=7711f5f714754e09a77c6d2640f1111e"
    news_url2 = "https://newsapi.org/v2/everything?domains=wsj.com&apiKey=7711f5f714754e09a77c6d2640f1111e"

    logger.info("Fetching news data...")
    news_data, scraped_data = web_scraper.scrape_website(news_url, news_url2, db)
   
    # Process the news data safely
    processed_news = process_news_data(news_data)
    processed_wsj = process_news_data(scraped_data)

    event_service.create_event(events.EventCreate(
        event_name="DATA_PROCESSED",
        event_value=0,
        event_message="Data processing complete",
        event_details={"news_data_len": len(processed_news), "scraped_data_len": len(processed_wsj)}
    ), db)

    combined_news = {"newsapi": processed_news, "wsj": processed_wsj}
    logger.info(f"Combined news: {combined_news}")
    return combined_news




