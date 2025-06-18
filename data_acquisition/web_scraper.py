import os
import requests
import logging
from bs4 import BeautifulSoup
from fastapi import HTTPException
from sqlalchemy.orm import Session
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from Backend.schemas import NewsArticle
from Backend.services import event_service
from Backend.api import events
from Backend.models import scraped_data_model
from Backend.data_acquisition import crypto_api  # Import crypto_api

logger = logging.getLogger(__name__)

def scrape_website(url1: str, url2: str, db: Session):
    """
    Scrapes data from two websites, stores it in the database, and returns the data.
    """
    try:
        # Fetch data from the first URL (NewsAPI) using crypto_api
        data1 = crypto_api.fetch_news_data(url1, db)

        # Scrape data from the second URL (WSJ)
        response2 = requests.get(url2)
        response2.raise_for_status()
        soup2 = BeautifulSoup(response2.content, 'html.parser')
        data2 = extract_data_wsj(soup2, url2)

        # Store the scraped data in the database
        store_data(data1, data2, db)

        return data1, data2

    except requests.exceptions.RequestException as e:
        event_service.create_event(events.EventCreate(
            event_name="SCRAPING_ERROR",
            event_value=1,
            event_message=f"Error scraping website: {e}",
            event_details={"url1": url1, "url2": url2, "error": str(e)}
        ), db)
        raise HTTPException(status_code=500, detail=f"Scraping error: {e}")

def extract_data_wsj(soup: BeautifulSoup, url: str):
    """
    Extracts relevant data from the BeautifulSoup object for WSJ.
    """
    try:
        title = soup.find('h1', class_='headline').text if soup.find('h1', class_='headline') else None
        content = soup.find('div', class_='article-content').text if soup.find('div', class_='article-content') else None
        logger.info(f"Extracted WSJ title: {title}")
        logger.info(f"Extracted WSJ content: {content}")
        return {"url": url, "title": title, "content": content}

    except Exception as e:
        logger.error(f"Error extracting data from {url}: {e}")
        return {}

def store_data(data1: list, data2: dict, db: Session):
    """
    Stores the scraped data in the database.
    """
    try:
        # Process NewsAPI data
        if data1:
            for item in data1:
                new_entry = scraped_data_model.ScrapedData(
                    url=item.get('url', 'N/A'),
                    title=item.get('title', 'N/A'),
                    description=item.get('description', 'N/A'),
                    content='N/A',
                    publication_date=None,
                    source='NewsAPI'
                )
                db.add(new_entry)

        # Process WSJ data
        if data2:
            new_entry = scraped_data_model.ScrapedData(
                url=data2.get('url', 'N/A'),
                title=data2.get('title', 'N/A'),
                description='N/A',
                content=data2.get('content', 'N/A'),
                publication_date=None,
                source='WSJ'
            )
            db.add(new_entry)

        db.commit()

    except Exception as e:
        logger.error(f"Error storing data: {e}")
        db.rollback()