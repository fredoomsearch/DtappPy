import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch

from main import app
from utils.database import Base, get_db
from models import crypto_model, event_model
from services import crypto_service

# Define the test database URL (use an in-memory SQLite database for testing)
SQLALCHEMY_DATABASE_URL = "postgresql://myuser:mypassword@localhost/mydb"

# Create a test database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a test session maker
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pytest fixture to create a test database session
@pytest.fixture()
def session():
    event_model.Base.metadata.create_all(bind=engine)
    crypto_model.Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pytest fixture to create a test client
@pytest.fixture()
def client(session):
    def override_get_db():
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    del app.dependency_overrides[get_db]

def test_xml_endpoint(client, session):
    # Mock the crypto_service.create_xml_from_db function
    with patch("Backend.services.crypto_service.CryptoService.create_xml_from_db") as mock_create_xml_from_db:
        mock_create_xml_from_db.return_value = "<xml>Test XML Data</xml>"
        response = client.get("/crypto/xml")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert response.text == "<xml>Test XML Data</xml>"

def test_api_error_event(client, session):
    with patch("Backend.services.crypto_service.CryptoService.fetch_crypto_data_from_api") as mock_fetch_crypto_data_from_api:
        mock_fetch_crypto_data_from_api.side_effect = HTTPException(status_code=500, detail="Mocked error")

        response = client.get("/analyze/")

        assert response.status_code == 500
        events = session.query(event_model.EventData).all()
        assert len(events) > 0

        error_event = session.query(event_model.EventData).filter(event_model.EventData.event_name == "API_ERROR").first()
        assert error_event is not None
        assert "Mocked error" in error_event.event_message

def test_analyze_data_xml_creation(client, session):
    # Mock the API response
    with patch("Backend.services.crypto_service.CryptoService.fetch_crypto_data_from_api") as mock_fetch_crypto_data_from_api:
        mock_fetch_crypto_data_from_api.return_value = [{"current_price": 123.45, "total_volume": 67890}]

        # Call the analyze endpoint
        response = client.get("/analyze/")
        assert response.status_code == 200

        # Verify database entry
        crypto_entry = session.query(crypto_model.CryptoData).first()
        assert crypto_entry is not None
        assert crypto_entry.current_price == 123.45
        assert crypto_entry.total_volume == 67890
