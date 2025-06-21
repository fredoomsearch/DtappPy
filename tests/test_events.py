import pytest
import requests
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app
from utils import database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SQLAlchemy Database URL
SQLALCHEMY_DATABASE_URL = "postgresql://myuser:mypassword@localhost/mydb"

# Create engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test if the connection works
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")


