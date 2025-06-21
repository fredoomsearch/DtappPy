from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import EventCreate # Import from schemas
from models.event_model import EventData
from utils import database
from services import event_service

router = APIRouter()

@router.post("/crypto_events/")
def create_event(event: EventCreate, db: Session = Depends(database.get_db)):
    return event_service.create_event(event, db)
