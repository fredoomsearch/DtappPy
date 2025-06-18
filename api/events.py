from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from Backend.schemas import EventCreate # Import from schemas
from Backend.models.event_model import EventData
from Backend.utils import database
from Backend.services import event_service

router = APIRouter()

@router.post("/crypto_events/")
def create_event(event: EventCreate, db: Session = Depends(database.get_db)):
    return event_service.create_event(event, db)