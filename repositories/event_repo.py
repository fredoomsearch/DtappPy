from sqlalchemy.orm import Session
from models import event_model
from schemas import EventCreate # Import from schemas

def create_event(db: Session, event: EventCreate):
    db_event = event_model.EventData(**event.model_dump())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event
