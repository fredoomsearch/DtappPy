from sqlalchemy.orm import Session
from repositories import event_repo
from api.events import EventCreate

def create_event(event: EventCreate, db: Session):
    return event_repo.create_event(db, event)
