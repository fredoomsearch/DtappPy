from sqlalchemy.orm import Session
from Backend.models import crypto_model
from Backend.schemas import CryptoCreate
from typing import List, Optional
from Backend.repositories.base_repository import BaseRepository

class CryptoRepository(BaseRepository[crypto_model.CryptoData, CryptoCreate]):
    def __init__(self, db: Session):
        super().__init__(crypto_model.CryptoData, db)

    def get(self, id: int) -> Optional[crypto_model.CryptoData]:
        return self.db.query(self.model).filter(self.model.id == id).first()

    def get_multi(self, skip: int = 0, limit: int = 100) -> List[crypto_model.CryptoData]:
        return self.db.query(self.model).offset(skip).limit(limit).all()

    def create(self, obj_in: CryptoCreate) -> crypto_model.CryptoData:
        db_obj = crypto_model.CryptoData(**obj_in.model_dump())
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

def get_crypto_data(db: Session, skip: int = 0, limit: int = 100):
    repo = CryptoRepository(db)
    return repo.get_multi(skip=skip, limit=limit)

def get_crypto_data_by_id(db: Session, crypto_id: int):
    repo = CryptoRepository(db)
    return repo.get(id=crypto_id)

def create_crypto_data(db: Session, crypto: CryptoCreate):
    repo = CryptoRepository(db)
    return repo.create(obj_in=crypto)