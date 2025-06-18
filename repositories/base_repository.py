from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from typing import List, Optional, Generic, TypeVar, Type

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")

class BaseRepository(ABC, Generic[ModelType, CreateSchemaType]):
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db

    @abstractmethod
    def get(self, id: int) -> Optional[ModelType]:
        raise NotImplementedError

    @abstractmethod
    def get_multi(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        raise NotImplementedError

    @abstractmethod
    def create(self, obj_in: CreateSchemaType) -> ModelType:
        raise NotImplementedError

    # Add other common repository methods (update, delete) as needed