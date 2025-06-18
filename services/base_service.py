from abc import ABC, abstractmethod
from typing import Generic, TypeVar

CreateSchemaType = TypeVar("CreateSchemaType")

class BaseService(ABC, Generic[CreateSchemaType]):
    @abstractmethod
    def create(self, obj_in: CreateSchemaType):
        raise NotImplementedError

    # Add other common service methods as needed