# src/radar_core/infrastructure/crud/base_crud.py

class BaseCrud(object):
    """
    Base class for repository CRUD operations.
    """

    def __init__(self, base_model=None):
        self.base_model = base_model

