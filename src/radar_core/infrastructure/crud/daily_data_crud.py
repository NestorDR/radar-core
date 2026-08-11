# src/radar_core/infrastructure/crud/daily_data_crud.py

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import DailyData


class DailyDataCrud(BaseCrud):
    def __init__(self):
        super().__init__(DailyData)

