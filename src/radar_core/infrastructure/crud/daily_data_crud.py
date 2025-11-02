# src/radar_core/infrastructure/crud/daily_data_crud.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import date
# decimal: provides support for fast correctly rounded decimal floating-point arithmetic
from decimal import Decimal

# --- Third Party Libraries ---
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy import and_, asc, desc
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import DailyData


class DailyDataCrud(BaseCrud):
    def __init__(self):
        super().__init__(DailyData)

    def upsert(self,
               security_id: int,
               df: pl.DataFrame) -> None:
        """
        Update or insert daily prices

        :param security_id: Security id to update or insert
        :param df: Datasource

        :return: None
        """
        self.session.expire_on_commit = False

        # The parameter named=True: returns dictionaries instead of tuples.
        # The dictionaries are a mapping of column name to row value.
        # This is more expensive than returning a regular tuple, but allows for accessing values by column name.
        for row_ in df.iter_rows(named=True):
            statement_ = (select(DailyData)
                          .where(and_(DailyData.security_id == security_id, DailyData.date == row_['Date'])))
            saved_row_ = self.session.execute(statement_).first()

            if saved_row_:
                saved_record_: DailyData = saved_row_[0]

                # Set prices
                saved_record_.open = Decimal(row_['Open'])
                saved_record_.high = Decimal(row_['High'])
                saved_record_.low = Decimal(row_['Low'])
                saved_record_.close = Decimal(row_['Close'])
                saved_record_.volume = int(row_['Volume'])
                # Attempt to convert to numeric and check if it's not null (therefore successful)
                if row_['PercentChange'] is not None:
                    saved_record_.percent_change = Decimal(row_['PercentChange'])

                self.session.merge(saved_record_)
            else:
                new_record_ = DailyData(
                    # Set unique constraint key
                    security_id=security_id,
                    date=row_['Date'],

                    # Set prices
                    open=Decimal(row_['Open']),
                    high=Decimal(row_['High']),
                    low=Decimal(row_['Low']),
                    close=Decimal(row_['Close']),
                    volume=int(row_['Volume']),
                    percent_change=None if row_['PercentChange'] is None else Decimal(row_['PercentChange']),
                )
                self.session.add(new_record_)

        self.session.commit()

    def get_latest_prices_by_security(self,
                                      security_id: int) -> DailyData | None:
        """
        Gets historical daily prices for a stock/security for the most recent date in the database.
        :param security_id: Security id to get historical daily prices.
        :return: Historical daily prices for the most recent date in the database.
        """
        statement_ = (select(DailyData)
                      .where(and_(DailyData.security_id == security_id))
                      .order_by(desc(DailyData.date)))
        row_ = self.session.execute(statement_).first()
        if row_:
            return row_[0]
        return None

    def get_prices_by_security(self,
                               security_id: int | Mapped[int],
                               from_dt: date = None,
                               to_dt: date = None,
                               only_close=False) -> pl.DataFrame:
        """
        Gets historical daily prices for a stock/security in the requested date range.

        :param security_id: Security id to get historical daily prices.
        :param from_dt: Date from which historical prices are requested, if it is None does not filter by start date.
        :param to_dt: Date up to which historical prices are requested, if it is None does not filter by end date.
        :param only_close: Only return Close prices.

        :return: Polars.DataFrame formatted as [Date, Open, High, Low, Close, Volume] index datetime.
        """
        # Select only required columns...
        statement_ = select(
            DailyData.date.label('Date'),
            DailyData.close.label('Close'),
            DailyData.volume.label('Volume'),
            DailyData.percent_change.label('PercentChange')
        ) if only_close else select(
            DailyData.date.label('Date'),
            DailyData.open.label('Open'),
            DailyData.high.label('High'),
            DailyData.low.label('Low'),
            DailyData.close.label('Close'),
            DailyData.volume.label('Volume'),
            DailyData.percent_change.label('PercentChange')
        )
        # Init query
        # noinspection PyTypeChecker
        statement_ = statement_.where(DailyData.security_id == security_id)
        # filter by date range and...
        if from_dt is not None:
            statement_ = statement_.where(DailyData.date >= from_dt)
        if to_dt is not None:
            statement_ = statement_.where(DailyData.date <= to_dt)
        # order by date_time
        statement_ = statement_.order_by(asc(DailyData.date))

        # Read prices into dataframe with Pandas
        # df_ = pd.read_sql(statement_, engine, coerce_float=True) ←

        # Execute the query
        records_ = self.session.execute(statement_).all()

        # Write the results to a Polars DataFrame
        df_ = pl.DataFrame(records_)

        # Coerce specified columns to float and integer and return
        return df_.with_columns([
            df_["Open"].cast(pl.Float64),
            df_["High"].cast(pl.Float64),
            df_["Low"].cast(pl.Float64),
            df_["Close"].cast(pl.Float64),
            df_["PercentChange"].cast(pl.Float64),
            df_["Volume"].cast(pl.Int64)
        ])
