# tests/infrastructure/test_price_provider.py

# --- Python modules ---
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pandas as pd
import polars as pl

# --- App modules ---
from radar_core.helpers.constants import ORDERED_PRICE_COLS
from radar_core.infrastructure.price_cache import PriceCache, PriceCacheMetadata
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.infrastructure.security_repository import SecurityRepository
from radar_core.models import Securities


def _make_mock_yfinance_df(tickers: list[str],
                           dates: pd.DatetimeIndex | None = None,
                           close_vals: list[float] | None = None) -> pd.DataFrame:
    """
    Helper to construct a mock MultiIndex DataFrame matching yfinance download structure.

    :param tickers: List of provider ticker strings.
    :param dates: Optional DatetimeIndex for the rows.
    :param close_vals: Optional list of close price floats.

    :return: Formatted pandas DataFrame with MultiIndex columns.
    """
    if dates is None:
        dates = pd.date_range('2026-01-02', periods=3, freq='B', name='Date')
    periods_ = len(dates)
    data_ = {}
    for ticker_ in tickers:
        data_[(ticker_, 'Open')] = [100.0 + i_ for i_ in range(periods_)]
        data_[(ticker_, 'High')] = [105.0 + i_ for i_ in range(periods_)]
        data_[(ticker_, 'Low')] = [99.0 + i_ for i_ in range(periods_)]
        data_[(ticker_, 'Close')] = close_vals if close_vals is not None else [103.0 + i_ for i_ in range(periods_)]
        data_[(ticker_, 'Volume')] = [100000 + i_ * 10000 for i_ in range(periods_)]

    df_ = pd.DataFrame(data_, index=dates)
    df_.columns = pd.MultiIndex.from_tuples(df_.columns)
    return df_


def test_get_or_create_security_returns_new_security():
    """
    GIVEN a security symbol 'NVDA' not present in the DB
    WHEN _get_or_create_security is called and Yahoo Finance returns info
    THEN it creates and returns the new Securities instance.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_by_symbol', return_value=None), \
         patch.object(repo_._SecurityRepository__security_crud, 'add_security') as mock_add_, \
         patch('yfinance.Ticker') as mock_ticker_cls_:

        mock_ticker_inst_ = MagicMock()
        mock_ticker_inst_.info = {'longName': 'NVIDIA Corporation'}
        mock_ticker_cls_.return_value = mock_ticker_inst_

        security_ = repo_._get_or_create_security('NVDA')

        assert security_ is not None
        assert security_.symbol == 'NVDA'
        assert security_.description == 'NVIDIA Corporation'
        mock_add_.assert_called_once()


def test_map_symbol_to_ticker_auto_creates_missing_symbols():
    """
    GIVEN symbols ['SPY', 'NEW_SYM'] where 'NEW_SYM' is missing from DB
    WHEN map_symbol_to_ticker is called
    THEN missing symbols are created and mapped successfully.
    """
    repo_ = SecurityRepository()
    mock_db_map_ = {'SPY': 'SPY'}
    new_sec_ = Securities(id=2, symbol='NEW_SYM', description='New Symbol Inc')

    with patch.object(repo_._SecurityRepository__security_crud, 'get_tickers_by_symbols', return_value=mock_db_map_), \
         patch.object(repo_, '_get_or_create_security', return_value=new_sec_) as mock_create_, \
         patch.object(repo_, '_get_ticker', return_value='NEW_SYM'):

        result_ = repo_.map_symbol_to_ticker(['SPY', 'NEW_SYM'])

        assert result_ == {'SPY': 'SPY', 'NEW_SYM': 'NEW_SYM'}
        mock_create_.assert_called_once_with('NEW_SYM')


def test_map_symbol_to_ticker_omits_symbols_not_in_yahoo():
    """
    GIVEN a symbol 'INVALID_SYM' not in DB and not found on Yahoo Finance
    WHEN map_symbol_to_ticker is called
    THEN the invalid symbol is omitted from the returned mapping.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_tickers_by_symbols', return_value={}), \
         patch.object(repo_, '_get_or_create_security', return_value=None):

        result_ = repo_.map_symbol_to_ticker(['INVALID_SYM'])

        assert result_ == {}


def test_price_provider_empty_symbols_guard():
    """
    GIVEN an empty list of symbols
    WHEN PriceProvider.get_prices is called
    THEN it returns an empty dictionary immediately without mapping or downloading.
    """
    provider_ = PriceProvider()
    with patch.object(SecurityRepository, 'map_symbol_to_ticker') as mock_map_, \
         patch('yfinance.download') as mock_download_:
        results_ = provider_.get_prices([], datetime.now(timezone.utc))
        assert results_ == {}
        mock_map_.assert_not_called()
        mock_download_.assert_not_called()


def test_price_provider_empty_tickers_guard():
    """
    GIVEN a list of symbols that maps to 0 tickers
    WHEN PriceProvider.get_prices is called
    THEN it returns an empty dictionary cleanly without calling yfinance download.
    """
    provider_ = PriceProvider()
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value={}), \
         patch('yfinance.download') as mock_download_:
        results_ = provider_.get_prices(['INVALID_SYMBOL'], datetime.now(timezone.utc))
        assert results_ == {}
        mock_download_.assert_not_called()


def test_full_download_writes_price_cache(tmp_path):
    """
    GIVEN PriceProvider configured with an active cache directory
    WHEN get_prices is called and all requested symbols are downloaded successfully
    THEN a complete price cache is saved containing all symbols and metadata.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_

    symbols_ = ['SPY', 'QQQ']
    mock_mapping_ = {'SPY': 'SPY', 'QQQ': 'QQQ'}
    mock_df_ = _make_mock_yfinance_df(['SPY', 'QQQ'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mock_mapping_), \
         patch('yfinance.download', return_value=mock_df_):

        results_ = provider_.get_prices(symbols_, datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))

        assert len(results_) == 2
        assert 'SPY' in results_
        assert 'QQQ' in results_
        assert results_['SPY'].columns == ORDERED_PRICE_COLS

        # Verify price cache was saved to disk
        cached_df_ = cache_.read_data()
        metadata_ = cache_.read_metadata()
        assert cached_df_ is not None
        assert metadata_ is not None
        assert metadata_.is_complete is True
        assert metadata_.symbol_to_ticker == mock_mapping_
        assert metadata_.start_date == str(provider_.start_date)

        # Verify cached DataFrame structure
        assert 'Symbol' in cached_df_.columns
        assert 'PercentChange' not in cached_df_.columns
        assert set(cached_df_['Symbol'].unique().to_list()) == {'SPY', 'QQQ'}


def test_partial_download_skips_cache_write(tmp_path):
    """
    GIVEN PriceProvider configured with an active cache directory
    WHEN get_prices is called but one symbol fails to download (partial result)
    THEN the price cache is not written to disk and partial results are returned.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_

    symbols_ = ['SPY', 'FAILING']
    mock_mapping_ = {'SPY': 'SPY', 'FAILING': 'FAILING'}
    mock_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mock_mapping_), \
         patch('yfinance.download', return_value=mock_df_):

        results_ = provider_.get_prices(symbols_, datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))

        assert len(results_) == 1
        assert 'SPY' in results_
        assert 'FAILING' not in results_
        assert cache_.read_data() is None
        assert cache_.read_metadata() is None


def test_cache_write_disabled_skips_save(tmp_path):
    """
    GIVEN PriceProvider configured with write=False in cache settings
    WHEN get_prices successfully downloads all symbols
    THEN the price cache is not written to disk.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_._cache_settings = dict(provider_._cache_settings)
    provider_._cache_settings['write'] = False

    symbols_ = ['SPY']
    mock_mapping_ = {'SPY': 'SPY'}
    mock_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mock_mapping_), \
         patch('yfinance.download', return_value=mock_df_):

        results_ = provider_.get_prices(symbols_, datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))

        assert len(results_) == 1
        assert 'SPY' in results_
        assert cache_.read_data() is None
        assert cache_.read_metadata() is None


def test_cache_save_error_does_not_crash_get_prices(tmp_path):
    """
    GIVEN PriceProvider configured with an active cache directory
    WHEN PriceCache.save raises an unexpected exception (e.g. disk failure)
    THEN the exception is handled gracefully and get_prices returns downloaded data.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_

    symbols_ = ['SPY']
    mock_mapping_ = {'SPY': 'SPY'}
    mock_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mock_mapping_), \
         patch('yfinance.download', return_value=mock_df_), \
         patch.object(cache_, 'save', side_effect=OSError('Disk write error')):

        results_ = provider_.get_prices(symbols_, datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))

        assert len(results_) == 1
        assert 'SPY' in results_


def test_cache_eligibility_production_window(tmp_path):
    """
    GIVEN PriceProvider configured with price cache
    WHEN evaluating _is_cache_eligible across different times, weekdays, and metadata states
    THEN it strictly permits only weekday market hours with matching session date.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    provider_._cache_settings = dict(provider_._cache_settings)
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}
    trading_time_ = datetime(2026, 1, 9, 11, 0, tzinfo=tz_)

    # 1. Disabled or ignored cache
    provider_._cache_settings['enabled'] = False
    assert provider_._is_cache_eligible(mapping_, trading_time_) is False
    provider_._cache_settings['enabled'] = True

    provider_._cache_settings['ignore'] = True
    assert provider_._is_cache_eligible(mapping_, trading_time_) is False
    provider_._cache_settings['ignore'] = False

    # 2. Weekend (Saturday Jan 10, 2026)
    sat_ = datetime(2026, 1, 10, 11, 0, tzinfo=tz_)
    assert provider_._is_cache_eligible(mapping_, now=sat_) is False

    # 3. Weekday outside trading window (Friday Jan 9, 2026 at 09:00 and 17:30)
    early_ = datetime(2026, 1, 9, 9, 0, tzinfo=tz_)
    late_ = datetime(2026, 1, 9, 17, 30, tzinfo=tz_)
    assert provider_._is_cache_eligible(mapping_, now=early_) is False
    assert provider_._is_cache_eligible(mapping_, now=late_) is False

    # 4. Weekday inside trading window (Friday Jan 9, 2026 at 11:00) with no cache on disk
    trading_time_ = datetime(2026, 1, 9, 11, 0, tzinfo=tz_)
    assert provider_._is_cache_eligible(mapping_, now=trading_time_) is False

    # 5. Save cache with session_date = '2026-01-09'
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-01-09T16:00:00+00:00',
    )
    df_ = pl.DataFrame({
        'Date': [date(2026, 1, 8), date(2026, 1, 9)],
        'Open': [100.0, 102.0],
        'High': [105.0, 106.0],
        'Low': [99.0, 101.0],
        'Close': [103.0, 105.0],
        'Volume': [1000, 1200],
        'Symbol': ['SPY', 'SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(df_, meta_)

    # Inside window and session date matches -> Eligible
    assert provider_._is_cache_eligible(mapping_, now=trading_time_) is True

    # Next session date (Monday Jan 12, 2026) -> Ineligible due to session date mismatch
    next_monday_ = datetime(2026, 1, 12, 11, 0, tzinfo=tz_)
    assert provider_._is_cache_eligible(mapping_, now=next_monday_) is False


def test_cache_eligibility_development_environment(tmp_path):
    """
    GIVEN PriceProvider in development mode (app_environment='dev')
    WHEN evaluating _is_cache_eligible outside trading hours
    THEN it permits cache reuse within dev_max_age_minutes, rejects when stale, while prod rejects outside hours.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    # 20:05 is outside market window (09:30 - 16:00)
    now_within_ttl_ = datetime(2026, 1, 9, 20, 5, tzinfo=tz_)
    now_utc_ = now_within_ttl_.astimezone(timezone.utc)
    updated_at_utc_ = (now_utc_ - timedelta(minutes=5)).isoformat()
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=updated_at_utc_,
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)

    # Post-market cache on Friday (updated 20:00 >= 16:00) -> Eligible in both DEV and PROD (even after 20 minutes)
    assert provider_._is_cache_eligible(mapping_, now_within_ttl_) is True
    provider_.app_environment = 'prod'
    assert provider_._is_cache_eligible(mapping_, now_within_ttl_) is True

    # 20 minutes later on the same evening: still eligible in both DEV and PROD because it is settled post-close
    stale_post_close_now_ = datetime(2026, 1, 9, 20, 20, tzinfo=tz_)
    assert provider_._is_cache_eligible(mapping_, stale_post_close_now_) is True
    provider_.app_environment = 'dev'
    assert provider_._is_cache_eligible(mapping_, stale_post_close_now_) is True

    # Pre-market cache on Friday (updated 08:30 < 09:30) -> Ineligible after market open in both PROD and DEV
    premarket_meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-premarket',
        is_complete=True,
        updated_at_utc=datetime(2026, 1, 9, 8, 30, tzinfo=tz_).astimezone(timezone.utc).isoformat(),
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), premarket_meta_)
    assert provider_._is_cache_eligible(mapping_, now_within_ttl_) is False
    provider_.app_environment = 'prod'
    assert provider_._is_cache_eligible(mapping_, now_within_ttl_) is False

    # Weekend (not post-market settled): within 5 min TTL -> Eligible in DEV, Ineligible in PROD
    sat_now_ = datetime(2026, 1, 10, 11, 5, tzinfo=tz_)
    sat_meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-10',
        generation_id='generation-sat',
        is_complete=True,
        updated_at_utc=datetime(2026, 1, 10, 11, 0, tzinfo=tz_).astimezone(timezone.utc).isoformat(),
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 10)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), sat_meta_)
    provider_.app_environment = 'dev'
    assert provider_._is_cache_eligible(mapping_, sat_now_) is True
    provider_.app_environment = 'prod'
    assert provider_._is_cache_eligible(mapping_, sat_now_) is False

    # Weekend when stale (20 min old > 10 min TTL) -> Ineligible in both DEV and PROD
    sat_stale_now_ = datetime(2026, 1, 10, 11, 20, tzinfo=tz_)
    provider_.app_environment = 'dev'
    assert provider_._is_cache_eligible(mapping_, sat_stale_now_) is False
    provider_.app_environment = 'prod'
    assert provider_._is_cache_eligible(mapping_, sat_stale_now_) is False

    # Inside market window -> Eligible in both DEV and PROD
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)
    trading_now_ = datetime(2026, 1, 9, 11, 0, tzinfo=tz_)
    provider_.app_environment = 'dev'
    assert provider_._is_cache_eligible(mapping_, trading_now_) is True
    provider_.app_environment = 'prod'
    assert provider_._is_cache_eligible(mapping_, trading_now_) is True


def test_eligible_execution_makes_current_day_request_and_refreshes_rows(tmp_path):
    """
    GIVEN an existing valid cache with historical data up to current session date
    WHEN get_prices runs in an eligible production window
    THEN it requests only the current date from Yahoo, replaces today's row, and recalculates PercentChange.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    # Existing cache has Jan 2 (100.0) and Jan 5 (105.0)
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-01-05T16:00:00+00:00',
    )
    cached_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 2), date(2026, 1, 5)],
        'Open': [100.0, 102.0],
        'High': [105.0, 106.0],
        'Low': [99.0, 101.0],
        'Close': [100.0, 105.0],
        'Volume': [100000, 120000],
        'Symbol': ['SPY', 'SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(cached_df_, meta_)

    # Current time is Monday Jan 5, 2026 at 14:00 (inside window, session date matches)
    mock_now_ = datetime(2026, 1, 5, 14, 0, tzinfo=tz_)

    # Refreshed bar for Jan 5 has updated Close=110.0
    today_dates_ = pd.DatetimeIndex(['2026-01-05'], name='Date')
    mock_today_df_ = _make_mock_yfinance_df(['SPY'], dates=today_dates_, close_vals=[110.0])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=mock_today_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], mock_now_)

        assert len(results_) == 1
        assert 'SPY' in results_
        spy_df_ = results_['SPY']

        # Verify yfinance was called for today starting at current_date (start=2026-01-05, end=provider_.end_date)
        mock_download_.assert_called_once()
        args_, _ = mock_download_.call_args
        assert args_[1] == date(2026, 1, 5)
        assert args_[2] == provider_.end_date

        # Verify row replacement and recalculation of PercentChange
        assert spy_df_.height == 2
        dates_list_ = spy_df_['Date'].to_list()
        assert dates_list_ == [date(2026, 1, 2), date(2026, 1, 5)]
        closes_list_ = spy_df_['Close'].to_list()
        assert closes_list_ == [100.0, 110.0]

        # PercentChange: Jan 2 is null, Jan 5 is ((110 - 100) / 100) * 100 = 10.0%
        pct_list_ = spy_df_['PercentChange'].to_list()
        assert pct_list_[0] is None
        assert abs(pct_list_[1] - 10.0) < 1e-4

        # Verify in-memory results have updated row while disk cache preserves settled baseline (no provisional overwrite)
        loaded_df_ = cache_.read_data()
        loaded_meta_ = cache_.read_metadata()
        assert loaded_df_ is not None
        assert loaded_meta_ is not None
        assert loaded_df_['Close'][-1] == 105.0
        assert loaded_meta_.session_date == '2026-01-05'


def test_ineligible_execution_makes_complete_request(tmp_path):
    """
    GIVEN an existing cache but execution is outside the trading window (e.g. Saturday)
    WHEN get_prices is called
    THEN it performs a full historical download starting from self.start_date.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    # Save cache with session_date = '2026-01-09'
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-01-09T16:00:00+00:00',
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [103.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)

    # Weekend run (Saturday Jan 10)
    mock_now_ = datetime(2026, 1, 10, 11, 0, tzinfo=tz_)
    full_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], mock_now_)

        assert len(results_) == 1
        # Called with full historical range (positional args: tickers, start_date, end_date)
        mock_download_.assert_called_once()
        args_, _ = mock_download_.call_args
        assert args_[1] == provider_.start_date
        assert args_[2] == provider_.end_date


def test_failed_refresh_falls_back_to_full_download(tmp_path):
    """
    GIVEN an eligible cache state where current-day download returns empty
    WHEN get_prices is called
    THEN it falls back cleanly to the full historical download and preserves existing cache.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-01-05T16:00:00+00:00',
    )
    cached_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 2), date(2026, 1, 5)],
        'Open': [100.0, 102.0], 'High': [105.0, 106.0],
        'Low': [99.0, 101.0], 'Close': [100.0, 105.0],
        'Volume': [100000, 120000], 'Symbol': ['SPY', 'SPY']
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(cached_df_, meta_)

    mock_now_ = datetime(2026, 1, 5, 14, 0, tzinfo=tz_)
    empty_today_df_ = pd.DataFrame()
    full_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', side_effect=[empty_today_df_, full_df_]) as mock_download_:

        results_ = provider_.get_prices(['SPY'], mock_now_)

        assert len(results_) == 1
        assert mock_download_.call_count == 2
        # First call was today refresh attempt
        assert mock_download_.call_args_list[0].args[1] == date(2026, 1, 5)
        # Second call was full download
        assert mock_download_.call_args_list[1].args[1] == provider_.start_date

        # Existing cache remained intact
        post_meta_ = cache_.read_metadata()
        assert post_meta_ is not None
        assert post_meta_.is_complete is True


def test_dev_cache_eligible_performs_current_day_refresh(tmp_path):
    """
    GIVEN a compatible price cache updated within dev_max_age_minutes
    WHEN get_prices runs in development mode (app_environment='dev')
    THEN it performs a current-day refresh downloading only today's bar and merges with history.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    mapping_ = {'SPY': 'SPY'}

    now_utc_ = datetime(2026, 1, 9, 20, 0, tzinfo=timezone.utc)
    updated_at_utc_ = (now_utc_ - timedelta(minutes=2)).isoformat()

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=updated_at_utc_,
    )
    cached_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 8), date(2026, 1, 9)],
        'Open': [100.0, 102.0],
        'High': [105.0, 106.0],
        'Low': [99.0, 101.0],
        'Close': [100.0, 105.0],
        'Volume': [100000, 120000],
        'Symbol': ['SPY', 'SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(cached_df_, meta_)

    today_dates_ = pd.DatetimeIndex(['2026-01-09'], name='Date')
    mock_today_df_ = _make_mock_yfinance_df(['SPY'], dates=today_dates_, close_vals=[110.0])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=mock_today_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], now_utc_)

        assert len(results_) == 1
        assert 'SPY' in results_
        spy_df_ = results_['SPY']
        assert spy_df_.columns == ORDERED_PRICE_COLS
        assert spy_df_.height == 2
        assert spy_df_['Close'].to_list() == [100.0, 110.0]
        # PercentChange: ((110 - 100) / 100) * 100 = 10.0%
        assert abs(spy_df_['PercentChange'][1] - 10.0) < 1e-4

        mock_download_.assert_called_once()
        args_, _ = mock_download_.call_args
        assert args_[1] == date(2026, 1, 9)
        assert args_[2] == provider_.end_date


def test_dev_cache_stale_triggers_complete_download(tmp_path):
    """
    GIVEN a compatible price cache that is older than dev_max_age_minutes
    WHEN get_prices runs in development mode (app_environment='dev')
    THEN it treats the cache as expired, performs a complete download, and updates the cache.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    mapping_ = {'SPY': 'SPY'}

    tz_ = provider_._cache_settings['timezone']
    # Saturday Jan 10, 2026 (outside market window and not post-market settled)
    sat_now_ = datetime(2026, 1, 10, 11, 0, tzinfo=tz_)
    now_utc_ = sat_now_.astimezone(timezone.utc)
    # Stale cache: 15 minutes old (dev_max_age_minutes defaults to 10)
    updated_at_utc_ = (now_utc_ - timedelta(minutes=15)).isoformat()

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-10',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=updated_at_utc_,
    )
    cached_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 10)],
        'Open': [100.0], 'High': [105.0], 'Low': [99.0], 'Close': [100.0],
        'Volume': [100000], 'Symbol': ['SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(cached_df_, meta_)

    mock_df_ = _make_mock_yfinance_df(['SPY'], close_vals=[120.0, 125.0, 130.0])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=mock_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], sat_now_)

        assert len(results_) == 1
        mock_download_.assert_called_once()
        args_, _ = mock_download_.call_args
        assert args_[1] == provider_.start_date

        # Cache was updated on disk with new data
        loaded_df_ = cache_.read_data()
        assert loaded_df_ is not None
        assert loaded_df_['Close'][-1] == 130.0


def test_dev_max_age_not_permitted_in_production(tmp_path):
    """
    GIVEN a compatible price cache updated 2 minutes ago
    WHEN get_prices runs in production mode (app_environment='prod') outside the market window (e.g. Saturday)
    THEN dev_max_age_minutes does NOT permit reuse and a complete download is performed.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    # Saturday Jan 10, 2026 (outside market window)
    sat_now_ = datetime(2026, 1, 10, 11, 0, tzinfo=tz_)
    now_utc_ = sat_now_.astimezone(timezone.utc)
    updated_at_utc_ = (now_utc_ - timedelta(minutes=2)).isoformat()

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=updated_at_utc_,
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)

    full_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], sat_now_)

        assert len(results_) == 1
        mock_download_.assert_called_once()
        args_, _ = mock_download_.call_args
        assert args_[1] == provider_.start_date


def test_cache_disabled_and_ignore_bypass_cache_reads(tmp_path):
    """
    GIVEN a valid fresh cache on disk
    WHEN cache is disabled (enabled=False) or ignored (ignore=True)
    THEN get_prices bypasses cache reads and performs a complete download from Yahoo Finance.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    provider_._cache_settings = dict(provider_._cache_settings)
    mapping_ = {'SPY': 'SPY'}

    now_utc_ = datetime(2026, 1, 9, 20, 0, tzinfo=timezone.utc)
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=now_utc_.isoformat(),
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)

    full_df_ = _make_mock_yfinance_df(['SPY'])

    # Case 1: cache disabled (enabled=False)
    provider_._cache_settings['enabled'] = False
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], now_utc_)
        assert len(results_) == 1
        mock_download_.assert_called_once()

    # Case 2: cache ignored (ignore=True)
    provider_._cache_settings['enabled'] = True
    provider_._cache_settings['ignore'] = True
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], now_utc_)
        assert len(results_) == 1
        mock_download_.assert_called_once()


def test_cache_disabled_skips_save(tmp_path):
    """
    GIVEN cache settings with enabled=False
    WHEN get_prices successfully downloads all symbols
    THEN the price cache is not written to disk.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_._cache_settings = dict(provider_._cache_settings)
    provider_._cache_settings['enabled'] = False

    symbols_ = ['SPY']
    mock_mapping_ = {'SPY': 'SPY'}
    mock_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mock_mapping_), \
         patch('yfinance.download', return_value=mock_df_):

        results_ = provider_.get_prices(symbols_, datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))

        assert len(results_) == 1
        assert cache_.read_data() is None
        assert cache_.read_metadata() is None


def test_cache_consumption_failure_falls_back_to_download(tmp_path):
    """
    GIVEN a fresh development cache on disk with malformed or missing columns
    WHEN get_prices is called in development mode
    THEN _get_dev_cached_prices catches the error, returns None, and get_prices cleanly falls back to complete download.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    mapping_ = {'SPY': 'SPY'}

    now_utc_ = datetime(2026, 1, 9, 20, 0, tzinfo=timezone.utc)
    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=now_utc_.isoformat(),
    )
    # Save a malformed DataFrame missing required OHLCV columns (e.g. missing 'Close')
    malformed_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 9)],
        'Open': [100.0],
        'Symbol': ['SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(malformed_df_, meta_)

    full_df_ = _make_mock_yfinance_df(['SPY'], close_vals=[150.0, 155.0, 160.0])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], now_utc_)

        assert len(results_) == 1
        assert mock_download_.call_count == 2
        # First call was today refresh attempt
        assert mock_download_.call_args_list[0].args[1] == date(2026, 1, 9)
        # Second call was full download fallback
        assert mock_download_.call_args_list[1].args[1] == provider_.start_date
        assert results_['SPY']['Close'].to_list() == [150.0, 155.0, 160.0]


def test_dev_cache_invalid_timestamp_falls_back_to_download(tmp_path):
    """
    GIVEN a cache on disk with an invalid or unparseable updated_at_utc timestamp
    WHEN get_prices is called in development mode
    THEN _get_dev_cached_prices returns None and get_prices performs a complete download.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'dev'
    tz_ = provider_._cache_settings['timezone']
    now_market_ = datetime(2026, 1, 9, 20, 0, tzinfo=tz_)
    mapping_ = {'SPY': 'SPY'}

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='not_a_valid_timestamp',
    )
    cached_df_ = pl.DataFrame({
        'Date': [date(2026, 1, 9)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [100.0], 'Volume': [1000], 'Symbol': ['SPY'],
    }).with_columns(pl.col('Date').cast(pl.Date))
    cache_.save(cached_df_, meta_)

    full_df_ = _make_mock_yfinance_df(['SPY'])

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value=mapping_), \
         patch('yfinance.download', return_value=full_df_) as mock_download_:

        results_ = provider_.get_prices(['SPY'], now_market_)

        assert len(results_) == 1
        mock_download_.assert_called_once()


def test_price_provider_end_date_uses_cache_timezone():
    """
    GIVEN PriceProvider initialized with a specific timezone in cache settings
    WHEN end_date is calculated
    THEN it derives end_date from datetime.now in the configured timezone plus 1 day.
    """
    with patch('radar_core.infrastructure.price_provider.datetime') as mock_dt_:
        fixed_dt_ = datetime(2026, 1, 5, 23, 30, tzinfo=timezone.utc)
        mock_dt_.now.return_value = fixed_dt_

        provider_ = PriceProvider()
        assert provider_.end_date == date(2026, 1, 6)


def test_post_market_close_cache_eligible_in_both_environments(tmp_path):
    """
    GIVEN a compatible price cache generated after market close on a weekday
    WHEN evaluating _is_cache_eligible in production and development mode after trading hours
    THEN it permits cache reuse on the same session date across both environments.
    """
    provider_ = PriceProvider()
    cache_ = PriceCache(tmp_path)
    provider_._price_cache = cache_
    provider_.app_environment = 'prod'
    tz_ = provider_._cache_settings['timezone']
    mapping_ = {'SPY': 'SPY'}

    # Monday Jan 5, 2026 at 18:30 (after 16:00 market close)
    post_close_now_ = datetime(2026, 1, 5, 18, 30, tzinfo=tz_)
    post_close_updated_ = datetime(2026, 1, 5, 16, 30, tzinfo=tz_).astimezone(timezone.utc).isoformat()

    meta_ = PriceCacheMetadata(
        symbol_to_ticker=mapping_,
        start_date=str(provider_.start_date),
        session_date='2026-01-05',
        generation_id='generation-post-close',
        is_complete=True,
        updated_at_utc=post_close_updated_,
    )
    cache_.save(pl.DataFrame({
        'Date': [date(2026, 1, 5)], 'Open': [100.0], 'High': [105.0],
        'Low': [99.0], 'Close': [104.0], 'Volume': [1000], 'Symbol': ['SPY']
    }).with_columns(pl.col('Date').cast(pl.Date)), meta_)

    assert provider_._is_cache_eligible(mapping_, post_close_now_) is True
    provider_.app_environment = 'dev'
    assert provider_._is_cache_eligible(mapping_, post_close_now_) is True

