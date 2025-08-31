#!/usr/bin/env python3
"""
Test timezone-aware datetime filtering
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_timezone_fix(symbol: str) -> pd.DataFrame:
    """Test timezone-aware filtering"""
    try:
        logger.info(f"Testing timezone fix for {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10*365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Data index timezone: {df.index.tz}")
        logger.info(f"Original data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        
        # Timezone-aware filtering
        min_cutoff = datetime.now() - timedelta(days=8*365)
        logger.info(f"Cutoff date (naive): {min_cutoff}")
        
        if df.index.tz is not None:
            min_cutoff = pd.Timestamp(min_cutoff).tz_localize('UTC').tz_convert(df.index.tz)
            logger.info(f"Cutoff date (timezone-aware): {min_cutoff}")
        
        df_filtered = df[df.index >= min_cutoff]
        logger.info(f"Filtered data: {len(df_filtered)} rows from {df_filtered.index[0]} to {df_filtered.index[-1]}")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        return pd.DataFrame()

# Test
test_timezone_fix('AAPL')