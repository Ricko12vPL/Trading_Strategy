#!/usr/bin/env python3
"""
QUICK TEST - Data Download Fix
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_download(symbol: str) -> pd.DataFrame:
    """Test yfinance download with robust handling"""
    try:
        logger.info(f"Testing download for {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10*365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Original columns for {symbol}: {df.columns.tolist()}")
        logger.info(f"Data shape: {df.shape}")
        
        # Handle different formats
        if 'Open' in df.columns and 'High' in df.columns:
            required_cols = ['Open', 'High', 'Low', 'Close']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) >= 4:
                if 'Volume' in df.columns:
                    df = df[available_cols + ['Volume']]
                else:
                    df = df[available_cols]
                    df['Volume'] = 1000000
        
        logger.info(f"Final columns for {symbol}: {df.columns.tolist()}")
        logger.info(f"Downloaded {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return pd.DataFrame()

# Test a few symbols
test_symbols = ['AAPL', 'SPY', 'QQQ']

print("🔄 Testing yfinance downloads...")
successful = 0
for symbol in test_symbols:
    df = test_download(symbol)
    if not df.empty:
        successful += 1
        print(f"✅ {symbol}: SUCCESS ({len(df)} rows)")
    else:
        print(f"❌ {symbol}: FAILED")

print(f"\nResult: {successful}/{len(test_symbols)} symbols successful")