#!/usr/bin/env python3
"""
MINI TEST - Just 3 symbols to verify the complete pipeline works
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import vectorbt as vbt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 1200

class MiniTestBacktest:
    """Mini test with 3 symbols"""
    
    def __init__(self):
        self.initial_capital = 100000
        
        # Just 3 symbols for testing
        self.instruments = {
            'AAPL': {'category': 'mega_tech', 'sector': 'Technology'},
            'SPY': {'category': 'etf', 'sector': 'Market'},
            'QQQ': {'category': 'etf', 'sector': 'Technology'}
        }
        
        self.data = {}
        self.results = {}
    
    def download_yfinance_robust(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """Download historical data with robust yfinance handling"""
        try:
            logger.info(f"📊 Downloading {years} years data for {symbol}...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365 + 100)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                return pd.DataFrame()
            
            # Handle column formats
            if 'Open' in df.columns and 'High' in df.columns:
                required_cols = ['Open', 'High', 'Low', 'Close']
                available_cols = [col for col in required_cols if col in df.columns]
                
                if len(available_cols) >= 4:
                    if 'Volume' in df.columns:
                        df = df[available_cols + ['Volume']]
                    else:
                        df = df[available_cols]
                        df['Volume'] = 1000000
                else:
                    raise ValueError(f"Missing required OHLC columns for {symbol}")
            
            # Handle timezone-aware filtering
            min_cutoff = datetime.now() - timedelta(days=8*365)
            if df.index.tz is not None:
                min_cutoff = pd.Timestamp(min_cutoff).tz_localize('UTC').tz_convert(df.index.tz)
            df = df[df.index >= min_cutoff]
            
            if len(df) >= 2000:
                logger.info(f"✅ Downloaded {len(df)} days for {symbol}")
                return df
            else:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} days")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            if len(df) < 100:
                return df
            
            # Simple RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Returns and momentum
            df['Returns'] = df['Close'].pct_change()
            df['Momentum'] = df['Close'].pct_change(periods=5) * 100
            
            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            df = df.dropna()
            logger.info(f"✅ Calculated indicators for {symbol}, {len(df)} valid data points")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate indicators for {symbol}: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate trading signals"""
        try:
            df['Signal'] = 0
            df['Position'] = 0
            
            current_position = 0
            
            for i in range(1, len(df)):
                if (pd.isna(df['RSI'].iloc[i]) or pd.isna(df['BB_Lower'].iloc[i])):
                    df['Position'].iloc[i] = current_position
                    continue
                
                signals = []
                
                # Basic signals
                if (df['RSI'].iloc[i] < 35 and df['Close'].iloc[i] < df['BB_Lower'].iloc[i]):
                    signals.append(1)
                
                if df['RSI'].iloc[i] > 70:
                    signals.append(-1)
                
                # Volume confirmation
                if df['Volume_Ratio'].iloc[i] > 1.5:
                    if df['Returns'].iloc[i] > 0.02:
                        signals.append(1)
                    elif df['Returns'].iloc[i] < -0.02:
                        signals.append(-1)
                
                # Position management
                if len(signals) > 0:
                    signal_sum = sum(signals)
                    if signal_sum > 0 and current_position == 0:
                        df['Signal'].iloc[i] = 1
                        current_position = 1
                    elif signal_sum < 0 and current_position == 1:
                        df['Signal'].iloc[i] = -1
                        current_position = 0
                
                df['Position'].iloc[i] = current_position
            
            df['Entry'] = (df['Signal'] == 1).astype(int)
            df['Exit'] = (df['Signal'] == -1).astype(int)
            
            logger.info(f"✅ Generated signals for {symbol}: {df['Entry'].sum()} entries, {df['Exit'].sum()} exits")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals for {symbol}: {e}")
            return df
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run vectorbt backtest"""
        try:
            logger.info(f"🔄 Running backtest for {symbol}...")
            
            if 'Entry' not in df.columns or 'Exit' not in df.columns:
                return {}
            
            entries = df['Entry'].astype(bool)
            exits = df['Exit'].astype(bool)
            
            # Set frequency for VectorBT
            vbt.settings['array_wrapper']['freq'] = 'D'  # Daily frequency
            
            pf = vbt.Portfolio.from_signals(
                df['Close'],
                entries,
                exits,
                init_cash=self.initial_capital,
                fees=0.001,
                slippage=0.001,
                freq='D'
            )
            
            total_return = pf.total_return() * 100
            annual_return = pf.annualized_return() * 100
            sharpe_ratio = pf.sharpe_ratio()
            max_drawdown = abs(pf.max_drawdown()) * 100
            
            trades = pf.trades.records_readable
            if len(trades) > 0:
                winning_trades = trades[trades['PnL'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100
                
                if len(winning_trades) > 0 and len(trades[trades['PnL'] < 0]) > 0:
                    avg_win = winning_trades['PnL'].mean()
                    avg_loss = abs(trades[trades['PnL'] < 0]['PnL'].mean())
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                else:
                    profit_factor = 0
            else:
                win_rate = 0
                profit_factor = 0
            
            returns = pf.returns()
            volatility = returns.std() * np.sqrt(252)
            
            years = (df.index[-1] - df.index[0]).days / 365.25
            
            results = {
                'total_return': total_return,
                'annualized_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades),
                'data_period': f"{df.index[0].date()} to {df.index[-1].date()}",
                'data_points': len(df),
                'years': years,
                'category': self.instruments[symbol]['category']
            }
            
            logger.info(f"✅ Backtest completed for {symbol}: {annual_return:.1f}% annual return")
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {e}")
            return {}
    
    def run_test(self):
        """Run mini test"""
        logger.info("🚀 STARTING MINI TEST (3 symbols)")
        logger.info("=" * 60)
        
        for symbol in self.instruments:
            try:
                logger.info(f"📊 Processing {symbol}...")
                
                # Download data
                df = self.download_yfinance_robust(symbol, years=10)
                
                if df.empty:
                    logger.warning(f"⚠️ Skipping {symbol} - no data")
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df, symbol)
                
                if df.empty or len(df) < 500:
                    logger.warning(f"⚠️ Skipping {symbol} - insufficient processed data")
                    continue
                
                # Generate signals
                df = self.generate_signals(df, symbol)
                
                # Run backtest
                results = self.run_backtest(df, symbol)
                
                if results:
                    self.data[symbol] = df
                    self.results[symbol] = results
                    logger.info(f"✅ Successfully processed {symbol}")
                else:
                    logger.warning(f"⚠️ No results for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
        
        # Display results
        if self.results:
            print("\n" + "=" * 80)
            print("🎯 MINI TEST RESULTS")
            print("=" * 80)
            
            for symbol, results in self.results.items():
                print(f"\n**{symbol} - {self.instruments[symbol]['category']}**")
                print("```")
                print(f"Total Return      :  {results['total_return']:8.2f}%")
                print(f"Annualized Return :  {results['annualized_return']:8.2f}%")
                print(f"Sharpe Ratio      :  {results['sharpe_ratio']:8.3f}")
                print(f"Max Drawdown      :  {results['max_drawdown']:8.2f}%")
                print(f"Win Rate          :  {results['win_rate']:8.2f}%")
                print(f"Total Trades      :  {results['total_trades']:8.0f}")
                print("```")
                print(f"Data: {results['data_period']} ({results['years']:.1f} years)")
            
            print(f"\n✅ MINI TEST COMPLETED: {len(self.results)}/{len(self.instruments)} successful")
        else:
            print("❌ MINI TEST FAILED - No successful backtests")

def main():
    backtest = MiniTestBacktest()
    backtest.run_test()

if __name__ == "__main__":
    main()