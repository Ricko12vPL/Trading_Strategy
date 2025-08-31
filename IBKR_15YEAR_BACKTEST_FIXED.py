#!/usr/bin/env python3
"""
IBKR 15-YEAR HISTORICAL DATA BACKTEST - FIXED VERSION
=====================================================
Download 15+ years of data from IBKR API and re-run complete backtest analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ib_insync import *
import talib
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IBKR15YearBacktest:
    """Complete 15-year backtest using IBKR API data with fixed indicators"""
    
    def __init__(self):
        self.ib = None
        self.instruments = {
            # TIER 2 - High Confidence
            'XAR': {'name': 'iShares U.S. Aerospace & Defense ETF', 'tier': 2},
            'EWG': {'name': 'iShares MSCI Germany ETF', 'tier': 2}, 
            'XLF': {'name': 'Financial Select Sector SPDR Fund', 'tier': 2},
            'TSLA': {'name': 'Tesla Inc', 'tier': 2},
            'FXI': {'name': 'iShares China Large-Cap ETF', 'tier': 2},
            
            # TIER 3 - Speculative  
            'AVGO': {'name': 'Broadcom Inc', 'tier': 3},
            'NVDA': {'name': 'NVIDIA Corporation', 'tier': 3}
        }
        
        self.data = {}
        self.results = {}
        
        # Strategy parameters (optimized from Phase 2)
        self.strategy_params = {
            'XAR': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'bb_period': 20, 'bb_std': 2.0, 'vol_threshold': 1.5, 'mom_period': 5},
            'EWG': {'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75, 'bb_period': 20, 'bb_std': 2.0, 'vol_threshold': 1.8, 'mom_period': 5},
            'XLF': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'bb_period': 20, 'bb_std': 2.0, 'vol_threshold': 1.5, 'mom_period': 5},
            'TSLA': {'rsi_period': 14, 'rsi_oversold': 20, 'rsi_overbought': 80, 'bb_period': 15, 'bb_std': 2.5, 'vol_threshold': 2.0, 'mom_period': 3},
            'FXI': {'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75, 'bb_period': 20, 'bb_std': 2.0, 'vol_threshold': 1.6, 'mom_period': 5},
            'AVGO': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'bb_period': 20, 'bb_std': 2.0, 'vol_threshold': 1.4, 'mom_period': 5},
            'NVDA': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'bb_period': 15, 'bb_std': 2.5, 'vol_threshold': 2.2, 'mom_period': 3}
        }
        
    def connect_to_ibkr(self) -> bool:
        """Connect to IBKR API"""
        try:
            logger.info("🔌 Connecting to IBKR API...")
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=21)
            logger.info("✅ Successfully connected to IBKR")
            return True
        except Exception as e:
            logger.warning(f"⚠️ IBKR connection failed, will use yfinance fallback: {e}")
            return False
    
    def disconnect_from_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IBKR")
    
    def download_historical_data_ibkr(self, symbol: str, years: int = 15) -> pd.DataFrame:
        """Download historical data from IBKR"""
        try:
            logger.info(f"📊 Downloading {years} years of data for {symbol} from IBKR...")
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{years} Y',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"⚠️ No IBKR data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            logger.info(f"✅ Downloaded {len(df)} days of IBKR data for {symbol} from {df.index[0].date()} to {df.index[-1].date()}")
            time.sleep(1)  # Rate limiting
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ IBKR download failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_historical_data_yfinance(self, symbol: str, years: int = 15) -> pd.DataFrame:
        """Download historical data from yfinance as fallback"""
        try:
            logger.info(f"📊 Downloading {years} years of data for {symbol} from yfinance...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365 + 100)
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"⚠️ No yfinance data for {symbol}")
                return pd.DataFrame()
            
            # Ensure proper column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            logger.info(f"✅ Downloaded {len(df)} days of yfinance data for {symbol} from {df.index[0].date()} to {df.index[-1].date()}")
            return df
            
        except Exception as e:
            logger.error(f"❌ yfinance download failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_historical_data(self, symbol: str, years: int = 15) -> pd.DataFrame:
        """Download historical data with IBKR primary, yfinance fallback"""
        
        # Try IBKR first if connected
        if self.ib and self.ib.isConnected():
            df = self.download_historical_data_ibkr(symbol, years)
            if not df.empty:
                return df
        
        # Fallback to yfinance
        return self.download_historical_data_yfinance(symbol, years)
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators using talib"""
        try:
            params = self.strategy_params[symbol]
            
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} days")
                return df
            
            # Convert to numpy arrays for talib
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            # RSI
            rsi = talib.RSI(close, timeperiod=params['rsi_period'])
            df['RSI'] = rsi
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                       timeperiod=params['bb_period'], 
                                                       nbdevup=params['bb_std'],
                                                       nbdevdn=params['bb_std'])
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle  
            df['BB_Lower'] = bb_lower
            
            # Volume moving average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Momentum
            df['Momentum'] = df['Close'].pct_change(periods=params['mom_period']) * 100
            df['Momentum_MA'] = df['Momentum'].rolling(window=10).mean()
            
            # Clean up NaN values
            df = df.dropna()
            
            logger.info(f"✅ Calculated indicators for {symbol}, {len(df)} valid data points")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate indicators for {symbol}: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate trading signals based on technical indicators"""
        try:
            params = self.strategy_params[symbol]
            
            # Initialize signals
            df['Signal'] = 0
            df['Signal_Strength'] = 0.0
            df['Position'] = 0  # Track position (1=long, 0=flat, -1=short)
            
            current_position = 0
            
            for i in range(1, len(df)):
                signals = []
                
                # Skip if we have NaN values
                if (pd.isna(df['RSI'].iloc[i]) or pd.isna(df['BB_Lower'].iloc[i]) or 
                    pd.isna(df['Volume_Ratio'].iloc[i]) or pd.isna(df['Momentum'].iloc[i])):
                    df['Position'].iloc[i] = current_position
                    continue
                
                # RSI Signals
                if df['RSI'].iloc[i] < params['rsi_oversold']:
                    signals.append(1)  # Buy
                elif df['RSI'].iloc[i] > params['rsi_overbought']:
                    signals.append(-1)  # Sell
                
                # Bollinger Bands Signals
                if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
                    signals.append(1)  # Buy (oversold)
                elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
                    signals.append(-1)  # Sell (overbought)
                
                # Volume Confirmation
                if df['Volume_Ratio'].iloc[i] > params['vol_threshold']:
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        signals.append(1)  # Volume + price up
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        signals.append(-1)  # Volume + price down
                
                # Momentum Signals
                if (df['Momentum'].iloc[i] > 1.5 and 
                    df['Momentum'].iloc[i] > df['Momentum_MA'].iloc[i]):
                    signals.append(1)  # Strong positive momentum
                elif (df['Momentum'].iloc[i] < -1.5 and 
                      df['Momentum'].iloc[i] < df['Momentum_MA'].iloc[i]):
                    signals.append(-1)  # Strong negative momentum
                
                # Combine signals
                if len(signals) > 0:
                    signal_sum = sum(signals)
                    if signal_sum > 0 and current_position == 0:
                        df['Signal'].iloc[i] = 1  # Enter long
                        current_position = 1
                    elif signal_sum < 0 and current_position == 1:
                        df['Signal'].iloc[i] = -1  # Exit long
                        current_position = 0
                    
                    # Signal strength (confidence)
                    df['Signal_Strength'].iloc[i] = min(abs(signal_sum) / len(signals), 1.0)
                
                df['Position'].iloc[i] = current_position
            
            # Create entry/exit signals
            df['Entry'] = (df['Signal'] == 1).astype(int)
            df['Exit'] = (df['Signal'] == -1).astype(int)
            
            logger.info(f"✅ Generated signals for {symbol}: {df['Entry'].sum()} entries, {df['Exit'].sum()} exits")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals for {symbol}: {e}")
            return df
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run backtest for a single instrument using simple vectorized approach"""
        try:
            logger.info(f"🔄 Running backtest for {symbol}...")
            
            if 'Entry' not in df.columns or 'Exit' not in df.columns:
                logger.error(f"❌ No signals available for {symbol}")
                return {}
            
            # Initialize portfolio tracking
            position = 0
            cash = 100000
            portfolio_value = []
            trades = []
            
            initial_cash = cash
            
            for i in range(len(df)):
                price = df['Close'].iloc[i]
                
                # Entry signal
                if df['Entry'].iloc[i] == 1 and position == 0:
                    # Buy with available cash (minus fees)
                    fees = cash * 0.001  # 0.1% fees
                    position = (cash - fees) / price
                    cash = 0
                    trades.append({'type': 'BUY', 'price': price, 'date': df.index[i]})
                
                # Exit signal  
                elif df['Exit'].iloc[i] == 1 and position > 0:
                    # Sell position
                    cash = position * price
                    fees = cash * 0.001
                    cash -= fees
                    trades.append({'type': 'SELL', 'price': price, 'date': df.index[i], 'pnl': cash - initial_cash})
                    position = 0
                
                # Calculate portfolio value
                if position > 0:
                    portfolio_value.append(position * price)
                else:
                    portfolio_value.append(cash)
            
            # Close final position if still open
            if position > 0:
                final_price = df['Close'].iloc[-1]
                cash = position * price * 0.999  # Include slippage
                trades.append({'type': 'SELL', 'price': final_price, 'date': df.index[-1], 'pnl': cash - initial_cash})
            
            # Calculate metrics
            df['Portfolio_Value'] = portfolio_value
            returns = pd.Series(portfolio_value).pct_change().dropna()
            
            # Performance metrics
            total_return = (portfolio_value[-1] / initial_cash - 1) * 100
            
            # Annualized return
            years = (df.index[-1] - df.index[0]).days / 365.25
            annual_return = ((portfolio_value[-1] / initial_cash) ** (1/years) - 1) * 100
            
            # Volatility and Sharpe
            volatility = returns.std() * np.sqrt(252)
            risk_free_rate = 0.045  # 4.5%
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Drawdown
            peak = np.maximum.accumulate(portfolio_value)
            drawdown = (np.array(portfolio_value) / peak - 1) * 100
            max_drawdown = abs(drawdown.min())
            
            # Trade metrics
            if len(trades) > 2:
                winning_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
                total_trades = len([t for t in trades if 'pnl' in t])
                win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
                
                if len(winning_trades) > 0:
                    avg_win = np.mean([t['pnl'] for t in winning_trades])
                    losing_trades = [t for t in trades if 'pnl' in t and t['pnl'] < 0]
                    avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                else:
                    profit_factor = 0
            else:
                win_rate = 0
                profit_factor = 0
                total_trades = 0
            
            results = {
                'total_return': total_return,
                'annualized_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'data_period': f"{df.index[0].date()} to {df.index[-1].date()}",
                'data_points': len(df),
                'years': years
            }
            
            logger.info(f"✅ Backtest completed for {symbol}: {annual_return:.1f}% annual return, {total_trades} trades")
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {e}")
            return {}
    
    def run_complete_analysis(self):
        """Run complete 15-year analysis for all instruments"""
        logger.info("🚀 STARTING 15-YEAR IBKR BACKTEST ANALYSIS")
        logger.info("=" * 80)
        
        # Connect to IBKR (optional - will fallback to yfinance)
        self.connect_to_ibkr()
        
        try:
            # Process each instrument
            for symbol in self.instruments:
                logger.info(f"📊 Processing {symbol} ({self.instruments[symbol]['name']})...")
                
                # Download historical data
                df = self.download_historical_data(symbol, years=15)
                
                if df.empty:
                    logger.warning(f"⚠️ Skipping {symbol} - no data available")
                    continue
                
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df, symbol)
                
                if df.empty:
                    logger.warning(f"⚠️ Skipping {symbol} - no valid indicators")
                    continue
                
                # Generate signals
                df = self.generate_signals(df, symbol)
                
                # Run backtest
                results = self.run_backtest(df, symbol)
                
                if results:
                    self.data[symbol] = df
                    self.results[symbol] = results
                else:
                    logger.warning(f"⚠️ No results for {symbol}")
            
            # Calculate combined strategy results
            if self.results:
                self.calculate_combined_strategy()
                
                # Display results
                self.display_results()
            else:
                logger.error("❌ No successful backtests - cannot display results")
            
        finally:
            self.disconnect_from_ibkr()
    
    def calculate_combined_strategy(self):
        """Calculate combined strategy performance"""
        if not self.results:
            return
        
        logger.info("📊 Calculating combined strategy performance...")
        
        # Portfolio allocation
        tier2_weight = 0.65 / sum(1 for inst in self.instruments.values() if inst['tier'] == 2)
        tier3_weight = 0.15 / sum(1 for inst in self.instruments.values() if inst['tier'] == 3)
        cash_weight = 0.20
        cash_return = 4.5  # 4.5% annual cash return
        
        # Calculate weighted metrics
        total_weight = 0
        weighted_metrics = {
            'total_return': 0, 'annualized_return': 0, 'volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0, 'profit_factor': 0
        }
        
        for symbol, results in self.results.items():
            tier = self.instruments[symbol]['tier']
            weight = tier2_weight if tier == 2 else tier3_weight
            
            for metric in weighted_metrics:
                weighted_metrics[metric] += results[metric] * weight
            
            total_weight += weight
        
        # Add cash component to returns
        weighted_metrics['annualized_return'] += cash_return * cash_weight
        
        # Calculate combined total return using average years
        avg_years = np.mean([self.results[sym]['years'] for sym in self.results])
        combined_total_return = ((1 + weighted_metrics['annualized_return']/100) ** avg_years - 1) * 100
        weighted_metrics['total_return'] = combined_total_return
        
        # Store combined results
        self.results['COMBINED'] = {
            **weighted_metrics,
            'allocation': f"TIER2: {0.65*100:.0f}%, TIER3: {0.15*100:.0f}%, Cash: {0.20*100:.0f}%",
            'years': avg_years
        }
        
        logger.info(f"✅ Combined strategy: {weighted_metrics['annualized_return']:.1f}% annual return")
    
    def display_results(self):
        """Display results in requested format"""
        logger.info("📊 DISPLAYING FINAL RESULTS")
        logger.info("=" * 80)
        
        print("\n" + "=" * 100)
        print("🏛️ IBKR 15-YEAR BACKTEST RESULTS - INSTITUTIONAL COMBINED STRATEGY")
        print("=" * 100)
        print(f"Data Period: Up to 15 years ending {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Data Source: Interactive Brokers API + yfinance fallback")
        print(f"Total Instruments Analyzed: {len(self.results) - (1 if 'COMBINED' in self.results else 0)} individual instruments")
        print("=" * 100)
        
        # Individual instrument results
        if len(self.results) > 1 or 'COMBINED' not in self.results:
            print("\n🎯 INDIVIDUAL INSTRUMENT RESULTS:")
            print("-" * 50)
            
            for symbol in ['XAR', 'EWG', 'XLF', 'TSLA', 'FXI', 'AVGO', 'NVDA']:
                if symbol in self.results:
                    results = self.results[symbol]
                    tier_name = "TIER 2" if self.instruments[symbol]['tier'] == 2 else "TIER 3"
                    
                    print(f"\n**{symbol} - {self.instruments[symbol]['name']} ({tier_name})**")
                    print("```")
                    print(f"📊 {symbol} PERFORMANCE SUMMARY")
                    print("=" * 80)
                    print(f"Metric               |           {symbol}")
                    print("-------------------- | -------------")
                    print(f"total_return         |     [{results['total_return']:6.2f}%]")
                    print(f"annualized_return    |     [{results['annualized_return']:6.2f}%]")
                    print(f"volatility           |      [{results['volatility']:6.3f}]")
                    print(f"sharpe_ratio         |      [{results['sharpe_ratio']:6.3f}]")
                    print(f"max_drawdown         |     [{results['max_drawdown']:6.2f}%]")
                    print(f"win_rate             |     [{results['win_rate']:6.2f}%]")
                    print(f"profit_factor        |       [{results['profit_factor']:5.2f}]")
                    print("```")
                    print(f"Data: {results['data_period']} ({results['years']:.1f} years)")
        
        # Combined strategy results
        if 'COMBINED' in self.results:
            results = self.results['COMBINED']
            print(f"\n🏆 COMBINED STRATEGY RESULTS:")
            print("-" * 40)
            print("\n**Portfolio-Weighted Combined Strategy (65% TIER2 + 15% TIER3 + 20% Cash)**")
            print("```")
            print("📊 COMBINED STRATEGY PERFORMANCE SUMMARY")
            print("=" * 80)
            print("Metric               |    COMBINED")
            print("-------------------- | -------------")
            print(f"total_return         |     [{results['total_return']:6.2f}%]")
            print(f"annualized_return    |     [{results['annualized_return']:6.2f}%]")
            print(f"volatility           |      [{results['volatility']:6.3f}]")
            print(f"sharpe_ratio         |      [{results['sharpe_ratio']:6.3f}]")
            print(f"max_drawdown         |     [{results['max_drawdown']:6.2f}%]")
            print(f"win_rate             |     [{results['win_rate']:6.2f}%]")
            print(f"profit_factor        |       [{results['profit_factor']:5.2f}]")
            print("```")
            print(f"\n**Allocation Details:** {results['allocation']}")
        
        print("\n" + "=" * 100)
        print("✅ 15-YEAR IBKR BACKTEST ANALYSIS COMPLETED")
        print("=" * 100)
        
        print(f"\n📋 SUMMARY: Successfully analyzed {len(self.results) - (1 if 'COMBINED' in self.results else 0)} instruments")
        if 'COMBINED' in self.results:
            print(f"🎯 Combined Strategy: {self.results['COMBINED']['annualized_return']:.1f}% annual return")


def main():
    """Main execution function"""
    backtest = IBKR15YearBacktest()
    backtest.run_complete_analysis()


if __name__ == "__main__":
    main()