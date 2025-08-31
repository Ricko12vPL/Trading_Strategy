#!/usr/bin/env python3
"""
IBKR 15-YEAR HISTORICAL DATA BACKTEST
====================================
Download 15+ years of data from IBKR API and re-run complete backtest analysis
for all 7 instruments in the institutional combined strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple
import vectorbt as vbt
import quantstats as qs
from ib_insync import *
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IBKR15YearBacktest:
    """Complete 15-year backtest using IBKR API data"""
    
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
            self.ib.connect('127.0.0.1', 7497, clientId=20)
            logger.info("✅ Successfully connected to IBKR")
            return True
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            return False
    
    def disconnect_from_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IBKR")
    
    def download_historical_data(self, symbol: str, years: int = 15) -> pd.DataFrame:
        """Download historical data from IBKR"""
        try:
            logger.info(f"📊 Downloading {years} years of data for {symbol}...")
            
            # Create contract
            if symbol in ['XAR', 'EWG', 'XLF', 'FXI']:  # ETFs
                contract = Stock(symbol, 'SMART', 'USD')
            else:  # Individual stocks
                contract = Stock(symbol, 'SMART', 'USD')
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365 + 100)  # Add buffer
            
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
                logger.warning(f"⚠️ No data received for {symbol}")
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
            
            logger.info(f"✅ Downloaded {len(df)} days of data for {symbol} from {df.index[0].date()} to {df.index[-1].date()}")
            
            # Add small delay to avoid rate limits
            time.sleep(1)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators for a given symbol"""
        try:
            params = self.strategy_params[symbol]
            
            # RSI
            df['RSI'] = vbt.RSI.run(df['Close'], window=params['rsi_period']).rsi
            
            # Bollinger Bands
            bb = vbt.BBANDS.run(df['Close'], window=params['bb_period'], std=params['bb_std'])
            df['BB_Upper'] = bb.upper
            df['BB_Middle'] = bb.middle
            df['BB_Lower'] = bb.lower
            
            # Volume moving average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Momentum
            df['Momentum'] = df['Close'].pct_change(periods=params['mom_period']) * 100
            df['Momentum_MA'] = df['Momentum'].rolling(window=10).mean()
            
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
            
            for i in range(1, len(df)):
                signals = []
                
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
                    if signal_sum > 0:
                        df['Signal'].iloc[i] = 1
                    elif signal_sum < 0:
                        df['Signal'].iloc[i] = -1
                    
                    # Signal strength (confidence)
                    df['Signal_Strength'].iloc[i] = min(abs(signal_sum) / len(signals), 1.0)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals for {symbol}: {e}")
            return df
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run backtest for a single instrument"""
        try:
            logger.info(f"🔄 Running backtest for {symbol}...")
            
            # Generate entries and exits
            entries = df['Signal'] == 1
            exits = df['Signal'] == -1
            
            # Run portfolio simulation
            pf = vbt.Portfolio.from_signals(
                df['Close'],
                entries,
                exits,
                init_cash=100000,
                fees=0.001,  # 0.1% fees
                slippage=0.0008  # 8bps slippage
            )
            
            # Calculate metrics
            total_return = pf.total_return() * 100
            annual_return = pf.annualized_return() * 100
            sharpe_ratio = pf.sharpe_ratio()
            max_drawdown = abs(pf.max_drawdown()) * 100
            
            # Additional metrics
            trades = pf.trades.records
            if len(trades) > 0:
                win_rate = (trades['pnl'] > 0).mean() * 100
                avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0
                avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if len(trades[trades['pnl'] < 0]) > 0 else 1
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
            
            # Calculate volatility
            returns = pf.returns()
            volatility = returns.std() * np.sqrt(252)
            
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
                'data_points': len(df)
            }
            
            logger.info(f"✅ Backtest completed for {symbol}: {annual_return:.1f}% annual return")
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {e}")
            return {}
    
    def run_complete_analysis(self):
        """Run complete 15-year analysis for all instruments"""
        logger.info("🚀 STARTING 15-YEAR IBKR BACKTEST ANALYSIS")
        logger.info("=" * 80)
        
        # Connect to IBKR
        if not self.connect_to_ibkr():
            logger.error("❌ Cannot proceed without IBKR connection")
            return
        
        try:
            # Download data for all instruments
            for symbol in self.instruments:
                logger.info(f"📊 Processing {symbol} ({self.instruments[symbol]['name']})...")
                
                # Download historical data
                df = self.download_historical_data(symbol, years=15)
                
                if df.empty:
                    logger.warning(f"⚠️ Skipping {symbol} - no data available")
                    continue
                
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df, symbol)
                
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
            self.calculate_combined_strategy()
            
            # Display results
            self.display_results()
            
        finally:
            self.disconnect_from_ibkr()
    
    def calculate_combined_strategy(self):
        """Calculate combined strategy performance"""
        if not self.results:
            return
        
        logger.info("📊 Calculating combined strategy performance...")
        
        # Portfolio allocation
        tier2_weight = 0.65 / 5  # 13% each for 5 TIER2 instruments
        tier3_weight = 0.15 / 2  # 7.5% each for 2 TIER3 instruments
        cash_weight = 0.20
        cash_return = 4.5  # 4.5% annual cash return
        
        # Calculate weighted returns
        weighted_return = 0
        weighted_volatility = 0
        weighted_sharpe = 0
        weighted_drawdown = 0
        weighted_win_rate = 0
        weighted_profit_factor = 0
        
        total_weight = 0
        
        for symbol, results in self.results.items():
            if self.instruments[symbol]['tier'] == 2:
                weight = tier2_weight
            else:  # tier 3
                weight = tier3_weight
            
            weighted_return += results['annualized_return'] * weight
            weighted_volatility += results['volatility'] * weight
            weighted_sharpe += results['sharpe_ratio'] * weight
            weighted_drawdown += results['max_drawdown'] * weight
            weighted_win_rate += results['win_rate'] * weight
            weighted_profit_factor += results['profit_factor'] * weight
            total_weight += weight
        
        # Add cash component
        weighted_return += cash_return * cash_weight
        
        # Calculate total return (assuming same time period)
        years = 15
        total_return = ((1 + weighted_return/100) ** years - 1) * 100
        
        # Store combined results
        self.results['COMBINED'] = {
            'total_return': total_return,
            'annualized_return': weighted_return,
            'volatility': weighted_volatility,
            'sharpe_ratio': weighted_sharpe,
            'max_drawdown': weighted_drawdown,
            'win_rate': weighted_win_rate,
            'profit_factor': weighted_profit_factor,
            'allocation': f"TIER2: {0.65*100:.0f}%, TIER3: {0.15*100:.0f}%, Cash: {0.20*100:.0f}%"
        }
    
    def display_results(self):
        """Display results in requested format"""
        logger.info("📊 DISPLAYING FINAL RESULTS")
        logger.info("=" * 80)
        
        print("\n" + "=" * 100)
        print("🏛️ IBKR 15-YEAR BACKTEST RESULTS - INSTITUTIONAL COMBINED STRATEGY")
        print("=" * 100)
        print(f"Data Period: 15 years ending {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Data Source: Interactive Brokers API")
        print(f"Total Instruments: {len(self.results) - 1} individual + 1 combined strategy")
        print("=" * 100)
        
        # Individual instrument results
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


def main():
    """Main execution function"""
    backtest = IBKR15YearBacktest()
    backtest.run_complete_analysis()


if __name__ == "__main__":
    main()