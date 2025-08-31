#!/usr/bin/env python3
"""
TEST SINGLE INSTRUMENT - NASDAQ TOP 50 ENHANCED
===============================================
Test single instrument to debug the issue
"""

import yfinance as yf
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Basic test
def test_single_instrument():
    """Test single instrument AAPL"""
    print("🧪 TESTING SINGLE INSTRUMENT: AAPL")
    print("="*60)
    
    try:
        # Get data
        ticker = yf.Ticker('AAPL')
        data = ticker.history(period="2y", interval="1d")
        print(f"✅ Data retrieved: {len(data)} records")
        
        # Basic processing
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
        print("✅ Basic indicators calculated")
        
        # Simple signals
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = (data['returns'].shift(1) > 0.01).astype(int)
        signals['sell_signal'] = (data['returns'].shift(1) < -0.01).astype(int)
        print(f"✅ Signals generated - Buy: {signals['buy_signal'].sum()}, Sell: {signals['sell_signal'].sum()}")
        
        # Simple backtest
        portfolio = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=signals['buy_signal'].astype(bool),
            exits=signals['sell_signal'].astype(bool),
            size=2500,
            fees=0.001,
            init_cash=100000,
            freq='D'
        )
        print("✅ VectorBT backtest completed")
        
        # Basic metrics
        total_return = (portfolio.value().iloc[-1] / portfolio.value().iloc[0]) - 1
        returns = portfolio.returns()
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        print("✅ Metrics calculated:")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Annual Return: {annual_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Quick Monte Carlo test (100 permutations)
        print("🧪 Running quick Monte Carlo test...")
        mc_returns = []
        for i in range(100):
            try:
                shuffled_returns = np.random.permutation(data['returns'].dropna().values)
                mc_return = np.prod(1 + shuffled_returns) - 1
                mc_returns.append(mc_return)
            except:
                mc_returns.append(0)
        
        better_count = sum(1 for ret in mc_returns if ret >= total_return)
        p_value = (better_count + 1) / 101
        
        print(f"✅ Monte Carlo completed:")
        print(f"   P-Value: {p_value:.4f}")
        print(f"   Statistically Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        print("="*60)
        print("🎉 SINGLE INSTRUMENT TEST SUCCESSFUL!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_single_instrument()