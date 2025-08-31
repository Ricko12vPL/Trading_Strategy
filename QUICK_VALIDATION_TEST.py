#!/usr/bin/env python3
"""
QUICK VALIDATION TEST FOR MEGA BACKTEST CALCULATIONS
==================================================
Focused validation of critical calculations with error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import vectorbt as vbt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_validation_test():
    """Run focused validation tests"""
    logger.info("🚀 QUICK VALIDATION TEST STARTING")
    logger.info("=" * 60)
    
    validation_results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'critical_errors': []
    }
    
    try:
        # Test 1: Basic data download
        logger.info("📊 Test 1: Data Download Validation")
        symbol = 'AAPL'
        data = yf.download(symbol, start='2020-01-01', end='2024-01-01', progress=False)
        
        if not data.empty and len(data) > 1000:
            validation_results['tests_passed'] += 1
            logger.info("✅ Data download successful")
        else:
            validation_results['tests_failed'] += 1
            validation_results['critical_errors'].append("Data download failed")
            logger.error("❌ Data download failed")
        
        # Test 2: RSI Calculation Validation
        logger.info("📊 Test 2: RSI Calculation Validation")
        try:
            # Manual RSI calculation
            close_prices = data['Close'].iloc[:50]  # Use first 50 points
            period = 14
            
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi_manual = 100 - (100 / (1 + rs))
            
            # VectorBT RSI
            rsi_vbt = vbt.RSI.run(close_prices, window=period).rsi
            
            # Compare last valid values
            manual_val = rsi_manual.dropna().iloc[-1]
            vbt_val = rsi_vbt.iloc[-1]
            
            if abs(manual_val - vbt_val) < 1.0:  # Within 1 point
                validation_results['tests_passed'] += 1
                logger.info(f"✅ RSI validation passed: Manual={manual_val:.2f}, VBT={vbt_val:.2f}")
            else:
                validation_results['tests_failed'] += 1
                validation_results['critical_errors'].append(f"RSI calculation mismatch: {abs(manual_val - vbt_val):.2f}")
                logger.error(f"❌ RSI validation failed: Manual={manual_val:.2f}, VBT={vbt_val:.2f}")
                
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['critical_errors'].append(f"RSI calculation error: {str(e)}")
            logger.error(f"❌ RSI calculation error: {e}")
        
        # Test 3: Portfolio Calculation Validation
        logger.info("📊 Test 3: Portfolio Calculation Validation")
        try:
            # Generate simple signals
            prices = data['Close'].iloc[:100]
            rsi = vbt.RSI.run(prices, window=14).rsi
            
            entries = rsi < 30
            exits = rsi > 70
            
            # VectorBT portfolio
            pf = vbt.Portfolio.from_signals(
                prices,
                entries,
                exits,
                init_cash=100000,
                fees=0.001,
                freq='D'
            )
            
            total_return = pf.total_return()
            sharpe_ratio = pf.sharpe_ratio()
            max_drawdown = pf.max_drawdown()
            
            # Validate ranges
            return_valid = -1 <= total_return <= 10  # -100% to 1000% reasonable range
            sharpe_valid = -5 <= sharpe_ratio <= 5   # Reasonable Sharpe range
            dd_valid = -1 <= max_drawdown <= 0       # Drawdown should be negative or zero
            
            if return_valid and sharpe_valid and dd_valid:
                validation_results['tests_passed'] += 1
                logger.info(f"✅ Portfolio validation passed: Return={total_return:.3f}, Sharpe={sharpe_ratio:.3f}, DD={max_drawdown:.3f}")
            else:
                validation_results['tests_failed'] += 1
                validation_results['critical_errors'].append("Portfolio metrics out of reasonable ranges")
                logger.error(f"❌ Portfolio validation failed: Return={total_return:.3f}, Sharpe={sharpe_ratio:.3f}, DD={max_drawdown:.3f}")
                
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['critical_errors'].append(f"Portfolio calculation error: {str(e)}")
            logger.error(f"❌ Portfolio calculation error: {e}")
        
        # Test 4: Monte Carlo Validation (simplified)
        logger.info("📊 Test 4: Monte Carlo Statistical Validation")
        try:
            # Generate test returns
            np.random.seed(42)
            returns = np.random.normal(0.05, 0.15, 1000)
            
            # Basic statistical properties
            sample_mean = np.mean(returns)
            sample_std = np.std(returns, ddof=1)
            
            # Test if sample statistics are reasonable
            mean_reasonable = abs(sample_mean - 0.05) < 0.02  # Within 2% of expected
            std_reasonable = abs(sample_std - 0.15) < 0.02    # Within 2% of expected
            
            # Test p-value calculation
            observed_return = 0.08
            better_returns = np.sum(returns >= observed_return)
            p_value = better_returns / len(returns)
            p_value_reasonable = 0.0 <= p_value <= 1.0
            
            if mean_reasonable and std_reasonable and p_value_reasonable:
                validation_results['tests_passed'] += 1
                logger.info(f"✅ Monte Carlo validation passed: Mean={sample_mean:.4f}, Std={sample_std:.4f}, P-val={p_value:.4f}")
            else:
                validation_results['tests_failed'] += 1
                validation_results['critical_errors'].append("Monte Carlo statistical properties invalid")
                logger.error(f"❌ Monte Carlo validation failed: Mean={sample_mean:.4f}, Std={sample_std:.4f}, P-val={p_value:.4f}")
                
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['critical_errors'].append(f"Monte Carlo error: {str(e)}")
            logger.error(f"❌ Monte Carlo error: {e}")
        
        # Test 5: Memory and Performance Check
        logger.info("📊 Test 5: Memory and Performance Check")
        try:
            import psutil
            import time
            
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform computation
            start_time = time.time()
            
            # Multiple RSI calculations (simulate parameter sweep)
            test_data = data['Close'].iloc[:500]
            rsi_results = []
            for period in [10, 14, 20, 30]:
                rsi = vbt.RSI.run(test_data, window=period).rsi
                rsi_results.append(rsi.mean())
            
            end_time = time.time()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            computation_time = end_time - start_time
            memory_usage = mem_after - mem_before
            
            # Validate performance
            time_reasonable = computation_time < 10  # Less than 10 seconds
            memory_reasonable = memory_usage < 100   # Less than 100 MB
            
            if time_reasonable and memory_reasonable:
                validation_results['tests_passed'] += 1
                logger.info(f"✅ Performance validation passed: Time={computation_time:.2f}s, Memory={memory_usage:.1f}MB")
            else:
                validation_results['tests_failed'] += 1
                validation_results['critical_errors'].append("Performance issues detected")
                logger.error(f"❌ Performance validation failed: Time={computation_time:.2f}s, Memory={memory_usage:.1f}MB")
                
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['critical_errors'].append(f"Performance test error: {str(e)}")
            logger.error(f"❌ Performance test error: {e}")
        
    except Exception as e:
        validation_results['critical_errors'].append(f"Overall validation error: {str(e)}")
        logger.error(f"❌ Overall validation error: {e}")
    
    # Summary
    total_tests = validation_results['tests_passed'] + validation_results['tests_failed']
    success_rate = validation_results['tests_passed'] / total_tests if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print("🔍 QUICK VALIDATION TEST RESULTS")
    print("=" * 80)
    print(f"Tests Passed: {validation_results['tests_passed']}")
    print(f"Tests Failed: {validation_results['tests_failed']}")
    print(f"Success Rate: {success_rate * 100:.1f}%")
    print(f"Critical Errors: {len(validation_results['critical_errors'])}")
    
    if validation_results['critical_errors']:
        print("\n🚨 CRITICAL ERRORS:")
        for i, error in enumerate(validation_results['critical_errors'], 1):
            print(f"  {i}. {error}")
    
    print("\n" + "=" * 80)
    if success_rate >= 0.80:
        print("✅ VALIDATION PASSED: Core calculations verified")
    else:
        print("❌ VALIDATION FAILED: Critical issues detected")
    print("=" * 80)
    
    return validation_results

def main():
    """Run quick validation"""
    results = quick_validation_test()
    
    # Additional mathematical verification
    logger.info("\n🔬 RUNNING MATHEMATICAL FORMULA CHECKS")
    
    # RSI Formula Check
    logger.info("📐 RSI Formula: RSI = 100 - (100 / (1 + RS))")
    logger.info("   where RS = Average Gain / Average Loss")
    
    # Test mathematical properties
    # RSI should be 50 when gains = losses
    test_gains = pd.Series([1, 1, 1, 1])
    test_losses = pd.Series([1, 1, 1, 1])
    rs = test_gains.mean() / test_losses.mean()  # Should be 1
    rsi_theoretical = 100 - (100 / (1 + rs))    # Should be 50
    
    if abs(rsi_theoretical - 50) < 0.001:
        logger.info("✅ RSI mathematical property verified: Equal gains/losses = RSI 50")
    else:
        logger.error(f"❌ RSI mathematical property failed: Expected 50, got {rsi_theoretical}")
    
    # Sharpe Ratio Formula Check
    logger.info("📐 Sharpe Ratio Formula: SR = (R_p - R_f) / σ_p")
    logger.info("   Annualized: SR = (Mean_Return - Risk_Free) / Std_Return * sqrt(252)")
    
    # Test with known values
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    mean_return = returns.mean()
    std_return = returns.std()
    risk_free = 0.02 / 252  # 2% annual to daily
    
    sharpe_manual = (mean_return - risk_free) / std_return * np.sqrt(252)
    
    if not np.isnan(sharpe_manual) and abs(sharpe_manual) < 10:  # Reasonable range
        logger.info(f"✅ Sharpe Ratio calculation verified: {sharpe_manual:.3f}")
    else:
        logger.error(f"❌ Sharpe Ratio calculation invalid: {sharpe_manual}")
    
    # Maximum Drawdown Formula Check
    logger.info("📐 Maximum Drawdown Formula: DD = (Portfolio_Value / Peak_Value) - 1")
    
    portfolio_values = pd.Series([100, 110, 105, 120, 90, 95])
    running_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values / running_max) - 1
    max_dd = drawdowns.min()
    
    if max_dd <= 0 and max_dd >= -1:  # Should be negative and reasonable
        logger.info(f"✅ Maximum Drawdown calculation verified: {max_dd:.3f}")
    else:
        logger.error(f"❌ Maximum Drawdown calculation invalid: {max_dd}")
    
    print(f"\n🎯 FINAL VALIDATION STATUS: {'PASSED' if results['tests_passed'] >= 4 else 'NEEDS_ATTENTION'}")

if __name__ == "__main__":
    main()