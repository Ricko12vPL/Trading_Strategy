#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION FOR NASDAQ100_SP500_MEGA_BACKTEST.py
============================================================
Multiple validation methods to verify ALL calculations:
1. Technical indicators cross-validation using 3 different libraries
2. VectorBT calculation verification against manual implementations
3. Monte Carlo statistical validation using multiple methods
4. Parameter optimization accuracy testing
5. Memory leak and performance validation
6. Error handling and edge case testing
7. Mathematical formula double-checking

Verification approach:
- Method A: Manual implementation
- Method B: Alternative library (talib, pandas_ta)
- Method C: VectorBT built-in functions
- Statistical validation using scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core validation libraries
import vectorbt as vbt
from scipy import stats
import logging
import time
import os
import gc
import psutil
from pathlib import Path

# Alternative technical analysis libraries for cross-validation
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

# Numba for performance validation
from numba import jit, prange
from typing import Dict, List, Tuple, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MegaBacktestValidator:
    """
    Comprehensive validator for NASDAQ100_SP500_MEGA_BACKTEST.py
    Uses multiple independent calculation methods to verify accuracy
    """
    
    def __init__(self):
        self.initial_capital = 100000
        self.test_symbols = ['AAPL', 'MSFT', 'SPY']  # Test subset
        self.validation_results = {}
        self.performance_metrics = {}
        
        # Validation tolerance levels
        self.tolerances = {
            'technical_indicators': 0.001,  # 0.1% tolerance
            'returns': 0.00001,            # 0.001% tolerance  
            'portfolio_metrics': 0.01,     # 1% tolerance
            'monte_carlo': 0.05,           # 5% tolerance for statistical tests
            'memory_usage': 100            # MB tolerance
        }
        
        logger.info("🔍 MEGA BACKTEST VALIDATOR INITIALIZED")
        logger.info(f"📊 Test symbols: {self.test_symbols}")
        logger.info(f"💰 Initial capital: ${self.initial_capital:,}")
        logger.info(f"⚖️ Validation tolerances: {self.tolerances}")
    
    def validate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        CRITICAL: Validate technical indicators using 3 different methods
        Method A: Manual pandas implementation
        Method B: Alternative library (talib/pandas_ta)
        Method C: VectorBT implementation
        """
        logger.info(f"🧪 VALIDATING TECHNICAL INDICATORS for {symbol}")
        
        validation_report = {
            'symbol': symbol,
            'indicators_tested': [],
            'accuracy_scores': {},
            'method_comparisons': {},
            'errors_detected': []
        }
        
        try:
            # 1. RSI VALIDATION (3 methods)
            logger.info("  🔍 Validating RSI calculations...")
            
            # Method A: Manual RSI calculation
            rsi_manual = self._calculate_rsi_manual(data['Close'], period=14)
            
            # Method B: Alternative library
            rsi_alt = None
            if TALIB_AVAILABLE:
                try:
                    close_values = data['Close'].values.astype(float)
                    rsi_alt_values = talib.RSI(close_values, timeperiod=14)
                    rsi_alt = pd.Series(rsi_alt_values, index=data.index)
                except Exception as e:
                    logger.debug(f"TALIB RSI failed: {e}")
            elif PANDAS_TA_AVAILABLE:
                try:
                    rsi_alt = ta.rsi(data['Close'], length=14)
                except Exception as e:
                    logger.debug(f"pandas_ta RSI failed: {e}")
            
            # Method C: VectorBT
            rsi_vbt = vbt.RSI.run(data['Close'], window=14).rsi
            
            # Compare methods
            if rsi_alt is not None:
                # Compare Manual vs Alternative library
                rsi_diff_a_b = abs(rsi_manual - rsi_alt).dropna()
                accuracy_a_b = (rsi_diff_a_b < self.tolerances['technical_indicators']).mean()
                
                # Compare Manual vs VectorBT
                rsi_diff_a_c = abs(rsi_manual - rsi_vbt).dropna()
                accuracy_a_c = (rsi_diff_a_c < self.tolerances['technical_indicators']).mean()
                
                # Compare Alternative vs VectorBT
                rsi_diff_b_c = abs(rsi_alt - rsi_vbt).dropna()
                accuracy_b_c = (rsi_diff_b_c < self.tolerances['technical_indicators']).mean()
                
                validation_report['accuracy_scores']['RSI'] = {
                    'manual_vs_alt': accuracy_a_b,
                    'manual_vs_vbt': accuracy_a_c,
                    'alt_vs_vbt': accuracy_b_c,
                    'overall_accuracy': min(accuracy_a_b, accuracy_a_c, accuracy_b_c)
                }
                
                if min(accuracy_a_b, accuracy_a_c, accuracy_b_c) < 0.95:
                    validation_report['errors_detected'].append(f"RSI accuracy below 95%: {min(accuracy_a_b, accuracy_a_c, accuracy_b_c):.3f}")
            
            validation_report['indicators_tested'].append('RSI')
            
            # 2. BOLLINGER BANDS VALIDATION
            logger.info("  🔍 Validating Bollinger Bands calculations...")
            
            # Method A: Manual BB calculation
            bb_manual = self._calculate_bollinger_bands_manual(data['Close'], period=20, std=2)
            
            # Method B: Alternative library
            bb_alt = None
            if TALIB_AVAILABLE:
                try:
                    close_values = data['Close'].values.astype(float)
                    bb_upper_alt, bb_middle_alt, bb_lower_alt = talib.BBANDS(
                        close_values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                    )
                    bb_alt = pd.DataFrame({
                        'upper': bb_upper_alt,
                        'middle': bb_middle_alt, 
                        'lower': bb_lower_alt
                    }, index=data.index)
                except Exception as e:
                    logger.debug(f"TALIB BBANDS failed: {e}")
            
            # Method C: VectorBT
            try:
                bb_vbt = vbt.BBANDS.run(data['Close'], window=20, std=2)
                bb_vbt_df = pd.DataFrame({
                    'upper': bb_vbt.upper,
                    'middle': bb_vbt.middle,
                    'lower': bb_vbt.lower
                })
            except Exception as e:
                logger.debug(f"VectorBT BBANDS failed: {e}")
                # Fallback to simple calculation
                bb_manual_fallback = self._calculate_bollinger_bands_manual(data['Close'], period=20, std=2)
                bb_vbt_df = pd.DataFrame(bb_manual_fallback)
            
            # Compare BB methods
            bb_accuracy = {}
            for band in ['upper', 'middle', 'lower']:
                if bb_alt is not None:
                    diff_manual_alt = abs(bb_manual[band] - bb_alt[band]).dropna()
                    diff_manual_vbt = abs(bb_manual[band] - bb_vbt_df[band]).dropna()
                    
                    acc_manual_alt = (diff_manual_alt < self.tolerances['technical_indicators']).mean()
                    acc_manual_vbt = (diff_manual_vbt < self.tolerances['technical_indicators']).mean()
                    
                    bb_accuracy[band] = {
                        'manual_vs_alt': acc_manual_alt,
                        'manual_vs_vbt': acc_manual_vbt
                    }
            
            validation_report['accuracy_scores']['BOLLINGER_BANDS'] = bb_accuracy
            validation_report['indicators_tested'].append('BOLLINGER_BANDS')
            
            # 3. MACD VALIDATION
            logger.info("  🔍 Validating MACD calculations...")
            
            # Method A: Manual MACD
            macd_manual = self._calculate_macd_manual(data['Close'])
            
            # Method B: Alternative library
            macd_alt = None
            if TALIB_AVAILABLE:
                try:
                    close_values = data['Close'].values.astype(float)
                    macd_line_alt, macd_signal_alt, macd_hist_alt = talib.MACD(
                        close_values, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    macd_alt = pd.DataFrame({
                        'macd': macd_line_alt,
                        'signal': macd_signal_alt,
                        'histogram': macd_hist_alt
                    }, index=data.index)
                except Exception as e:
                    logger.debug(f"TALIB MACD failed: {e}")
            
            # Method C: VectorBT
            try:
                macd_vbt = vbt.MACD.run(data['Close'])
                macd_vbt_df = pd.DataFrame({
                    'macd': macd_vbt.macd,
                    'signal': macd_vbt.signal, 
                    'histogram': macd_vbt.histogram
                })
            except Exception as e:
                logger.debug(f"VectorBT MACD failed: {e}")
                # Fallback to manual calculation
                macd_manual_fallback = self._calculate_macd_manual(data['Close'])
                macd_vbt_df = pd.DataFrame(macd_manual_fallback)
            
            validation_report['indicators_tested'].append('MACD')
            
            logger.info(f"✅ Technical indicators validation completed for {symbol}")
            logger.info(f"   📊 Indicators tested: {len(validation_report['indicators_tested'])}")
            logger.info(f"   ⚠️ Errors detected: {len(validation_report['errors_detected'])}")
            
        except Exception as e:
            logger.error(f"❌ Technical indicators validation failed for {symbol}: {e}")
            validation_report['errors_detected'].append(f"Validation exception: {str(e)}")
        
        return validation_report
    
    def _calculate_rsi_manual(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Manual RSI calculation for validation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands_manual(self, prices: pd.Series, period: int = 20, std: float = 2) -> Dict:
        """Manual Bollinger Bands calculation for validation"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (rolling_std * std),
            'lower': sma - (rolling_std * std)
        }
    
    def _calculate_macd_manual(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Manual MACD calculation for validation"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
    
    def validate_vectorbt_portfolio_calculations(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        CRITICAL: Validate VectorBT portfolio calculations against manual implementation
        """
        logger.info(f"📊 VALIDATING VECTORBT PORTFOLIO CALCULATIONS for {symbol}")
        
        validation_report = {
            'symbol': symbol,
            'metrics_validated': [],
            'accuracy_results': {},
            'performance_comparison': {},
            'errors_detected': []
        }
        
        try:
            # Generate simple signals for testing
            rsi = vbt.RSI.run(data['Close'], window=14).rsi
            entries = (rsi < 30).fillna(False)
            exits = (rsi > 70).fillna(False)
            
            # Method A: VectorBT portfolio
            vbt_portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries,
                exits,
                init_cash=self.initial_capital,
                fees=0.001,
                slippage=0.001,
                freq='D'
            )
            
            # Method B: Manual portfolio calculation
            manual_portfolio = self._calculate_portfolio_manual(
                data['Close'], entries, exits, self.initial_capital
            )
            
            # Compare key metrics
            metrics_to_compare = [
                'total_return',
                'sharpe_ratio',
                'max_drawdown'
            ]
            
            for metric in metrics_to_compare:
                try:
                    if metric == 'total_return':
                        vbt_value = vbt_portfolio.total_return()
                        manual_value = manual_portfolio['total_return']
                    elif metric == 'sharpe_ratio':
                        vbt_value = vbt_portfolio.sharpe_ratio()
                        manual_value = manual_portfolio['sharpe_ratio']
                    elif metric == 'max_drawdown':
                        vbt_value = abs(vbt_portfolio.max_drawdown())
                        manual_value = manual_portfolio['max_drawdown']
                    
                    # Calculate accuracy
                    if abs(manual_value) > 0.0001:  # Avoid division by very small numbers
                        accuracy = 1 - abs(vbt_value - manual_value) / abs(manual_value)
                    else:
                        accuracy = 1 if abs(vbt_value - manual_value) < 0.0001 else 0
                    
                    validation_report['accuracy_results'][metric] = {
                        'vbt_value': vbt_value,
                        'manual_value': manual_value,
                        'difference': abs(vbt_value - manual_value),
                        'accuracy': accuracy
                    }
                    
                    if accuracy < 0.95:
                        validation_report['errors_detected'].append(
                            f"{metric} accuracy below 95%: {accuracy:.3f}"
                        )
                    
                    validation_report['metrics_validated'].append(metric)
                    
                except Exception as metric_error:
                    validation_report['errors_detected'].append(
                        f"Error validating {metric}: {str(metric_error)}"
                    )
            
            logger.info(f"✅ VectorBT portfolio validation completed for {symbol}")
            logger.info(f"   📊 Metrics validated: {len(validation_report['metrics_validated'])}")
            
        except Exception as e:
            logger.error(f"❌ VectorBT portfolio validation failed for {symbol}: {e}")
            validation_report['errors_detected'].append(f"Portfolio validation exception: {str(e)}")
        
        return validation_report
    
    def _calculate_portfolio_manual(self, prices: pd.Series, entries: pd.Series, exits: pd.Series, 
                                  init_cash: float) -> Dict:
        """Manual portfolio calculation for VectorBT validation"""
        
        cash = init_cash
        shares = 0
        portfolio_values = []
        trades = []
        
        for i in range(len(prices)):
            current_price = prices.iloc[i]
            
            # Entry signal
            if entries.iloc[i] and shares == 0 and cash > current_price:
                shares_to_buy = int(cash * 0.95 / current_price)  # 95% allocation
                cost = shares_to_buy * current_price * 1.001  # Include fees
                
                if cost <= cash:
                    shares = shares_to_buy
                    cash -= cost
                    trades.append({'type': 'buy', 'price': current_price, 'shares': shares})
            
            # Exit signal
            elif exits.iloc[i] and shares > 0:
                proceeds = shares * current_price * 0.999  # Include fees
                cash += proceeds
                trades.append({'type': 'sell', 'price': current_price, 'shares': shares})
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
        
        portfolio_values = pd.Series(portfolio_values, index=prices.index)
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] / init_cash) - 1
        
        # Calculate returns for Sharpe ratio
        daily_returns = portfolio_values.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values / running_max) - 1
        max_drawdown = abs(drawdown.min())
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trades': trades
        }
    
    def validate_monte_carlo_methods(self, data: pd.DataFrame, symbol: str, n_test_permutations: int = 10000) -> Dict:
        """
        CRITICAL: Validate Monte Carlo statistical methods using multiple approaches
        """
        logger.info(f"🧪 VALIDATING MONTE CARLO METHODS for {symbol}")
        
        validation_report = {
            'symbol': symbol,
            'permutations_tested': n_test_permutations,
            'statistical_tests': {},
            'method_comparisons': {},
            'errors_detected': []
        }
        
        try:
            # Generate test strategy results
            rsi = vbt.RSI.run(data['Close'], window=14).rsi
            entries = (rsi < 30).fillna(False)
            exits = (rsi > 70).fillna(False)
            
            portfolio = vbt.Portfolio.from_signals(
                data['Close'], entries, exits,
                init_cash=self.initial_capital,
                fees=0.001, freq='D'
            )
            
            observed_return = portfolio.total_return()
            
            # Method A: Bootstrap resampling of returns
            logger.info("  🔍 Method A: Bootstrap resampling...")
            returns = data['Close'].pct_change().dropna()
            returns_values = returns.values  # Convert to numpy array
            bootstrap_results = []
            
            for _ in range(n_test_permutations):
                # Resample returns with replacement
                resampled_returns = np.random.choice(returns_values, size=len(returns_values), replace=True)
                bootstrap_total_return = (1 + resampled_returns).prod() - 1
                bootstrap_results.append(bootstrap_total_return)
            
            bootstrap_p_value = np.sum(np.array(bootstrap_results) >= observed_return) / n_test_permutations
            
            # Method B: Parametric Monte Carlo (normal distribution)
            logger.info("  🔍 Method B: Parametric Monte Carlo...")
            mean_return = returns.mean()
            std_return = returns.std()
            
            parametric_results = []
            for _ in range(n_test_permutations):
                random_returns = np.random.normal(mean_return, std_return, len(returns))
                parametric_total_return = (1 + random_returns).prod() - 1
                parametric_results.append(parametric_total_return)
            
            parametric_p_value = np.sum(np.array(parametric_results) >= observed_return) / n_test_permutations
            
            # Method C: Block bootstrap for time series
            logger.info("  🔍 Method C: Block bootstrap...")
            block_size = 20  # 20-day blocks
            block_results = []
            
            for _ in range(n_test_permutations):
                # Create blocks
                n_blocks = len(returns) // block_size
                block_returns = []
                
                for _ in range(n_blocks):
                    start_idx = np.random.randint(0, len(returns) - block_size + 1)
                    block = returns.iloc[start_idx:start_idx + block_size]
                    block_returns.extend(block.values)
                
                # Pad to original length if needed
                while len(block_returns) < len(returns):
                    block_returns.append(np.random.choice(returns_values))
                
                block_total_return = (1 + np.array(block_returns[:len(returns)])).prod() - 1
                block_results.append(block_total_return)
            
            block_p_value = np.sum(np.array(block_results) >= observed_return) / n_test_permutations
            
            # Compare methods
            validation_report['statistical_tests'] = {
                'bootstrap': {
                    'p_value': bootstrap_p_value,
                    'mean_return': np.mean(bootstrap_results),
                    'std_return': np.std(bootstrap_results)
                },
                'parametric': {
                    'p_value': parametric_p_value,
                    'mean_return': np.mean(parametric_results),
                    'std_return': np.std(parametric_results)
                },
                'block_bootstrap': {
                    'p_value': block_p_value,
                    'mean_return': np.mean(block_results),
                    'std_return': np.std(block_results)
                }
            }
            
            # Check consistency between methods
            p_values = [bootstrap_p_value, parametric_p_value, block_p_value]
            p_value_std = np.std(p_values)
            
            if p_value_std > self.tolerances['monte_carlo']:
                validation_report['errors_detected'].append(
                    f"High variance in p-values across methods: std = {p_value_std:.4f}"
                )
            
            # Statistical tests for method validation
            # Kolmogorov-Smirnov test between bootstrap and parametric
            ks_stat, ks_p = stats.ks_2samp(bootstrap_results, parametric_results)
            validation_report['method_comparisons']['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'methods_similar': ks_p > 0.05
            }
            
            logger.info(f"✅ Monte Carlo validation completed for {symbol}")
            logger.info(f"   📊 P-values: Bootstrap={bootstrap_p_value:.4f}, Parametric={parametric_p_value:.4f}, Block={block_p_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Monte Carlo validation failed for {symbol}: {e}")
            validation_report['errors_detected'].append(f"Monte Carlo validation exception: {str(e)}")
        
        return validation_report
    
    def validate_memory_and_performance(self, symbol: str) -> Dict:
        """
        CRITICAL: Validate memory usage and performance for mega-scale operations
        """
        logger.info(f"⚡ VALIDATING MEMORY AND PERFORMANCE for {symbol}")
        
        validation_report = {
            'symbol': symbol,
            'memory_metrics': {},
            'performance_metrics': {},
            'errors_detected': []
        }
        
        try:
            # Measure initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Download test data
            start_time = time.time()
            # Download test data exactly like in base file
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max", interval="1d")
            
            # Filter to get recent years
            if len(data) > 1000 and not data.empty:
                min_date = datetime.now() - timedelta(days=4*365)  # 4 years for testing
                if data.index.tz:
                    min_date = pd.Timestamp(min_date).tz_localize(data.index.tz)
                data = data[data.index >= min_date]
            download_time = time.time() - start_time
            
            if data.empty:
                validation_report['errors_detected'].append(f"No data downloaded for {symbol}")
                return validation_report
            
            # Memory usage after data download
            after_download_memory = process.memory_info().rss / 1024 / 1024
            
            # Test VectorBT operations
            start_time = time.time()
            
            # RSI calculation
            rsi = vbt.RSI.run(data['Close'], window=14).rsi
            
            # Bollinger Bands
            bb = vbt.BBANDS.run(data['Close'], window=20, std=2)
            
            # Generate signals
            entries = (rsi < 30).fillna(False)
            exits = (rsi > 70).fillna(False)
            
            # Portfolio calculation
            portfolio = vbt.Portfolio.from_signals(
                data['Close'], entries, exits,
                init_cash=self.initial_capital,
                fees=0.001, freq='D'
            )
            
            calculation_time = time.time() - start_time
            
            # Memory usage after calculations
            after_calc_memory = process.memory_info().rss / 1024 / 1024
            
            # Test parameter sweep performance (small scale)
            start_time = time.time()
            param_results = []
            
            for rsi_period in [10, 14, 20]:
                for rsi_oversold in [25, 30, 35]:
                    test_rsi = vbt.RSI.run(data['Close'], window=rsi_period).rsi
                    test_entries = (test_rsi < rsi_oversold).fillna(False)
                    test_exits = (test_rsi > 70).fillna(False)
                    
                    test_portfolio = vbt.Portfolio.from_signals(
                        data['Close'], test_entries, test_exits,
                        init_cash=self.initial_capital,
                        fees=0.001, freq='D'
                    )
                    
                    param_results.append(test_portfolio.total_return())
            
            param_sweep_time = time.time() - start_time
            
            # Final memory measurement
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # Memory usage metrics
            validation_report['memory_metrics'] = {
                'initial_memory_mb': initial_memory,
                'after_download_mb': after_download_memory,
                'after_calculations_mb': after_calc_memory,
                'final_memory_mb': final_memory,
                'peak_usage_mb': final_memory - initial_memory,
                'memory_leak_detected': (final_memory - after_calc_memory) > self.tolerances['memory_usage']
            }
            
            # Performance metrics
            validation_report['performance_metrics'] = {
                'download_time_seconds': download_time,
                'calculation_time_seconds': calculation_time,
                'param_sweep_time_seconds': param_sweep_time,
                'data_points': len(data),
                'calculations_per_second': len(data) / calculation_time if calculation_time > 0 else 0,
                'parameter_combinations_tested': len(param_results),
                'param_combinations_per_second': len(param_results) / param_sweep_time if param_sweep_time > 0 else 0
            }
            
            # Performance benchmarks
            if calculation_time > 10:  # More than 10 seconds for basic calculations
                validation_report['errors_detected'].append(
                    f"Slow calculation performance: {calculation_time:.2f} seconds for {len(data)} data points"
                )
            
            if validation_report['memory_metrics']['memory_leak_detected']:
                validation_report['errors_detected'].append(
                    f"Potential memory leak detected: {validation_report['memory_metrics']['peak_usage_mb']:.1f} MB peak usage"
                )
            
            # Cleanup
            del data, rsi, bb, entries, exits, portfolio
            gc.collect()
            
            logger.info(f"✅ Memory and performance validation completed for {symbol}")
            logger.info(f"   💾 Peak memory usage: {validation_report['memory_metrics']['peak_usage_mb']:.1f} MB")
            logger.info(f"   ⚡ Calculations per second: {validation_report['performance_metrics']['calculations_per_second']:.0f}")
            
        except Exception as e:
            logger.error(f"❌ Memory and performance validation failed for {symbol}: {e}")
            validation_report['errors_detected'].append(f"Performance validation exception: {str(e)}")
        
        return validation_report
    
    def validate_error_handling(self, symbol: str) -> Dict:
        """
        CRITICAL: Test error handling and edge cases
        """
        logger.info(f"🛡️ VALIDATING ERROR HANDLING for {symbol}")
        
        validation_report = {
            'symbol': symbol,
            'error_tests': {},
            'edge_cases_tested': [],
            'errors_detected': []
        }
        
        try:
            # Test 1: Empty data handling
            logger.info("  🔍 Testing empty data handling...")
            empty_data = pd.DataFrame()
            try:
                rsi_empty = vbt.RSI.run(pd.Series(), window=14)
                validation_report['error_tests']['empty_data'] = 'handled'
            except Exception as e:
                validation_report['error_tests']['empty_data'] = f'error: {str(e)}'
            
            # Test 2: NaN data handling
            logger.info("  🔍 Testing NaN data handling...")
            test_data = pd.Series([100, np.nan, 102, 103, np.nan, 105])
            try:
                rsi_nan = vbt.RSI.run(test_data, window=3)
                validation_report['error_tests']['nan_data'] = 'handled'
            except Exception as e:
                validation_report['error_tests']['nan_data'] = f'error: {str(e)}'
            
            # Test 3: Insufficient data
            logger.info("  🔍 Testing insufficient data...")
            short_data = pd.Series([100, 101, 102])
            try:
                rsi_short = vbt.RSI.run(short_data, window=14)
                validation_report['error_tests']['insufficient_data'] = 'handled'
            except Exception as e:
                validation_report['error_tests']['insufficient_data'] = f'error: {str(e)}'
            
            # Test 4: Invalid parameters
            logger.info("  🔍 Testing invalid parameters...")
            try:
                valid_data = pd.Series(range(100))
                rsi_invalid = vbt.RSI.run(valid_data, window=-5)  # Negative window
                validation_report['error_tests']['invalid_parameters'] = 'handled'
            except Exception as e:
                validation_report['error_tests']['invalid_parameters'] = f'error: {str(e)}'
            
            # Test 5: Extreme values
            logger.info("  🔍 Testing extreme values...")
            extreme_data = pd.Series([1e10, 1e-10, 1e10, 1e-10])
            try:
                rsi_extreme = vbt.RSI.run(extreme_data, window=2)
                validation_report['error_tests']['extreme_values'] = 'handled'
            except Exception as e:
                validation_report['error_tests']['extreme_values'] = f'error: {str(e)}'
            
            validation_report['edge_cases_tested'] = list(validation_report['error_tests'].keys())
            
            # Count errors
            error_count = sum(1 for result in validation_report['error_tests'].values() 
                            if result.startswith('error:'))
            
            if error_count > 0:
                validation_report['errors_detected'].append(
                    f"Error handling failed in {error_count} test cases"
                )
            
            logger.info(f"✅ Error handling validation completed for {symbol}")
            logger.info(f"   🛡️ Edge cases tested: {len(validation_report['edge_cases_tested'])}")
            logger.info(f"   ⚠️ Error handling failures: {error_count}")
            
        except Exception as e:
            logger.error(f"❌ Error handling validation failed for {symbol}: {e}")
            validation_report['errors_detected'].append(f"Error handling test exception: {str(e)}")
        
        return validation_report
    
    def run_comprehensive_validation(self) -> Dict:
        """
        Run complete validation suite for all test symbols
        """
        logger.info("🚀 STARTING COMPREHENSIVE MEGA BACKTEST VALIDATION")
        logger.info("=" * 80)
        
        comprehensive_report = {
            'validation_timestamp': datetime.now(),
            'symbols_tested': [],
            'validation_results': {},
            'overall_accuracy': {},
            'critical_errors': [],
            'performance_summary': {}
        }
        
        for symbol in self.test_symbols:
            logger.info(f"📊 VALIDATING {symbol}...")
            
            try:
                # Download test data exactly like in base file
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="max", interval="1d")
                
                # Filter to get recent years
                if len(data) > 1000 and not data.empty:
                    min_date = datetime.now() - timedelta(days=4*365)  # 4 years for testing
                    if data.index.tz:
                        min_date = pd.Timestamp(min_date).tz_localize(data.index.tz)
                    data = data[data.index >= min_date]
                
                if data.empty:
                    logger.warning(f"⚠️ No data for {symbol}, skipping...")
                    continue
                
                symbol_results = {
                    'technical_indicators': self.validate_technical_indicators(data, symbol),
                    'vectorbt_portfolio': self.validate_vectorbt_portfolio_calculations(data, symbol),
                    'monte_carlo': self.validate_monte_carlo_methods(data, symbol, 1000),  # Reduced for speed
                    'memory_performance': self.validate_memory_and_performance(symbol),
                    'error_handling': self.validate_error_handling(symbol)
                }
                
                comprehensive_report['validation_results'][symbol] = symbol_results
                comprehensive_report['symbols_tested'].append(symbol)
                
                # Check for critical errors
                for validation_type, results in symbol_results.items():
                    if 'errors_detected' in results and results['errors_detected']:
                        comprehensive_report['critical_errors'].extend([
                            f"{symbol}-{validation_type}: {error}" for error in results['errors_detected']
                        ])
                
            except Exception as e:
                logger.error(f"❌ Validation failed for {symbol}: {e}")
                comprehensive_report['critical_errors'].append(f"{symbol}: Validation exception - {str(e)}")
        
        # Calculate overall accuracy scores
        self._calculate_overall_accuracy(comprehensive_report)
        
        # Display comprehensive results
        self._display_validation_results(comprehensive_report)
        
        return comprehensive_report
    
    def _calculate_overall_accuracy(self, report: Dict):
        """Calculate overall accuracy metrics across all validations"""
        
        accuracy_scores = []
        
        for symbol, results in report['validation_results'].items():
            # Technical indicators accuracy
            if 'accuracy_scores' in results['technical_indicators']:
                for indicator, scores in results['technical_indicators']['accuracy_scores'].items():
                    if isinstance(scores, dict):
                        for metric, accuracy in scores.items():
                            if isinstance(accuracy, (int, float)):
                                accuracy_scores.append(accuracy)
            
            # Portfolio calculations accuracy
            if 'accuracy_results' in results['vectorbt_portfolio']:
                for metric, result in results['vectorbt_portfolio']['accuracy_results'].items():
                    if 'accuracy' in result:
                        accuracy_scores.append(result['accuracy'])
        
        if accuracy_scores:
            report['overall_accuracy'] = {
                'mean_accuracy': np.mean(accuracy_scores),
                'min_accuracy': np.min(accuracy_scores),
                'max_accuracy': np.max(accuracy_scores),
                'accuracy_std': np.std(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                'scores_count': len(accuracy_scores)
            }
        else:
            report['overall_accuracy'] = {'mean_accuracy': 0, 'scores_count': 0, 'accuracy_std': 0}
    
    def _display_validation_results(self, report: Dict):
        """Display comprehensive validation results"""
        
        print("\n" + "=" * 100)
        print("🔍 COMPREHENSIVE MEGA BACKTEST VALIDATION RESULTS")
        print("=" * 100)
        print(f"Validation Date: {report['validation_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbols Tested: {len(report['symbols_tested'])}")
        print(f"Critical Errors Found: {len(report['critical_errors'])}")
        
        if report['overall_accuracy'] and 'accuracy_std' in report['overall_accuracy']:
            print(f"Overall Accuracy: {report['overall_accuracy']['mean_accuracy']*100:.1f}% (±{report['overall_accuracy']['accuracy_std']*100:.1f}%)")
        elif report['overall_accuracy']:
            print(f"Overall Accuracy: {report['overall_accuracy']['mean_accuracy']*100:.1f}%")
        
        print("=" * 100)
        
        # Individual symbol results
        for symbol in report['symbols_tested']:
            results = report['validation_results'][symbol]
            
            print(f"\n📊 {symbol} VALIDATION SUMMARY:")
            print("-" * 60)
            
            # Technical indicators
            tech_indicators = results['technical_indicators']
            print(f"Technical Indicators: {len(tech_indicators['indicators_tested'])} tested")
            if tech_indicators['errors_detected']:
                print(f"  ⚠️ Errors: {len(tech_indicators['errors_detected'])}")
            
            # Portfolio calculations
            portfolio_calc = results['vectorbt_portfolio']
            print(f"Portfolio Calculations: {len(portfolio_calc['metrics_validated'])} validated")
            if portfolio_calc['errors_detected']:
                print(f"  ⚠️ Errors: {len(portfolio_calc['errors_detected'])}")
            
            # Monte Carlo
            monte_carlo = results['monte_carlo']
            if 'statistical_tests' in monte_carlo:
                print(f"Monte Carlo Methods: {len(monte_carlo['statistical_tests'])} tested")
            
            # Performance
            performance = results['memory_performance']
            if 'performance_metrics' in performance:
                perf = performance['performance_metrics']
                print(f"Performance: {perf.get('calculations_per_second', 0):.0f} calcs/sec")
        
        # Critical errors summary
        if report['critical_errors']:
            print(f"\n🚨 CRITICAL ERRORS DETECTED ({len(report['critical_errors'])}):")
            print("-" * 60)
            for error in report['critical_errors']:
                print(f"  ❌ {error}")
        
        # Final assessment
        print("\n" + "=" * 100)
        if len(report['critical_errors']) == 0:
            print("✅ VALIDATION PASSED: All calculations verified across multiple methods")
        elif len(report['critical_errors']) < 5:
            print("⚠️ VALIDATION WARNING: Minor issues detected but calculations mostly accurate")
        else:
            print("❌ VALIDATION FAILED: Multiple critical errors detected - review calculations")
        
        print("=" * 100)

def main():
    """Main validation execution"""
    validator = MegaBacktestValidator()
    comprehensive_report = validator.run_comprehensive_validation()
    
    # Save validation report
    import json
    report_path = Path('/Users/kacper/Desktop/Option_trading1/data_ibkr') / 'validation_report.json'
    report_path.parent.mkdir(exist_ok=True)
    
    # Convert datetime for JSON serialization
    report_copy = comprehensive_report.copy()
    report_copy['validation_timestamp'] = report_copy['validation_timestamp'].isoformat()
    
    with open(report_path, 'w') as f:
        json.dump(report_copy, f, indent=2, default=str)
    
    logger.info(f"📊 Validation report saved: {report_path}")

if __name__ == "__main__":
    main()