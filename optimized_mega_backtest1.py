#!/usr/bin/env python3
"""
ULTRA MEGA-SCALE BACKTEST - 10+ MILIONÓW RZECZYWISTYCH TESTÓW
===========================================================================
Wykorzystuje w pierwszej kolejności dane IBKR (15 lat) z lokalnego folderu
Następnie dopiero pobiera nowe dane jeśli potrzeba

10 MILIONÓW testów na instrument:
- Random Sampling: 4,000,000 testów
- Bayesian Optimization: 2,000,000 testów  
- Genetic Algorithm: 4,000,000 testów
"""

import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
import pickle
import warnings
import json
warnings.filterwarnings('ignore')

# Optymalizacja Bayesowska
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Algorytm Genetyczny
from deap import base, creator, tools, algorithms
import random

# Parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import os
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraMegaScaleBacktest:
    """
    10+ MILIONÓW rzeczywistych testów parametrów na instrument
    Wykorzystuje dane IBKR z lokalnego folderu jako pierwsze źródło
    """
    
    def __init__(self):
        self.initial_capital = 100000
        
        # Ścieżki do danych
        self.ibkr_data_dir = Path('/Users/kacper/Desktop/Option_trading1/data_ibkr')
        self.results_dir = Path('/Users/kacper/Desktop/Option_trading1/optimization_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # ZWIĘKSZONE x2 liczby testów - 10 MILIONÓW!
        self.target_tests = 10_000_000  # 10 milionów testów na instrument
        self.tests_per_method = {
            'random_sampling': 4_000_000,    # 4M testów (x2)
            'bayesian': 2_000_000,           # 2M testów (x2)
            'genetic': 4_000_000              # 4M testów (x2)
        }
        
        # Przestrzeń parametrów (rozszerzona)
        self.param_space = {
            'rsi_period': (5, 50),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'bb_period': (10, 50),
            'bb_std': (1.5, 3.0),
            'ma_fast': (5, 20),
            'ma_slow': (20, 100),
            'volume_multiplier': (1.2, 3.0),
            'atr_multiplier': (1.0, 3.0),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.10),
            'position_size': (0.01, 0.10),
            # Dodatkowe parametry
            'macd_fast': (8, 16),
            'macd_slow': (20, 35),
            'macd_signal': (5, 12)
        }
        
        self.n_cores = mp.cpu_count()
        
        # Skanuj dostępne dane IBKR
        self.available_ibkr_data = self.scan_ibkr_data()
        
        logger.info(f"🚀 Ultra Mega-Scale Backtest initialized")
        logger.info(f"📁 IBKR Data Directory: {self.ibkr_data_dir}")
        logger.info(f"📊 Found {len(self.available_ibkr_data)} instruments with IBKR data")
        logger.info(f"🎯 Target: {self.target_tests:,} tests per instrument")
        logger.info(f"💻 Cores: {self.n_cores}")
    
    def scan_ibkr_data(self) -> Dict[str, Path]:
        """Skanuj folder z danymi IBKR i zwróć dostępne instrumenty"""
        available = {}
        
        if not self.ibkr_data_dir.exists():
            logger.warning(f"⚠️ IBKR data directory not found: {self.ibkr_data_dir}")
            return available
        
        # Szukaj plików .pkl
        pkl_files = list(self.ibkr_data_dir.glob("*.pkl"))
        
        for file_path in pkl_files:
            # Wyciągnij symbol z nazwy pliku (zakładając format: SYMBOL_15Y_IBKR.pkl)
            filename = file_path.stem
            if '_' in filename:
                symbol = filename.split('_')[0]
                available[symbol] = file_path
                logger.debug(f"Found IBKR data for {symbol}: {file_path.name}")
        
        logger.info(f"📂 Scanned IBKR data: {len(available)} instruments found")
        if available:
            logger.info(f"   Symbols: {', '.join(list(available.keys())[:10])}{'...' if len(available) > 10 else ''}")
        
        return available
    
    def load_ibkr_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Załaduj dane IBKR z lokalnego pliku .pkl"""
        if symbol not in self.available_ibkr_data:
            return None
        
        file_path = self.available_ibkr_data[symbol]
        
        try:
            with open(file_path, 'rb') as f:
                data_package = pickle.load(f)
            
            # Sprawdź strukturę danych
            if isinstance(data_package, dict):
                data = data_package.get('data', data_package)
                logger.info(f"✅ Loaded IBKR data for {symbol}: {len(data)} records from {file_path.name}")
                return data
            elif isinstance(data_package, pd.DataFrame):
                logger.info(f"✅ Loaded IBKR data for {symbol}: {len(data_package)} records")
                return data_package
            else:
                logger.warning(f"⚠️ Unknown data format for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load IBKR data for {symbol}: {e}")
            return None
    
    def get_data(self, symbol: str) -> pd.DataFrame:
        """Pobierz dane - najpierw sprawdź IBKR, potem Yahoo"""
        # Najpierw spróbuj załadować dane IBKR
        data = self.load_ibkr_data(symbol)
        
        if data is not None:
            logger.info(f"📊 Using IBKR data for {symbol} (15 years)")
            return data
        
        # Jeśli nie ma danych IBKR, pobierz z Yahoo
        logger.info(f"📊 No IBKR data for {symbol}, downloading from Yahoo...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="max", interval="1d")
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        return data
    
    def calculate_indicators(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Oblicz wszystkie wskaźniki techniczne"""
        try:
            # RSI
            rsi = vbt.RSI.run(data['Close'], window=int(params['rsi_period'])).rsi.values
            
            # Bollinger Bands
            bb = vbt.BBANDS.run(data['Close'], window=int(params['bb_period']), alpha=params['bb_std'])
            bb_upper = bb.upper.values
            bb_lower = bb.lower.values
            
            # Moving Averages
            ma_fast = vbt.MA.run(data['Close'], window=int(params['ma_fast'])).ma.values
            ma_slow = vbt.MA.run(data['Close'], window=int(params['ma_slow'])).ma.values
            
            # MACD
            macd = vbt.MACD.run(
                data['Close'],
                fast_window=int(params.get('macd_fast', 12)),
                slow_window=int(params.get('macd_slow', 26)),
                signal_window=int(params.get('macd_signal', 9))
            )
            macd_line = macd.macd.values
            signal_line = macd.signal.values
            
            # ATR for volatility
            atr = vbt.ATR.run(data['High'], data['Low'], data['Close'], window=14).atr.values
            
            # Volume
            volume_sma = vbt.MA.run(data['Volume'], window=20).ma.values
            volume_ratio = data['Volume'].values / (volume_sma + 1e-10)
            
            return {
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'atr': atr,
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return None
    
    def evaluate_strategy(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Szybka ewaluacja pojedynczej strategii"""
        try:
            indicators = self.calculate_indicators(data, params)
            if indicators is None:
                return {'score': -999, 'return': 0, 'sharpe': 0, 'max_dd': -1}
            
            # Generuj sygnały
            entries = np.zeros(len(data), dtype=bool)
            exits = np.zeros(len(data), dtype=bool)
            
            # Warunki wejścia (wszystkie muszą być spełnione)
            entries = (
                (indicators['rsi'] < params['rsi_oversold']) &
                (data['Close'].values < indicators['bb_lower']) &
                (indicators['volume_ratio'] > params['volume_multiplier']) &
                (indicators['ma_fast'] > indicators['ma_slow']) &
                (indicators['macd_line'] > indicators['signal_line'])
            )
            
            # Warunki wyjścia
            exits = (
                (indicators['rsi'] > params['rsi_overbought']) |
                (data['Close'].values > indicators['bb_upper']) |
                (indicators['macd_line'] < indicators['signal_line'])
            )
            
            # Backtest z VectorBT
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries,
                exits,
                size=params['position_size'] * self.initial_capital,
                sl_stop=params['stop_loss'],
                tp_stop=params['take_profit'],
                fees=0.001,
                slippage=0.001,
                init_cash=self.initial_capital,
                freq='1D'
            )
            
            # Metryki
            total_return = portfolio.total_return()
            sharpe = portfolio.sharpe_ratio()
            max_dd = portfolio.max_drawdown()
            n_trades = portfolio.stats()['Total Trades'].values[0] if hasattr(portfolio.stats(), '__getitem__') else 0
            win_rate = portfolio.stats()['Win Rate [%]'].values[0] if hasattr(portfolio.stats(), '__getitem__') else 0
            
            # Score - wieloczynnikowa ocena
            if n_trades < 10:
                score = -100
            else:
                score = (total_return * 100) * max(sharpe, 0) * (1 - abs(max_dd)) * np.sqrt(n_trades/100) * (win_rate/50)
            
            return {
                'score': score,
                'return': total_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'params': params
            }
            
        except Exception as e:
            return {'score': -999, 'return': 0, 'sharpe': 0, 'max_dd': -1}
    
    def run_random_sampling(self, data: pd.DataFrame, n_samples: int = 4_000_000) -> List[Dict]:
        """Random Sampling - 4 MILIONY testów"""
        logger.info(f"🎲 Starting Random Sampling: {n_samples:,} tests")
        
        # Generuj losowe parametry
        random_params_list = []
        for i in range(n_samples):
            params = {}
            for param_name, (min_val, max_val) in self.param_space.items():
                if isinstance(min_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            random_params_list.append(params)
        
        # Parallel processing w paczkach
        batch_size = 10000
        n_batches = n_samples // batch_size
        all_results = []
        
        for batch_idx in tqdm(range(n_batches), desc="Random Sampling"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            batch_params = random_params_list[batch_start:batch_end]
            
            batch_results = Parallel(n_jobs=self.n_cores)(
                delayed(self.evaluate_strategy)(data, params) 
                for params in batch_params
            )
            
            all_results.extend(batch_results)
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"   Progress: {len(all_results):,}/{n_samples:,} ({len(all_results)/n_samples*100:.1f}%)")
        
        valid_results = [r for r in all_results if r['score'] > -100]
        sorted_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Random Sampling: {len(all_results):,} tested, {len(valid_results):,} valid")
        return sorted_results[:100]
    
    def run_bayesian_optimization(self, data: pd.DataFrame, n_calls: int = 2_000_000) -> List[Dict]:
        """Bayesian Optimization - 2 MILIONY testów"""
        logger.info(f"🧠 Starting Bayesian Optimization: {n_calls:,} tests")
        
        space = [
            Integer(5, 50, name='rsi_period'),
            Real(20, 40, name='rsi_oversold'),
            Real(60, 80, name='rsi_overbought'),
            Integer(10, 50, name='bb_period'),
            Real(1.5, 3.0, name='bb_std'),
            Integer(5, 20, name='ma_fast'),
            Integer(20, 100, name='ma_slow'),
            Real(1.2, 3.0, name='volume_multiplier'),
            Real(1.0, 3.0, name='atr_multiplier'),
            Real(0.01, 0.05, name='stop_loss'),
            Real(0.02, 0.10, name='take_profit'),
            Real(0.01, 0.10, name='position_size'),
            Integer(8, 16, name='macd_fast'),
            Integer(20, 35, name='macd_slow'),
            Integer(5, 12, name='macd_signal')
        ]
        
        all_tested_params = []
        all_scores = []
        
        @use_named_args(space)
        def objective(**params):
            result = self.evaluate_strategy(data, params)
            all_tested_params.append(params.copy())
            all_scores.append(result['score'])
            return -result['score']
        
        block_size = 20000
        n_blocks = n_calls // block_size
        
        best_score = -np.inf
        best_params = None
        
        for block_idx in tqdm(range(n_blocks), desc="Bayesian Optimization"):
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=block_size,
                n_initial_points=block_size // 10,
                acq_func='EI',
                n_jobs=self.n_cores,
                random_state=42 + block_idx
            )
            
            if -result.fun > best_score:
                best_score = -result.fun
                best_params = dict(zip([dim.name for dim in space], result.x))
            
            if (block_idx + 1) % 10 == 0:
                logger.info(f"   Block {block_idx+1}/{n_blocks}: Best score: {best_score:.2f}")
        
        results = []
        for params, score in zip(all_tested_params[-100:], all_scores[-100:]):
            result = self.evaluate_strategy(data, params)
            results.append(result)
        
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Bayesian Optimization: {len(all_tested_params):,} combinations tested")
        return sorted_results
    
    def run_genetic_algorithm(self, data: pd.DataFrame, population_size: int = 20000, 
                            n_generations: int = 200) -> List[Dict]:
        """Genetic Algorithm - 4 MILIONY testów (20k × 200)"""
        logger.info(f"🧬 Starting Genetic Algorithm: {population_size:,} × {n_generations} = {population_size*n_generations:,} tests")
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Definiuj geny
        toolbox.register("rsi_period", random.randint, 5, 50)
        toolbox.register("rsi_oversold", random.uniform, 20, 40)
        toolbox.register("rsi_overbought", random.uniform, 60, 80)
        toolbox.register("bb_period", random.randint, 10, 50)
        toolbox.register("bb_std", random.uniform, 1.5, 3.0)
        toolbox.register("ma_fast", random.randint, 5, 20)
        toolbox.register("ma_slow", random.randint, 20, 100)
        toolbox.register("volume_multiplier", random.uniform, 1.2, 3.0)
        toolbox.register("atr_multiplier", random.uniform, 1.0, 3.0)
        toolbox.register("stop_loss", random.uniform, 0.01, 0.05)
        toolbox.register("take_profit", random.uniform, 0.02, 0.10)
        toolbox.register("position_size", random.uniform, 0.01, 0.10)
        toolbox.register("macd_fast", random.randint, 8, 16)
        toolbox.register("macd_slow", random.randint, 20, 35)
        toolbox.register("macd_signal", random.randint, 5, 12)
        
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.rsi_period, toolbox.rsi_oversold, toolbox.rsi_overbought,
                         toolbox.bb_period, toolbox.bb_std, toolbox.ma_fast, toolbox.ma_slow,
                         toolbox.volume_multiplier, toolbox.atr_multiplier,
                         toolbox.stop_loss, toolbox.take_profit, toolbox.position_size,
                         toolbox.macd_fast, toolbox.macd_slow, toolbox.macd_signal),
                        n=1)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate_individual(individual):
            params = {
                'rsi_period': int(individual[0]),
                'rsi_oversold': individual[1],
                'rsi_overbought': individual[2],
                'bb_period': int(individual[3]),
                'bb_std': individual[4],
                'ma_fast': int(individual[5]),
                'ma_slow': int(individual[6]),
                'volume_multiplier': individual[7],
                'atr_multiplier': individual[8],
                'stop_loss': individual[9],
                'take_profit': individual[10],
                'position_size': individual[11],
                'macd_fast': int(individual[12]),
                'macd_slow': int(individual[13]),
                'macd_signal': int(individual[14])
            }
            result = self.evaluate_strategy(data, params)
            return (result['score'],)
        
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pool = mp.Pool(processes=self.n_cores)
        toolbox.register("map", pool.map)
        
        population = toolbox.population(n=population_size)
        hall_of_fame = tools.HallOfFame(100)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        for gen in tqdm(range(n_generations), desc="Genetic Algorithm"):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
            
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, k=population_size)
            hall_of_fame.update(population)
            
            if (gen + 1) % 20 == 0:
                record = stats.compile(population)
                logger.info(f"   Gen {gen+1}: Max={record['max']:.2f}, Avg={record['avg']:.2f}")
        
        pool.close()
        
        results = []
        for ind in hall_of_fame:
            params = {
                'rsi_period': int(ind[0]),
                'rsi_oversold': ind[1],
                'rsi_overbought': ind[2],
                'bb_period': int(ind[3]),
                'bb_std': ind[4],
                'ma_fast': int(ind[5]),
                'ma_slow': int(ind[6]),
                'volume_multiplier': ind[7],
                'atr_multiplier': ind[8],
                'stop_loss': ind[9],
                'take_profit': ind[10],
                'position_size': ind[11],
                'macd_fast': int(ind[12]),
                'macd_slow': int(ind[13]),
                'macd_signal': int(ind[14])
            }
            result = self.evaluate_strategy(data, params)
            results.append(result)
        
        logger.info(f"✅ Genetic Algorithm: {population_size * n_generations:,} individuals evaluated")
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def create_summary_report(self, all_results: Dict) -> Path:
        """Stwórz kompleksowy raport HTML ze wszystkich wyników"""
        html_path = self.results_dir / f"ULTRA_MEGA_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Mega-Scale Backtest Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff41; text-align: center; }
        h2 { color: #00ff41; border-bottom: 2px solid #00ff41; padding-bottom: 5px; }
        .summary { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .symbol-section { background: #2a2a2a; padding: 15px; margin: 15px 0; border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th { background: #00ff41; color: #000; padding: 10px; text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #444; }
        tr:hover { background: #3a3a3a; }
        .positive { color: #00ff41; }
        .negative { color: #ff4040; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 12px; color: #aaa; }
        .best-strategy { background: #003d0a; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🚀 ULTRA MEGA-SCALE BACKTEST RESULTS</h1>
    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div>
            <div class="metric">
                <div class="metric-value">""" + str(len(all_results)) + """</div>
                <div class="metric-label">Symbols Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">""" + f"{len(all_results) * 10_000_000:,}" + """</div>
                <div class="metric-label">Total Tests Performed</div>
            </div>
            <div class="metric">
                <div class="metric-value">""" + datetime.now().strftime('%Y-%m-%d %H:%M') + """</div>
                <div class="metric-label">Analysis Date</div>
            </div>
        </div>
    </div>
"""
        
        # Dla każdego symbolu
        for symbol, results in all_results.items():
            best_combined = results['combined'][0] if results['combined'] else None
            
            html_content += f"""
    <div class="symbol-section">
        <h2>📈 {symbol}</h2>
"""
            
            if best_combined:
                html_content += f"""
        <div class="best-strategy">
            <h3>🏆 Best Strategy Performance</h3>
            <table>
                <tr>
                    <td><strong>Total Return:</strong></td>
                    <td class="{'positive' if best_combined['return'] > 0 else 'negative'}">{best_combined['return']*100:.2f}%</td>
                    <td><strong>Sharpe Ratio:</strong></td>
                    <td>{best_combined['sharpe']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>Max Drawdown:</strong></td>
                    <td class="negative">{best_combined['max_dd']*100:.2f}%</td>
                    <td><strong>Total Trades:</strong></td>
                    <td>{best_combined.get('n_trades', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Win Rate:</strong></td>
                    <td>{best_combined.get('win_rate', 'N/A'):.1f}%</td>
                    <td><strong>Strategy Score:</strong></td>
                    <td>{best_combined['score']:.2f}</td>
                </tr>
            </table>
        </div>
"""
            
            # Top 5 strategies comparison
            html_content += """
        <h3>Top 5 Strategies Comparison</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Method</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for i, strategy in enumerate(results['combined'][:5], 1):
                method = "Combined"
                html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{method}</td>
                    <td class="{'positive' if strategy['return'] > 0 else 'negative'}">{strategy['return']*100:.2f}%</td>
                    <td>{strategy['sharpe']:.3f}</td>
                    <td class="negative">{strategy['max_dd']*100:.2f}%</td>
                    <td>{strategy.get('n_trades', 'N/A')}</td>
                    <td>{strategy.get('win_rate', 'N/A'):.1f}%</td>
                    <td>{strategy['score']:.2f}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📝 HTML Report saved: {html_path}")
        return html_path
    
    def run_complete_optimization(self, symbols: List[str] = None):
        """
        Uruchom kompletną optymalizację - 10 MILIONÓW testów na symbol
        """
        # Jeśli nie podano symboli, użyj wszystkich z IBKR
        if symbols is None:
            if self.available_ibkr_data:
                symbols = list(self.available_ibkr_data.keys())[:20]  # Pierwsze 20 symboli
            else:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        print("="*100)
        print("🚀 ULTRA MEGA-SCALE BACKTEST - 10+ MILLION TESTS PER SYMBOL")
        print("="*100)
        print(f"📁 IBKR Data Directory: {self.ibkr_data_dir}")
        print(f"📊 Available IBKR Data: {len(self.available_ibkr_data)} instruments")
        print(f"📈 Symbols to analyze: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        print(f"🎯 Tests per symbol: {self.target_tests:,}")
        print(f"   - Random Sampling: {self.tests_per_method['random_sampling']:,}")
        print(f"   - Bayesian Optimization: {self.tests_per_method['bayesian']:,}")
        print(f"   - Genetic Algorithm: {self.tests_per_method['genetic']:,}")
        print(f"💾 Results will be saved to: {self.results_dir}")
        print("="*100)
        
        all_results = {}
        summary_data = []
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n{'='*100}")
            print(f"📈 [{idx}/{len(symbols)}] Processing {symbol}...")
            print(f"{'='*100}")
            
            try:
                # Pobierz dane (IBKR lub Yahoo)
                data = self.get_data(symbol)
                data_source = "IBKR (15 years)" if symbol in self.available_ibkr_data else "Yahoo Finance"
                
                print(f"📊 Data loaded: {len(data)} records from {data_source}")
                
                if len(data) < 100:
                    logger.warning(f"⚠️ Insufficient data for {symbol}")
                    continue
                
                symbol_results = {
                    'random': [],
                    'bayesian': [],
                    'genetic': [],
                    'combined': [],
                    'data_source': data_source,
                    'data_points': len(data)
                }
                
                # 1. Random Sampling - 4 MILIONY
                print(f"\n🎲 Method 1: Random Sampling ({self.tests_per_method['random_sampling']:,} tests)")
                random_results = self.run_random_sampling(data, self.tests_per_method['random_sampling'])
                symbol_results['random'] = random_results[:10]
                
                # 2. Bayesian Optimization - 2 MILIONY
                print(f"\n🧠 Method 2: Bayesian Optimization ({self.tests_per_method['bayesian']:,} tests)")
                bayesian_results = self.run_bayesian_optimization(data, self.tests_per_method['bayesian'])
                symbol_results['bayesian'] = bayesian_results[:10]
                
                # 3. Genetic Algorithm - 4 MILIONY
                print(f"\n🧬 Method 3: Genetic Algorithm ({self.tests_per_method['genetic']:,} tests)")
                genetic_results = self.run_genetic_algorithm(data, 20000, 200)
                symbol_results['genetic'] = genetic_results[:10]
                
                # Połącz najlepsze wyniki
                all_best = random_results[:30] + bayesian_results[:30] + genetic_results[:30]
                symbol_results['combined'] = sorted(all_best, key=lambda x: x['score'], reverse=True)[:10]
                
                # Zapisz szczegółowe wyniki dla symbolu
                symbol_file = self.results_dir / f"{symbol}_10M_results.json"
                with open(symbol_file, 'w') as f:
                    # Konwertuj numpy types do Python types
                    def convert_numpy(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj
                    
                    json.dump(symbol_results, f, indent=2, default=convert_numpy)
                
                # Podsumowanie dla symbolu
                total_tests = sum(self.tests_per_method.values())
                best = symbol_results['combined'][0] if symbol_results['combined'] else None
                
                print(f"\n{'='*100}")
                print(f"📊 {symbol} RESULTS SUMMARY")
                print(f"{'='*100}")
                print(f"Data Source: {data_source}")
                print(f"Data Points: {len(data):,}")
                print(f"Total Tests: {total_tests:,}")
                
                if best:
                    print(f"\n🏆 BEST STRATEGY:")
                    print(f"   Return: {best['return']*100:.2f}%")
                    print(f"   Sharpe: {best['sharpe']:.3f}")
                    print(f"   Max DD: {best['max_dd']*100:.2f}%")
                    print(f"   Trades: {best.get('n_trades', 'N/A')}")
                    print(f"   Win Rate: {best.get('win_rate', 'N/A'):.1f}%")
                    print(f"   Score: {best['score']:.2f}")
                    
                    summary_data.append({
                        'symbol': symbol,
                        'return': best['return'],
                        'sharpe': best['sharpe'],
                        'max_dd': best['max_dd'],
                        'score': best['score']
                    })
                
                all_results[symbol] = symbol_results
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        # Stwórz zbiorczy raport HTML
        html_report = self.create_summary_report(all_results)
        
        # Stwórz plik CSV z podsumowaniem
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('score', ascending=False)
            csv_path = self.results_dir / f"SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"📊 Summary CSV saved: {csv_path}")
        
        # Końcowe podsumowanie
        print(f"\n{'='*100}")
        print("🎉 ULTRA MEGA-SCALE OPTIMIZATION COMPLETE!")
        print(f"{'='*100}")
        print(f"✅ Symbols Analyzed: {len(all_results)}")
        print(f"✅ Total Tests: {len(all_results) * self.target_tests:,}")
        print(f"✅ HTML Report: {html_report}")
        print(f"✅ Results Directory: {self.results_dir}")
        print(f"\n📂 OUTPUT FILES:")
        print(f"   1. Main HTML Report: {html_report.name}")
        print(f"   2. Individual JSON files: {symbol}_10M_results.json")
        print(f"   3. Summary CSV: SUMMARY_*.csv")
        print(f"\n🔍 TO VIEW RESULTS:")
        print(f"   Open: {html_report}")
        
        return all_results


def main():
    """Główna funkcja uruchamiająca"""
    optimizer = UltraMegaScaleBacktest()
    
    # Automatycznie użyj wszystkich dostępnych danych IBKR
    # Lub określ konkretne symbole
    results = optimizer.run_complete_optimization()
    
    return results


if __name__ == "__main__":
    results = main()
