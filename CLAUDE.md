# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive options trading and algorithmic trading educational repository containing detailed markdown guides covering professional trading strategies, backtesting frameworks, and quantitative analysis techniques. The repository serves as an educational resource for experienced traders and quantitative analysts.

## Core Architecture

### Document Structure
- **`option_guide.md`** - Institutional-grade options trading strategies based on practices from Jane Street, Citadel Securities, Optiver, and SIG. Contains defensive trading approaches with comprehensive risk management.
- **`aggressive_option_guide.md`** - Advanced aggressive options strategies including 0DTE, gamma squeeze detection, and high-leverage techniques. Includes proper risk frameworks and psychological considerations for experienced traders.
- **`backtest_guide.md`** - Professional backtesting framework using modern Python libraries (VectorBT, Polars, DuckDB, QF-Lib). Covers institutional-grade backtesting with Monte Carlo permutation testing.
- **`strategy_guide.md`** - Comprehensive algorithmic trading strategy development guide with 46+ proven strategies and institutional best practices from top firms.
- **`ssrn-3247865.pdf`** - Academic research paper providing quantitative foundation for the trading concepts.

## Key Development Dependencies

### Python Libraries (from backtest_guide.md)
```bash
# Core libraries
pip install yfinance pandas numpy matplotlib seaborn scipy quantstats pandas-ta

# High-performance backtesting
pip install polars duckdb vectorbt riskfolio-lib openbb qf-lib

# Pattern detection and ML
pip install Cython scikit-learn hmmlearn plotnine pyarrow fastparquet numba joblib tqdm
```

### Professional Trading Tools
- **Execution Platforms**: Interactive Brokers, TD Ameritrade thinkorswim, Tradier
- **Analysis Tools**: TradingView, OptionStrat, Options AI
- **Flow Analysis**: FlowAlgo, Unusual Whales, BlackBox Stocks
- **AI/ML Platforms**: QuantConnect, Alpaca, TensorFlow/PyTorch

## Implementation Framework

### Backtesting Architecture
The guides reference a sophisticated backtesting framework using:
- **VectorBT** for ultra-fast vectorized backtesting
- **Polars** for high-performance data processing
- **DuckDB** for analytical SQL queries on financial data
- **QF-Lib** for institutional-grade event-driven backtesting
- **Riskfolio-Lib** for portfolio optimization

### Risk Management Systems
All strategies include comprehensive risk management with:
- Real-time position monitoring
- Dynamic Greeks management for options
- Portfolio-level risk limits
- Monte Carlo permutation testing for overfitting detection

## Code Style and Patterns

### Strategy Implementation Pattern
```python
class TradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.positions = {}
        self.risk_limits = config['risk_limits']
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Signal generation logic
        pass
    
    def calculate_position_size(self, signal: float, risk_metrics: Dict) -> float:
        # Position sizing with risk management
        pass
    
    def execute_trades(self, signals: pd.Series) -> Dict:
        # Trade execution with slippage and costs
        pass
```

### Options Strategy Pattern
```python
class OptionsStrategy:
    def calculate_greeks(self, option_data: Dict) -> Dict:
        # Greeks calculation (delta, gamma, theta, vega)
        pass
    
    def risk_management(self, positions: Dict) -> List[Dict]:
        # Real-time risk monitoring
        pass
    
    def execute_hedge(self, exposure: Dict) -> Dict:
        # Dynamic hedging logic
        pass
```

## Working with This Repository

### Adding New Strategies
When implementing new trading strategies:
1. Follow the established pattern of risk-first design
2. Include comprehensive backtesting with out-of-sample validation
3. Implement proper position sizing using Kelly Criterion or similar
4. Add Monte Carlo permutation testing to detect overfitting
5. Include both defensive and aggressive variants where appropriate

### Code Implementation
While this repository contains markdown guides, any Python implementations should:
- Use the high-performance libraries mentioned (VectorBT, Polars, DuckDB)
- Follow the QF-Lib framework for event-driven backtesting
- Implement proper error handling and logging
- Include comprehensive unit tests for all mathematical calculations
- Use type hints and proper documentation

### Performance Considerations
- Leverage vectorized operations using NumPy and Pandas
- Use Polars for large dataset operations (>1GB)
- Implement Numba JIT compilation for compute-intensive functions
- Consider DuckDB for complex analytical queries
- Use multiprocessing for parameter optimization

## Strategy Categories

### Defensive Options Strategies
- Market making with dynamic spread management
- Volatility arbitrage with GARCH/Heston models
- Greeks-based risk management systems
- Cross-asset arbitrage opportunities

### Aggressive Options Strategies  
- 0DTE trading with gamma scalping
- Gamma squeeze detection and exploitation
- High-frequency market making
- AI-driven sentiment analysis for options flow

### Algorithmic Trading Strategies
- Factor-based systematic strategies
- Machine learning prediction models
- Pattern recognition systems
- Portfolio optimization frameworks

## Security and Risk Considerations

- All strategies emphasize proper risk management over raw returns
- Position sizing limited to reasonable percentages (1-5% typical)
- Comprehensive stress testing and scenario analysis
- Real-time monitoring with automated circuit breakers
- Regulatory compliance considerations included

This repository serves as a comprehensive educational resource for quantitative trading, with emphasis on institutional-quality implementations and proper risk management throughout all strategies.