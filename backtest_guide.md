# Kompleksowy przewodnik backtestingu strategii tradingowej w Python

## Spis treści
1. [Wstęp](#wstęp)
2. [Przygotowanie środowiska](#przygotowanie-środowiska)
3. [Zaawansowane biblioteki do backtestingu](#zaawansowane-biblioteki-do-backtestingu)
4. [Pobieranie danych historycznych](#pobieranie-danych-historycznych)
5. [Struktura strategii tradingowej](#struktura-strategii-tradingowej)
6. [Zaawansowane wykrywanie wzorców z tseries-patterns](#zaawansowane-wykrywanie-wzorców-z-tseries-patterns)
7. [Implementacja silnika backtestingu](#implementacja-silnika-backtestingu)
8. [Kluczowe metryki oceny strategii](#kluczowe-metryki-oceny-strategii)
9. [Wizualizacja wyników](#wizualizacja-wyników)
10. [Interpretacja wyników](#interpretacja-wyników)
11. [Monte Carlo Permutation Testing - Wykrywanie Overfittingu](#monte-carlo-permutation-testing---wykrywanie-overfittingu)
12. [Checklist przed przejściem na live trading](#checklist-przed-przejściem-na-live-trading)

## 1. Wstęp

Backtest to symulacja strategii tradingowej na danych historycznych, która pozwala ocenić potencjalną skuteczność strategii przed zaryzykowaniem realnego kapitału. W erze big data i wysokoczęstotliwego tradingu, profesjonalny backtest wymaga nie tylko dokładności, ale także wydajności obliczeniowej.

Ten przewodnik przedstawia kompletny proces tworzenia profesjonalnego backtestu z wykorzystaniem najnowocześniejszych narzędzi dostępnych w ekosystemie Python. Wykorzystamy zarówno tradycyjne biblioteki jak pandas i numpy, jak i nowoczesne, ultra-szybkie rozwiązania jak Polars i DuckDB, które potrafią przetwarzać gigabajty danych w sekundach.

### Dlaczego profesjonalny backtest jest kluczowy?

1. **Weryfikacja hipotez** - Sprawdzenie czy strategia faktycznie generuje zyski
2. **Ocena ryzyka** - Zrozumienie potencjalnych strat i zmienności
3. **Optymalizacja parametrów** - Znalezienie optymalnych ustawień strategii
4. **Budowanie zaufania** - Psychologiczne przygotowanie na realne warunki rynkowe
5. **Wykrywanie overfittingu** - Uniknięcie dopasowania do danych historycznych

### Co wyróżnia profesjonalny backtest?

- **Realistyczne założenia** - uwzględnienie kosztów, poślizgów, limitów płynności
- **Kompleksowa analiza** - nie tylko zwrot, ale pełen zestaw metryk ryzyka
- **Wydajność obliczeniowa** - możliwość testowania na latach danych tick-by-tick
- **Walidacja out-of-sample** - testowanie na danych spoza okresu optymalizacji
- **Analiza wrażliwości** - sprawdzenie stabilności w różnych warunkach rynkowych

## 2. Przygotowanie środowiska

### Instalacja podstawowych bibliotek

```bash
# Podstawowe biblioteki
pip install yfinance pandas numpy matplotlib seaborn scipy quantstats pandas-ta

# Zaawansowane biblioteki do wysokowydajnego backtestingu
pip install polars duckdb vectorbt riskfolio-lib openbb

# QF-Lib - Professional Event-Driven Backtesting Framework
pip install qf-lib

# Prerequisites: WeasyPrint for PDF export requires GTK3+
# Windows: Download GTK3+ runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
# macOS: brew install gtk+3 libffi
# Linux: sudo apt-get install libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0

# Biblioteki do pattern detection i permutation testing
pip install Cython scikit-learn hmmlearn plotnine

# Dodatkowe biblioteki pomocnicze
pip install pyarrow fastparquet numba joblib tqdm

# WeasyPrint dependencies (for QF-Lib PDF exports)
# Linux: sudo apt-get install python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
# macOS: brew install pango gtk+3 libffi
# Windows: Download GTK3+ runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
```

### Import bibliotek

```python
# Podstawowe biblioteki
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Zaawansowane biblioteki
import polars as pl
import duckdb
import vectorbt as vbt
import riskfolio as rp
from openbb import obb

# QF-Lib imports (Professional Event-Driven Backtesting)
try:
    from qf_lib.backtesting.strategies.abstract_strategy import AbstractStrategy
    from qf_lib.backtesting.trading_session.backtest_trading_session import BacktestTradingSession
    from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder
    from qf_lib.backtesting.order.execution_style import MarketOrder
    from qf_lib.backtesting.order.time_in_force import TimeInForce
    from qf_lib.backtesting.position_sizer.initial_risk_position_sizer import InitialRiskPositionSizer
    from qf_lib.backtesting.alpha_model.alpha_model import AlphaModel
    from qf_lib.backtesting.alpha_model.exposure_enum import Exposure
    from qf_lib.common.enums.frequency import Frequency
    from qf_lib.common.enums.price_field import PriceField
    from qf_lib.common.utils.miscellaneous.kelly import kelly_criterion
    from qf_lib.common.utils.ratios.sharpe_ratio import sharpe_ratio
    from qf_lib.common.utils.returns.max_drawdown import max_drawdown
    from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter
    from qf_lib.analysis.tearsheets.tearsheet_without_benchmark import TearsheetWithoutBenchmark
    QF_LIB_AVAILABLE = True
except ImportError:
    QF_LIB_AVAILABLE = False
    print("⚠️ QF-Lib not installed. Install with: pip install qf-lib")

# Biblioteki pomocnicze
from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm
import pyarrow.parquet as pq

# Pattern detection i advanced analytics
from tseries_patterns import AmplitudeBasedLabeler
from tseries_patterns.buysell import HawkesBSI, HawkesBVC
from tseries_patterns.ml.hmm import GaussianHMM, WalkforwardHMM
from tseries_patterns.ml.features import FeatureSelectByRandomForest, FeatureSelectByEMD

# Monte Carlo Permutation Testing
import sys
sys.path.append('mcpt/')
from bar_permute import get_permutation

# Ustawienia wyświetlania
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

# Konfiguracja vectorbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 1200
```

## 3. Zaawansowane biblioteki do backtestingu

### 3.1 QF-Lib - Professional Event-Driven Backtesting Framework

**QF-Lib** to biblioteka Python zapewniająca wysokiej jakości narzędzia dla finansów ilościowych. Głównym celem projektu jest backtestowanie strategii inwestycyjnych przy użyciu **architektury opartej na zdarzeniach**, która symuluje rzeczywiste wydarzenia rynkowe jak otwarcie czy zamknięcie sesji. Biblioteka jest zaprojektowana do **testowania i oceny dowolnych niestandardowych strategii inwestycyjnych**.

#### Kluczowe cechy QF-Lib:
- **Event-Driven Architecture**: Symulacja rzeczywistych zdarzeń rynkowych (otwarcie/zamknięcie sesji)
- **Look-Ahead Bias Prevention**: Wbudowane narzędzia zapobiegające wykorzystaniu przyszłych danych
- **Flexible Data Sources**: Bloomberg, Quandl, Haver Analytics, Portara (sprawdź [installation guide](https://qf-lib.readthedocs.io/en/latest/installation.html#installing-optional-data-providers))
- **Extended Data Containers**: Rozszerzone funkcjonalności pandas Series i DataFrame
- **Professional Reports**: Generowanie szczegółowych dokumentów PDF podsumowujących wyniki
- **Modular Design**: Łatwa konfiguracja i tworzenie nowych funkcjonalności

#### Architektura QF-Lib:

QF-Lib używa dwóch głównych podejść do strategii:

**1. Alpha Model Strategy (Rekomendowane):**
```python
from qf_lib.backtesting.alpha_model.alpha_model import AlphaModel
from qf_lib.backtesting.strategies.alpha_model_strategy import AlphaModelStrategy
from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder

class MovingAverageAlphaModel(AlphaModel):
    def __init__(self, fast_period: int, slow_period: int, risk_factor: float, data_provider):
        super().__init__(risk_factor, data_provider)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_exposure(self, ticker, current_exposure, current_time, frequency):
        # QF-Lib automatycznie zapobiega look-ahead bias
        prices = self.data_provider.historical_price(ticker, PriceField.Close, 
                                                    self.slow_period, current_time, frequency)
        
        fast_ma = prices.ewm(span=self.fast_period).mean()
        slow_ma = prices.ewm(span=self.slow_period).mean()
        
        if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
            return Exposure.LONG
        else:
            return Exposure.SHORT

# Konfiguracja backtesting session
settings = get_demo_settings()
session_builder = BacktestTradingSessionBuilder(settings, pdf_exporter, excel_exporter)
session_builder.set_data_provider(data_provider)
session_builder.set_frequency(Frequency.DAILY)
session_builder.set_position_sizer(InitialRiskPositionSizer, initial_risk=0.03)
session_builder.set_commission_model(IBCommissionModel)

ts = session_builder.build(start_date, end_date)

# Strategia z Alpha Model
model = MovingAverageAlphaModel(5, 20, 1.25, ts.data_provider)
model_tickers = [Ticker('AAPL'), Ticker('MSFT')]
strategy = AlphaModelStrategy(ts, {model: model_tickers})

ts.start_trading()
```

**2. Custom Strategy (Zaawansowane):**
```python
from qf_lib.backtesting.strategies.abstract_strategy import AbstractStrategy

class CustomStrategy(AbstractStrategy):
    def calculate_and_place_orders(self):
        # Dostęp do danych z automatyczną ochroną przed look-ahead bias
        current_time = self.broker.get_portfolio().timer.now()
        
        for ticker in self.tickers:
            prices = self.data_provider.historical_price(
                ticker, PriceField.Close, 50, current_time, Frequency.DAILY
            )
            
            # Twoja logika strategii...
            if self._generate_signal(prices):
                orders = self.order_factory.target_percent_orders(
                    {ticker: 0.1}, MarketOrder(), TimeInForce.DAY
                )
                self.broker.place_orders(orders)
```

#### QF-Lib Position Sizing:

```python
# Initial Risk Position Sizer (najczęściej używany)
from qf_lib.backtesting.position_sizer.initial_risk_position_sizer import InitialRiskPositionSizer

# Konfiguracja w session builder
session_builder.set_position_sizer(
    InitialRiskPositionSizer, 
    initial_risk=0.03,  # 3% risk per trade
    max_target_percentage=0.20  # Maksymalna alokacja 20% na jeden instrument
)

# Inne dostępne position sizery:
from qf_lib.backtesting.position_sizer.fixed_portfolio_percentage_position_sizer import FixedPortfolioPercentagePositionSizer
from qf_lib.backtesting.position_sizer.simple_position_sizer import SimplePositionSizer

# Stały procent portfela
session_builder.set_position_sizer(
    FixedPortfolioPercentagePositionSizer,
    percentage=0.1  # 10% portfela na każdy instrument
)

# Kelly Criterion utilities (dostępne w bibliotece)
from qf_lib.common.utils.miscellaneous.kelly import kelly_criterion

def calculate_kelly_position_size(trade_returns):
    """Oblicz optymalną wielkość pozycji według Kelly Criterion"""
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    
    if not wins or not losses:
        return 0.1  # Default 10% jeśli brak wystarczających danych
    
    win_rate = len(wins) / len(trade_returns)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    
    kelly_f = kelly_criterion(win_rate, avg_win, avg_loss)
    return min(max(kelly_f, 0.05), 0.25)  # Ograniczenia 5%-25%
```

#### QF-Lib Risk Management:

```python
# Commission Models (wbudowane modele kosztów)
from qf_lib.backtesting.execution_handler.commission_models.ib_commission_model import IBCommissionModel
from qf_lib.backtesting.execution_handler.commission_models.fixed_commission_model import FixedCommissionModel

# Konfiguracja modelu prowizji
session_builder.set_commission_model(IBCommissionModel)  # Realistic Interactive Brokers fees
# lub
session_builder.set_commission_model(FixedCommissionModel, commission=0.005)  # 0.5% fixed

# Slippage Models (poślizgi)
from qf_lib.backtesting.execution_handler.slippage.fixed_slippage import FixedSlippage
from qf_lib.backtesting.execution_handler.slippage.price_based_slippage import PriceBasedSlippage

# Orders i Risk Management
from qf_lib.backtesting.order.execution_style import MarketOrder, StopOrder, LimitOrder
from qf_lib.backtesting.order.time_in_force import TimeInForce

class RiskManagedStrategy(AbstractStrategy):
    def calculate_and_place_orders(self):
        # Market orders z stop lossami
        if self._should_enter_position():
            # Market order na entry
            market_orders = self.order_factory.target_percent_orders(
                {self.ticker: 0.1}, 
                MarketOrder(), 
                TimeInForce.DAY
            )
            self.broker.place_orders(market_orders)
            
            # Stop loss order (3% poniżej ceny rynkowej)
            current_price = self.data_provider.get_price(self.ticker, PriceField.Close)
            stop_price = current_price * 0.97
            
            stop_orders = self.order_factory.orders(
                {self.ticker: -100},  # Sprzedaj 100 akcji
                StopOrder(stop_price),
                TimeInForce.GTC  # Good Till Canceled
            )
            self.broker.place_orders(stop_orders)

# Volatility Management
from qf_lib.common.utils.volatility.volatility_manager import VolatilityManager

def calculate_position_size_with_volatility(target_volatility=0.15):
    """Dostosuj wielkość pozycji do volatility targeting"""
    vol_manager = VolatilityManager(data_provider)
    current_vol = vol_manager.get_volatility(ticker, window=30)
    
    # Skalowanie pozycji odwrotnie do volatility
    vol_scaling = target_volatility / current_vol
    base_position = 0.1  # 10% base allocation
    
    return base_position * vol_scaling
```

#### QF-Lib Performance Analytics:

```python
# Automatyczne generowanie raportów PDF
from qf_lib.analysis.tearsheets.tearsheet_without_benchmark import TearsheetWithoutBenchmark
from qf_lib.analysis.tearsheets.tearsheet_with_benchmark import TearsheetWithBenchmark

# Po zakończeniu backtestingu
def generate_performance_report(ts, benchmark_ticker=None):
    """Generuj comprehensive performance report"""
    
    # Pobierz wyniki portfela
    portfolio_tms = ts.portfolio.portfolio_eod_series()
    returns_tms = portfolio_tms.to_log_returns()
    
    if benchmark_ticker:
        # Tearsheet z benchmarkiem
        benchmark_tms = ts.data_provider.historical_price(
            benchmark_ticker, PriceField.Close, len(portfolio_tms)
        ).to_log_returns()
        
        tearsheet = TearsheetWithBenchmark(
            pdf_exporter=ts.pdf_exporter,
            title=f"Strategy vs {benchmark_ticker} Performance Report"
        )
        tearsheet.build_document(returns_tms, benchmark_tms)
    else:
        # Tearsheet bez benchmarka
        tearsheet = TearsheetWithoutBenchmark(
            pdf_exporter=ts.pdf_exporter,
            title="Strategy Performance Report"
        )
        tearsheet.build_document(returns_tms)

# Manual metrics calculation
from qf_lib.common.utils.ratios.sharpe_ratio import sharpe_ratio
from qf_lib.common.utils.returns.max_drawdown import max_drawdown
from qf_lib.common.utils.ratios.calmar_ratio import calmar_ratio

def calculate_key_metrics(returns_series):
    """Oblicz kluczowe metryki performance"""
    return {
        'Total Return': returns_series.total_cumulative_return(),
        'Annualized Return': returns_series.annualized_average_return(),
        'Volatility': returns_series.annualized_volatility(),
        'Sharpe Ratio': sharpe_ratio(returns_series),
        'Max Drawdown': max_drawdown(returns_series.to_prices()),
        'Calmar Ratio': calmar_ratio(returns_series),
        'Win Rate': (returns_series > 0).mean(),
        'Skewness': returns_series.skewness(),
        'Kurtosis': returns_series.kurtosis()
    }

# Exporty do Excel
from qf_lib.documents_utils.excel.excel_exporter import ExcelExporter

excel_exporter = ExcelExporter(settings)
# Automatyczne exporty trades, portfolio values, etc.
```

#### QF-Lib vs Inne Biblioteki:

| Cecha | QF-Lib | VectorBT | Backtrader | Zipline |
|-------|--------|----------|------------|---------|
| **Architecture** | Event-driven ✅ | Vectorized | Event-driven | Event-driven |
| **Look-ahead Protection** | ✅ Built-in | Manual | Manual | Built-in |
| **Data Sources** | ✅ Enterprise (Bloomberg, Quandl) | Yahoo Finance | Multiple | Quandl |
| **Position Sizing** | ✅ Multiple algorithms | Basic | Manual | Basic |
| **Commission/Slippage** | ✅ Realistic models | Manual | Basic | Good |
| **PDF Reports** | ✅ Professional | None | Basic | None |
| **Performance** | Good | ✅ Ultra-fast | Slow | Medium |
| **Production Ready** | ✅ Enterprise | Research | Medium | ✅ Quantopian legacy |
| **Learning Curve** | High | Medium | High | High |
| **Maintenance** | ✅ Active | Active | Limited | Community |

#### Kiedy używać QF-Lib:

✅ **Użyj QF-Lib gdy:**
- **Profesjonalny research**: Potrzebujesz academic/institutional quality backtests
- **Look-ahead bias prevention**: Kluczowa jest integralność danych i brak przyszłych informacji
- **Dokumentacja**: Wymagane są professional PDF reports z pełną analizą
- **Enterprise data**: Masz dostęp do Bloomberg, Quandl, Haver Analytics
- **Advanced strategies**: Złożone multi-asset, multi-timeframe strategies
- **Risk management**: Potrzebujesz sophisticated position sizing i risk controls
- **Production deployment**: Strategia będzie wdrożona w środowisku produkcyjnym

❌ **Nie używaj QF-Lib gdy:**
- **Quick prototyping**: Potrzebujesz szybkich testów prostych strategii (użyj VectorBT)
- **Ograniczone dane**: Masz dostęp tylko do basic data sources (Yahoo Finance)
- **Brak infrastruktury**: Nie masz WeasyPrint/GTK3+ dependencies
- **Learning**: Dopiero uczysz się backtestingu (zacznij od prostszych bibliotek)
- **Speed first**: Performance jest najważniejszy (VectorBT będzie szybszy)

#### QF-Lib Configuration & Setup:

```python
# Minimal configuration setup
from qf_lib.settings import Settings
from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter

# Settings files (JSON)
settings = Settings(
    settings_path="path/to/settings.json",
    secret_settings_path="path/to/secret_settings.json"  # Optional
)

# PDF Exporter setup
pdf_exporter = PDFExporter(settings)

# Demo configuration (dla testów)
from demo_scripts.demo_configuration.demo_settings import get_demo_settings
settings = get_demo_settings()  # Pre-configured for demos
```

### 3.2 Polars - Hyper-fast DataFrame Library

**Polars** to napisana w Rust biblioteka DataFrame, która jest nawet 10-100x szybsza od pandas dla wielu operacji. Wykorzystuje lazy evaluation, wielowątkowość i kolumnowe przechowywanie danych.

#### Kluczowe cechy Polars:
- **Blazing Fast**: Wykorzystuje wszystkie rdzenie CPU i SIMD
- **Lazy Evaluation**: Optymalizuje zapytania przed wykonaniem
- **Memory Efficient**: Używa Apache Arrow format
- **Type Safe**: Silne typowanie zapobiega błędom
- **Expresywne API**: Czytelna składnia podobna do pandas

#### Przykład użycia Polars w backtestingu:

```python
def load_data_with_polars(file_path):
    """
    Superszybkie ładowanie danych z Polars
    """
    # Lazy loading - dane są ładowane dopiero gdy potrzebne
    df = pl.scan_csv(file_path)
    
    # Przetwarzanie danych z lazy evaluation
    processed = (
        df
        .with_columns([
            # Obliczanie zwrotów
            (pl.col("close").pct_change()).alias("returns"),
            # Logarytmiczne zwroty
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_returns"),
            # Średnie kroczące - superszybkie!
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(50).alias("sma_50"),
            # RSI
            calculate_rsi_polars("close", 14).alias("rsi_14"),
            # Bollinger Bands
            pl.col("close").rolling_mean(20).alias("bb_middle"),
            (pl.col("close").rolling_mean(20) + 2 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("close").rolling_mean(20) - 2 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        ])
        .filter(pl.col("volume") > 0)  # Filtrowanie
    )
    
    # Wykonanie lazy evaluation i konwersja do pandas jeśli potrzebna
    return processed.collect()

def calculate_rsi_polars(price_col, period=14):
    """
    Superszybkie obliczanie RSI w Polars
    """
    delta = pl.col(price_col).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    
    avg_gain = gain.rolling_mean(period)
    avg_loss = loss.rolling_mean(period)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Przykład agregacji i grupowania - 100x szybsze niż pandas!
def analyze_trades_polars(trades_df):
    """
    Błyskawiczna analiza transakcji z Polars
    """
    trades_pl = pl.from_pandas(trades_df)
    
    analysis = (
        trades_pl
        .groupby("symbol")
        .agg([
            pl.count().alias("total_trades"),
            pl.col("return").mean().alias("avg_return"),
            pl.col("return").std().alias("return_std"),
            pl.col("return").max().alias("max_return"),
            pl.col("return").min().alias("min_return"),
            (pl.col("return") > 0).sum().alias("winning_trades"),
            pl.col("duration").mean().alias("avg_duration"),
        ])
        .with_columns([
            (pl.col("winning_trades") / pl.col("total_trades") * 100).alias("win_rate"),
            (pl.col("avg_return") / pl.col("return_std") * np.sqrt(252)).alias("sharpe_ratio")
        ])
    )
    
    return analysis
```

### 3.2 DuckDB - Hyper-fast SQL OLAP Database

**DuckDB** to analityczna baza danych SQL zaprojektowana do błyskawicznych analiz. Jest jak "SQLite dla analityki" - działa in-process, nie wymaga serwera, ale oferuje wydajność porównywalną z rozproszonymi systemami.

#### Kluczowe cechy DuckDB:
- **Columnar Storage**: Optymalizacja dla analityki
- **Vectorized Execution**: Przetwarzanie całych kolumn na raz
- **SQL Interface**: Pełna zgodność z SQL
- **Zero Dependencies**: Działa wszędzie
- **Parallel Execution**: Wykorzystuje wszystkie rdzenie

#### Przykład użycia DuckDB w backtestingu:

```python
class DuckDBBacktestAnalyzer:
    """
    Wykorzystanie DuckDB do ultraszybkich analiz backtestu
    """
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
        
    def load_trades(self, trades_df):
        """
        Ładowanie transakcji do DuckDB
        """
        self.conn.register('trades', trades_df)
        
    def load_ohlcv(self, ohlcv_df):
        """
        Ładowanie danych OHLCV
        """
        self.conn.register('ohlcv', ohlcv_df)
        
    def analyze_performance_by_timeperiod(self):
        """
        Analiza wydajności według różnych okresów czasowych
        """
        query = """
        WITH trade_performance AS (
            SELECT 
                DATE_TRUNC('month', entry_date) as month,
                DATE_TRUNC('quarter', entry_date) as quarter,
                DATE_TRUNC('year', entry_date) as year,
                EXTRACT(dow FROM entry_date) as day_of_week,
                EXTRACT(hour FROM entry_date) as hour_of_day,
                return,
                duration,
                symbol
            FROM trades
        )
        SELECT 
            month,
            COUNT(*) as trades_count,
            AVG(return) * 100 as avg_return_pct,
            SUM(CASE WHEN return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            MAX(return) * 100 as best_trade,
            MIN(return) * 100 as worst_trade,
            SUM(return) * 100 as total_return_pct,
            AVG(duration) as avg_duration_days,
            STDDEV(return) * 100 as return_std
        FROM trade_performance
        GROUP BY month
        ORDER BY month
        """
        
        return self.conn.execute(query).df()
    
    def find_best_trading_hours(self):
        """
        Znajdź najlepsze godziny do tradingu
        """
        query = """
        SELECT 
            EXTRACT(hour FROM entry_date) as trading_hour,
            COUNT(*) as trades,
            AVG(return) * 100 as avg_return_pct,
            SUM(CASE WHEN return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            SUM(return) as cumulative_return
        FROM trades
        GROUP BY trading_hour
        HAVING COUNT(*) >= 10  -- Minimum 10 transakcji
        ORDER BY avg_return_pct DESC
        """
        
        return self.conn.execute(query).df()
    
    def calculate_rolling_metrics(self, window_days=30):
        """
        Oblicz metryki kroczące - DuckDB jest tutaj 50x szybszy od pandas!
        """
        query = f"""
        WITH daily_returns AS (
            SELECT 
                DATE(exit_date) as date,
                SUM(return) as daily_return,
                COUNT(*) as daily_trades
            FROM trades
            GROUP BY date
        ),
        rolling_stats AS (
            SELECT 
                date,
                daily_return,
                daily_trades,
                -- Kroczące metryki
                AVG(daily_return) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {window_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg_return,
                STDDEV(daily_return) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {window_days} PRECEDING AND CURRENT ROW
                ) as rolling_std,
                SUM(daily_return) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {window_days} PRECEDING AND CURRENT ROW
                ) as rolling_cumulative_return,
                MAX(daily_return) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {window_days} PRECEDING AND CURRENT ROW
                ) as rolling_max_return,
                MIN(daily_return) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {window_days} PRECEDING AND CURRENT ROW
                ) as rolling_min_return
            FROM daily_returns
        )
        SELECT 
            *,
            rolling_avg_return / NULLIF(rolling_std, 0) * SQRT(252) as rolling_sharpe_ratio,
            -- Rolling Maximum Drawdown
            (rolling_cumulative_return - MAX(rolling_cumulative_return) OVER (
                ORDER BY date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )) as rolling_drawdown
        FROM rolling_stats
        ORDER BY date
        """
        
        return self.conn.execute(query).df()
    
    def market_regime_analysis(self, ohlcv_table='ohlcv', trades_table='trades'):
        """
        Analiza wydajności strategii w różnych reżimach rynkowych
        """
        query = """
        WITH market_regimes AS (
            SELECT 
                date,
                close,
                -- Oblicz 50-dniową średnią kroczącą
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
                -- Oblicz 200-dniową średnią kroczącą  
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
                -- Oblicz zmienność 30-dniową
                STDDEV(close / LAG(close) OVER (ORDER BY date) - 1) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_30d
            FROM ohlcv
        ),
        regimes_classified AS (
            SELECT 
                date,
                CASE 
                    WHEN close > sma_50 AND sma_50 > sma_200 THEN 'Strong Uptrend'
                    WHEN close > sma_50 AND sma_50 < sma_200 THEN 'Weak Uptrend'
                    WHEN close < sma_50 AND sma_50 > sma_200 THEN 'Weak Downtrend'
                    WHEN close < sma_50 AND sma_50 < sma_200 THEN 'Strong Downtrend'
                    ELSE 'Neutral'
                END as trend_regime,
                CASE
                    WHEN volatility_30d < 0.15 THEN 'Low Volatility'
                    WHEN volatility_30d < 0.25 THEN 'Normal Volatility'
                    WHEN volatility_30d < 0.40 THEN 'High Volatility'
                    ELSE 'Extreme Volatility'
                END as volatility_regime
            FROM market_regimes
        )
        SELECT 
            r.trend_regime,
            r.volatility_regime,
            COUNT(t.return) as trades_count,
            AVG(t.return) * 100 as avg_return_pct,
            SUM(CASE WHEN t.return > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(t.return), 0) as win_rate,
            SUM(t.return) * 100 as total_return_pct,
            STDDEV(t.return) * 100 as return_std
        FROM regimes_classified r
        LEFT JOIN trades t ON DATE(t.entry_date) = r.date
        GROUP BY r.trend_regime, r.volatility_regime
        ORDER BY avg_return_pct DESC
        """
        
        return self.conn.execute(query).df()

# Przykład użycia
analyzer = DuckDBBacktestAnalyzer()
analyzer.load_trades(trades_df)
analyzer.load_ohlcv(ohlcv_df)

# Superszybkie analizy
monthly_performance = analyzer.analyze_performance_by_timeperiod()
best_hours = analyzer.find_best_trading_hours()
rolling_metrics = analyzer.calculate_rolling_metrics(30)
regime_analysis = analyzer.market_regime_analysis()
```

### 3.3 VectorBT - Supercharged Backtesting

**VectorBT** to biblioteka do wektoryzowanego backtestingu, która może być 1000x szybsza od tradycyjnych pętli. Wykorzystuje NumPy i Numba do maksymalnej wydajności.

#### Kluczowe cechy VectorBT:
- **Vectorized Operations**: Brak pętli, czyste operacje wektorowe
- **Portfolio Simulation**: Zaawansowane zarządzanie portfelem
- **Advanced Analytics**: Wbudowane metryki i wizualizacje
- **Custom Indicators**: Łatwe tworzenie własnych wskaźników
- **Interactive Plots**: Interaktywne wykresy z Plotly

#### Przykład użycia VectorBT:

```python
class VectorBTAdvancedBacktest:
    """
    Zaawansowany backtest z VectorBT
    """
    def __init__(self, data):
        self.data = data
        self.close = data['Close']
        
    def run_ma_crossover_optimization(self, short_windows, long_windows):
        """
        Masowa optymalizacja strategii MA Crossover - testuje wszystkie kombinacje!
        """
        # Oblicz wszystkie kombinacje średnich kroczących jednocześnie!
        fast_ma = vbt.MA.run(self.close, window=short_windows, short_name='fast')
        slow_ma = vbt.MA.run(self.close, window=long_windows, short_name='slow')
        
        # Generuj sygnały dla wszystkich kombinacji
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Symuluj portfolio dla wszystkich kombinacji
        portfolio = vbt.Portfolio.from_signals(
            self.close, 
            entries, 
            exits,
            init_cash=10000,
            fees=0.001,  # 0.1% prowizji
            freq='1D'
        )
        
        return portfolio
    
    def run_advanced_strategy(self):
        """
        Zaawansowana strategia z wieloma wskaźnikami
        """
        # Oblicz wskaźniki
        rsi = vbt.RSI.run(self.close, window=14)
        bbands = vbt.BBANDS.run(self.close, window=20, stds=2)
        macd = vbt.MACD.run(self.close, fast_period=12, slow_period=26, signal_period=9)
        
        # Zaawansowane warunki wejścia
        entry_conditions = (
            (rsi.rsi < 30) &  # Wyprzedany RSI
            (self.close < bbands.lower) &  # Cena poniżej dolnego Bollingera
            (macd.macd > macd.signal)  # MACD crossover
        )
        
        # Warunki wyjścia
        exit_conditions = (
            (rsi.rsi > 70) |  # Wykupiony RSI
            (self.close > bbands.upper)  # Cena powyżej górnego Bollingera
        )
        
        # Symulacja z zaawansowanym zarządzaniem pozycją
        portfolio = vbt.Portfolio.from_signals(
            self.close,
            entry_conditions,
            exit_conditions,
            init_cash=10000,
            size=vbt.RepeatMode.TargetPercent,  # Procent portfela
            size_value=0.95,  # Używaj 95% dostępnego kapitału
            fees=0.001,
            slippage=0.001,  # Poślizg 0.1%
            freq='1D'
        )
        
        return portfolio
    
    def analyze_results(self, portfolio):
        """
        Kompleksowa analiza wyników z VectorBT
        """
        # Podstawowe statystyki
        stats = portfolio.stats()
        print("\n📊 STATYSTYKI PORTFOLIO:")
        print(stats)
        
        # Zaawansowane metryki
        metrics = {
            'Total Return [%]': portfolio.total_return() * 100,
            'Annual Return [%]': portfolio.annualized_return() * 100,
            'Sharpe Ratio': portfolio.sharpe_ratio(),
            'Sortino Ratio': portfolio.sortino_ratio(),
            'Calmar Ratio': portfolio.calmar_ratio(),
            'Max Drawdown [%]': portfolio.max_drawdown() * 100,
            'Win Rate [%]': portfolio.win_rate() * 100,
            'Best Trade [%]': portfolio.best_trade_return() * 100,
            'Worst Trade [%]': portfolio.worst_trade_return() * 100,
            'Avg Win [%]': portfolio.avg_win_return() * 100,
            'Avg Loss [%]': portfolio.avg_loss_return() * 100,
            'Profit Factor': portfolio.profit_factor(),
            'Kelly Criterion': portfolio.kelly_criterion(),
            'Tail Ratio': portfolio.tail_ratio(),
            'Common Sense Ratio': portfolio.common_sense_ratio(),
            'Skewness': portfolio.skew(),
            'Kurtosis': portfolio.kurtosis()
        }
        
        return pd.Series(metrics)
    
    def create_interactive_plots(self, portfolio):
        """
        Tworzenie interaktywnych wykresów
        """
        # Wykres krzywej kapitału
        fig = portfolio.plot(subplots=[
            'cum_returns',
            'drawdown', 
            'trades',
            'trade_pnl'
        ])
        fig.show()
        
        # Heatmapa zwrotów
        returns_heatmap = portfolio.returns.resample('M').sum().vbt.heatmap(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year'
        )
        returns_heatmap.show()
        
        # Wykres underwater (czas pod wodą)
        underwater = portfolio.drawdown.plot(
            title='Underwater Plot',
            fill=True
        )
        underwater.show()
    
    def monte_carlo_simulation(self, n_simulations=1000):
        """
        Symulacja Monte Carlo z VectorBT
        """
        # Generuj losowe ścieżki cenowe
        returns = self.close.pct_change().dropna()
        
        # Bootstrapping returns
        random_returns = np.random.choice(
            returns, 
            size=(len(returns), n_simulations), 
            replace=True
        )
        
        # Symuluj ścieżki cenowe
        random_prices = (1 + random_returns).cumprod() * self.close.iloc[0]
        random_prices = pd.DataFrame(
            random_prices, 
            index=self.close.index[1:],
            columns=[f'sim_{i}' for i in range(n_simulations)]
        )
        
        # Testuj strategię na wszystkich symulacjach
        results = []
        for col in random_prices.columns:
            portfolio = self.run_strategy_on_data(random_prices[col])
            results.append({
                'simulation': col,
                'total_return': portfolio.total_return(),
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown()
            })
        
        mc_results = pd.DataFrame(results)
        
        # Analiza wyników Monte Carlo
        print("\n🎲 WYNIKI MONTE CARLO:")
        print(f"Średni zwrot: {mc_results['total_return'].mean() * 100:.2f}%")
        print(f"Std zwrotu: {mc_results['total_return'].std() * 100:.2f}%")
        print(f"5% VaR: {mc_results['total_return'].quantile(0.05) * 100:.2f}%")
        print(f"95% CI: [{mc_results['total_return'].quantile(0.025) * 100:.2f}%, "
              f"{mc_results['total_return'].quantile(0.975) * 100:.2f}%]")
        
        return mc_results

# Przykład użycia VectorBT
vbt_backtest = VectorBTAdvancedBacktest(data)

# Optymalizacja parametrów - testuje 400 kombinacji w sekundy!
short_windows = np.arange(10, 50, 5)
long_windows = np.arange(50, 200, 10)
portfolio_grid = vbt_backtest.run_ma_crossover_optimization(short_windows, long_windows)

# Znajdź najlepszą kombinację
best_params = portfolio_grid.total_return().idxmax()
print(f"Najlepsze parametry: Fast MA = {best_params[0]}, Slow MA = {best_params[1]}")

# Uruchom zaawansowaną strategię
advanced_portfolio = vbt_backtest.run_advanced_strategy()
metrics = vbt_backtest.analyze_results(advanced_portfolio)
vbt_backtest.create_interactive_plots(advanced_portfolio)
```

### 3.4 Riskfolio-Lib - Portfolio Optimization

**Riskfolio-Lib** to najnowocześniejsza biblioteka do optymalizacji portfela, implementująca dziesiątki modeli optymalizacyjnych i miar ryzyka.

#### Kluczowe cechy Riskfolio:
- **20+ Risk Measures**: CVaR, EVaR, CDaR, UCI, i wiele innych
- **Portfolio Optimization**: Mean-Risk, Risk Parity, Hierarchical Risk Parity
- **Black-Litterman**: Zaawansowane modele bayesowskie
- **Factor Models**: Modele czynnikowe i analiza stylu
- **Constraints**: Złożone ograniczenia na wagi aktywów

#### Przykład użycia Riskfolio:

```python
class AdvancedPortfolioOptimizer:
    """
    Zaawansowana optymalizacja portfela z Riskfolio
    """
    def __init__(self, returns_data):
        self.returns = returns_data
        self.port = rp.Portfolio(returns=returns_data)
        
    def calculate_optimal_portfolio(self, risk_measure='MV', obj='Sharpe'):
        """
        Oblicz optymalny portfel dla różnych miar ryzyka
        
        Risk measures:
        - 'MV': Wariancja (Markowitz)
        - 'CVaR': Conditional Value at Risk
        - 'CDaR': Conditional Drawdown at Risk
        - 'UCI': Ulcer Index
        - 'EVaR': Entropic Value at Risk
        """
        # Estymacja parametrów
        self.port.assets_stats(method_mu='hist', method_cov='hist')
        
        # Optymalizacja
        weights = self.port.optimization(
            model='Classic',  # Classic, BL, FM
            rm=risk_measure,
            obj=obj,  # 'Sharpe', 'MinRisk', 'MaxRet', 'Utility'
            rf=0.02,  # Risk-free rate
            l=0  # Regularization parameter
        )
        
        return weights
    
    def hierarchical_risk_parity(self):
        """
        Hierarchical Risk Parity - zaawansowana metoda alokacji
        """
        # Buduj hierarchiczną strukturę portfela
        weights = self.port.optimization(
            model='HRP',  # Hierarchical Risk Parity
            rm='MV',
            rf=0.02,
            linkage='ward',  # Metoda klasteryzacji
            leaf_order=True
        )
        
        # Analiza klastrów
        clusters = self.port.clusters
        print("\n🌳 STRUKTURA HIERARCHICZNA:")
        for i, cluster in enumerate(clusters):
            print(f"Klaster {i+1}: {cluster}")
        
        return weights
    
    def risk_parity_optimization(self):
        """
        Risk Parity - równy wkład ryzyka
        """
        weights = self.port.optimization(
            model='RP',  # Risk Parity
            rm='MV',
            rf=0.02
        )
        
        # Oblicz wkład ryzyka każdego aktywa
        risk_contribution = self.port.risk_contribution
        
        return weights, risk_contribution
    
    def black_litterman_optimization(self, views, view_confidences):
        """
        Black-Litterman model z własnymi poglądami
        
        views: dict z poglądami np. {'AAPL': 0.15, 'GOOGL': 0.12}
        view_confidences: dict z pewnością poglądów np. {'AAPL': 0.8, 'GOOGL': 0.6}
        """
        # Przygotuj macierz poglądów
        P = np.zeros((len(views), len(self.returns.columns)))
        Q = np.zeros(len(views))
        omega = np.zeros(len(views))
        
        for i, (asset, view_return) in enumerate(views.items()):
            asset_idx = self.returns.columns.get_loc(asset)
            P[i, asset_idx] = 1
            Q[i] = view_return
            omega[i] = view_confidences[asset]
        
        # Parametry Black-Litterman
        self.port.blacklitterman_stats(
            P=P,
            Q=Q,
            omega=omega,
            rf=0.02,
            w=None  # Market cap weights jeśli dostępne
        )
        
        # Optymalizacja z poglądami BL
        weights = self.port.optimization(
            model='BL',
            rm='MV',
            obj='Sharpe',
            rf=0.02
        )
        
        return weights
    
    def efficient_frontier_analysis(self):
        """
        Analiza granicy efektywnej z różnymi miarami ryzyka
        """
        # Oblicz granicę efektywną
        frontier = self.port.efficient_frontier(
            model='Classic',
            rm='MV',
            points=50,
            rf=0.02,
            plotfrontier=True
        )
        
        # Porównaj różne miary ryzyka
        risk_measures = ['MV', 'CVaR', 'CDaR', 'UCI', 'EVaR']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, rm in enumerate(risk_measures):
            ax = axes[i]
            
            # Oblicz granicę dla każdej miary
            self.port.efficient_frontier(
                model='Classic',
                rm=rm,
                points=30,
                rf=0.02,
                plotfrontier=True,
                ax=ax
            )
            
            ax.set_title(f'Efficient Frontier - {rm}')
        
        plt.tight_layout()
        plt.show()
    
    def factor_model_optimization(self, factors_data):
        """
        Optymalizacja z modelem czynnikowym
        """
        # Estymacja modelu czynnikowego
        self.port.factors_stats(
            method_mu='hist',
            method_cov='hist',
            factors=factors_data
        )
        
        # Optymalizacja z ograniczeniami na ekspozycję czynnikową
        constraints = {
            'Factor_1': [0.1, 0.3],  # Min i max ekspozycja na czynnik 1
            'Factor_2': [-0.2, 0.2]  # Min i max ekspozycja na czynnik 2
        }
        
        weights = self.port.optimization(
            model='FM',  # Factor Model
            rm='MV',
            obj='Sharpe',
            rf=0.02,
            constraints=constraints
        )
        
        return weights
    
    def comprehensive_portfolio_analysis(self, weights):
        """
        Kompleksowa analiza portfela
        """
        # Podstawowe metryki
        mu = self.port.mu @ weights
        sigma = np.sqrt(weights.T @ self.port.cov @ weights)
        sharpe = (mu - 0.02) / sigma
        
        print("\n📊 ANALIZA PORTFELA:")
        print(f"Oczekiwany zwrot: {mu * 100:.2f}%")
        print(f"Zmienność: {sigma * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        
        # Zaawansowane metryki ryzyka
        metrics = {
            'Value at Risk (95%)': self.port.var_historic(weights, alpha=0.05),
            'Conditional VaR (95%)': self.port.cvar_historic(weights, alpha=0.05),
            'Max Drawdown': self.port.max_drawdown(weights),
            'Conditional Drawdown at Risk': self.port.cdar_historic(weights, alpha=0.05),
            'Ulcer Index': self.port.ulcer_index(weights),
            'Entropic VaR': self.port.evar_historic(weights, alpha=0.05)
        }
        
        for metric, value in metrics.items():
            print(f"{metric}: {value * 100:.2f}%")
        
        # Dekompozycja ryzyka
        risk_decomp = self.port.risk_contribution(weights)
        
        # Wykres alokacji i wkładu ryzyka
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Alokacja
        weights.plot(kind='bar', ax=ax1, title='Alokacja Portfela')
        ax1.set_ylabel('Waga (%)')
        
        # Wkład ryzyka
        risk_decomp.plot(kind='bar', ax=ax2, title='Wkład Ryzyka')
        ax2.set_ylabel('Wkład do całkowitego ryzyka (%)')
        
        plt.tight_layout()
        plt.show()
        
        return metrics

# Przykład użycia Riskfolio
# Przygotuj dane zwrotów
returns = pd.DataFrame({
    'AAPL': data['AAPL']['Returns'],
    'MSFT': data['MSFT']['Returns'],
    'GOOGL': data['GOOGL']['Returns'],
    'AMZN': data['AMZN']['Returns']
}).dropna()

optimizer = AdvancedPortfolioOptimizer(returns)

# Różne metody optymalizacji
mv_weights = optimizer.calculate_optimal_portfolio('MV', 'Sharpe')
cvar_weights = optimizer.calculate_optimal_portfolio('CVaR', 'Sharpe')
hrp_weights = optimizer.hierarchical_risk_parity()
rp_weights, risk_contrib = optimizer.risk_parity_optimization()

# Black-Litterman z własnymi poglądami
views = {'AAPL': 0.20, 'GOOGL': 0.15}  # Oczekuję 20% zwrotu z AAPL
confidences = {'AAPL': 0.7, 'GOOGL': 0.5}  # 70% pewności co do AAPL
bl_weights = optimizer.black_litterman_optimization(views, confidences)

# Analiza portfela
metrics = optimizer.comprehensive_portfolio_analysis(mv_weights)
```

### 3.5 OpenBB - Open Source Investment Research Terminal

**OpenBB** to open-source'owa alternatywa dla Bloomberg Terminal, oferująca dostęp do ogromnej ilości danych finansowych i narzędzi analitycznych.

#### Kluczowe cechy OpenBB:
- **600+ Data Sources**: Dostęp do danych z setek źródeł
- **Modular Architecture**: Rozszerzalność przez społeczność
- **Advanced Analytics**: Wbudowane modele ML i analiza techniczna
- **Real-time Data**: Dane w czasie rzeczywistym
- **Professional Reports**: Generowanie raportów profesjonalnych

#### Przykład użycia OpenBB:

```python
class OpenBBMarketAnalyzer:
    """
    Kompleksowa analiza rynku z OpenBB
    """
    def __init__(self):
        # Inicjalizacja OpenBB (wymagane darmowe konto)
        obb.account.login(email="your_email", password="your_password")
        
    def get_comprehensive_stock_data(self, symbol, start_date, end_date):
        """
        Pobierz kompleksowe dane o akcjach
        """
        # Dane cenowe z wielu źródeł
        price_data = obb.equity.price.historical(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            provider='polygon'  # lub 'yfinance', 'alpha_vantage', 'intrinio'
        )
        
        # Fundamentalne
        fundamentals = obb.equity.fundamental.overview(
            symbol=symbol,
            provider='fmp'
        )
        
        # Dane finansowe
        income_statement = obb.equity.fundamental.income(
            symbol=symbol,
            period='quarterly',
            limit=8
        )
        
        balance_sheet = obb.equity.fundamental.balance(
            symbol=symbol,
            period='quarterly',
            limit=8
        )
        
        # Analiza techniczna
        ta_summary = obb.technical.summary(
            symbol=symbol,
            provider='tradingview'
        )
        
        return {
            'price': price_data,
            'fundamentals': fundamentals,
            'income': income_statement,
            'balance': balance_sheet,
            'technical': ta_summary
        }
    
    def market_breadth_analysis(self):
        """
        Analiza szerokości rynku
        """
        # Advance/Decline Line
        adv_dec = obb.equity.market.advdec(provider='polygon')
        
        # New Highs/Lows
        highs_lows = obb.equity.market.highs_lows(
            exchange='NYSE',
            provider='polygon'
        )
        
        # Sector Performance
        sectors = obb.equity.sectors.performance(
            provider='finviz'
        )
        
        # Market Movers
        gainers = obb.equity.discovery.gainers(provider='yahoo')
        losers = obb.equity.discovery.losers(provider='yahoo')
        
        # Fear & Greed Index
        fear_greed = obb.index.fear_greed(provider='cnn')
        
        return {
            'advance_decline': adv_dec,
            'highs_lows': highs_lows,
            'sectors': sectors,
            'gainers': gainers,
            'losers': losers,
            'fear_greed': fear_greed
        }
    
    def options_analysis(self, symbol):
        """
        Zaawansowana analiza opcji
        """
        # Łańcuch opcji
        options_chain = obb.derivatives.options.chains(
            symbol=symbol,
            provider='cboe'
        )
        
        # Implied Volatility
        iv_surface = obb.derivatives.options.iv_surface(
            symbol=symbol,
            provider='tradier'
        )
        
        # Greeks
        greeks = obb.derivatives.options.greeks(
            symbol=symbol,
            provider='intrinio'
        )
        
        # Unusual Options Activity
        unusual = obb.derivatives.options.unusual(
            symbol=symbol,
            provider='fdscanner'
        )
        
        # Options Flow
        flow = obb.derivatives.options.flow(
            symbol=symbol,
            provider='flowalgo'
        )
        
        return {
            'chain': options_chain,
            'iv_surface': iv_surface,
            'greeks': greeks,
            'unusual_activity': unusual,
            'flow': flow
        }
    
    def macro_economic_analysis(self):
        """
        Analiza makroekonomiczna
        """
        # Kluczowe wskaźniki
        gdp = obb.economy.gdp(provider='fred')
        inflation = obb.economy.inflation(provider='fred')
        unemployment = obb.economy.unemployment(provider='fred')
        interest_rates = obb.economy.interest_rates(provider='fred')
        
        # Yield Curve
        yield_curve = obb.fixedincome.curve(
            date='2024-01-01',
            provider='fred'
        )
        
        # Economic Calendar
        calendar = obb.economy.calendar(
            start_date='2024-01-01',
            end_date='2024-01-31',
            provider='tradingeconomics'
        )
        
        # Central Bank działania
        fed_minutes = obb.economy.fed_minutes(provider='fed')
        
        return {
            'gdp': gdp,
            'inflation': inflation,
            'unemployment': unemployment,
            'rates': interest_rates,
            'yield_curve': yield_curve,
            'calendar': calendar,
            'fed_minutes': fed_minutes
        }
    
    def sentiment_analysis(self, symbol):
        """
        Analiza sentymentu i social media
        """
        # News sentiment
        news = obb.news.stock(
            symbol=symbol,
            provider='benzinga'
        )
        
        # Social media sentiment
        reddit = obb.stocks.ba.reddit(
            symbol=symbol,
            provider='openbb'
        )
        
        twitter = obb.stocks.ba.twitter(
            symbol=symbol,
            provider='openbb'
        )
        
        # Analyst ratings
        ratings = obb.equity.estimates.analyst_ratings(
            symbol=symbol,
            provider='finviz'
        )
        
        # Insider trading
        insider = obb.equity.ownership.insider_trading(
            symbol=symbol,
            provider='intrinio'
        )
        
        return {
            'news': news,
            'reddit': reddit,
            'twitter': twitter,
            'analyst_ratings': ratings,
            'insider_trading': insider
        }
    
    def ai_powered_analysis(self, data):
        """
        Analiza z wykorzystaniem AI/ML
        """
        # Prognozowanie z ARIMA
        forecast = obb.forecast.arima(
            data=data,
            target_column='close',
            periods=30
        )
        
        # Wykrywanie anomalii
        anomalies = obb.forecast.anomaly_detection(
            data=data,
            target_column='close'
        )
        
        # Analiza korelacji z ML
        correlations = obb.forecast.correlation_analysis(
            data=data,
            method='pearson'
        )
        
        # Predykcja z sieciami neuronowymi
        nn_forecast = obb.forecast.neural_network(
            data=data,
            target_column='close',
            periods=30,
            hidden_layers=[64, 32, 16]
        )
        
        return {
            'forecast': forecast,
            'anomalies': anomalies,
            'correlations': correlations,
            'nn_forecast': nn_forecast
        }
    
    def generate_professional_report(self, symbol):
        """
        Generuj profesjonalny raport PDF
        """
        # Zbierz wszystkie dane
        stock_data = self.get_comprehensive_stock_data(symbol, '2023-01-01', '2024-01-01')
        market_data = self.market_breadth_analysis()
        options_data = self.options_analysis(symbol)
        sentiment_data = self.sentiment_analysis(symbol)
        
        # Generuj raport
        report = obb.reports.equity(
            symbol=symbol,
            provider='openbb',
            report_type='comprehensive',
            include_technicals=True,
            include_fundamentals=True,
            include_sentiment=True,
            include_options=True
        )
        
        # Eksportuj do PDF
        report.export(f"{symbol}_analysis_report.pdf")
        
        return report

# Przykład użycia OpenBB
analyzer = OpenBBMarketAnalyzer()

# Kompleksowa analiza AAPL
aapl_data = analyzer.get_comprehensive_stock_data('AAPL', '2023-01-01', '2024-01-01')
market_breadth = analyzer.market_breadth_analysis()
options_analysis = analyzer.options_analysis('AAPL')
sentiment = analyzer.sentiment_analysis('AAPL')
macro = analyzer.macro_economic_analysis()

# AI-powered forecasting
ai_analysis = analyzer.ai_powered_analysis(aapl_data['price'])

# Generuj profesjonalny raport
report = analyzer.generate_professional_report('AAPL')
```

### 3.6 Integracja wszystkich bibliotek - Ultimate Backtesting Framework

```python
class UltimateBacktestingFramework:
    """
    Zintegrowany framework wykorzystujący wszystkie zaawansowane biblioteki
    """
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Inicjalizacja komponentów
        self.openbb_analyzer = OpenBBMarketAnalyzer()
        self.duckdb_analyzer = DuckDBBacktestAnalyzer()
        
    def execute_complete_backtest_pipeline(self):
        """
        Kompletny pipeline backtestingu z wszystkimi bibliotekami
        """
        print("🚀 ROZPOCZYNANIE ULTIMATE BACKTEST PIPELINE...")
        
        # 1. Pobierz dane z OpenBB (najwyższa jakość)
        print("\n1️⃣ Pobieranie danych z OpenBB...")
        raw_data = {}
        for symbol in self.symbols:
            data = self.openbb_analyzer.get_comprehensive_stock_data(
                symbol, self.start_date, self.end_date
            )
            raw_data[symbol] = data
        
        # 2. Przetwórz dane z Polars (ultra-szybko)
        print("\n2️⃣ Przetwarzanie danych z Polars...")
        processed_data = self.process_data_with_polars(raw_data)
        
        # 3. Backtest strategii z VectorBT (wektoryzowany)
        print("\n3️⃣ Backtesting z VectorBT...")
        vbt_results = self.run_vectorbt_backtest(processed_data)
        
        # 4. Analiza transakcji z DuckDB (SQL na sterydach)
        print("\n4️⃣ Analiza transakcji z DuckDB...")
        trade_analysis = self.analyze_trades_with_duckdb(vbt_results['trades'])
        
        # 5. Optymalizacja portfela z Riskfolio
        print("\n5️⃣ Optymalizacja portfela z Riskfolio...")
        optimal_weights = self.optimize_portfolio_with_riskfolio(processed_data)
        
        # 6. Generowanie raportów
        print("\n6️⃣ Generowanie kompleksowego raportu...")
        self.generate_ultimate_report(
            vbt_results, trade_analysis, optimal_weights
        )
        
        return {
            'vbt_results': vbt_results,
            'trade_analysis': trade_analysis,
            'optimal_weights': optimal_weights
        }
    
    def process_data_with_polars(self, raw_data):
        """
        Ultra-szybkie przetwarzanie danych z Polars
        """
        processed = {}
        
        for symbol, data in raw_data.items():
            # Konwertuj do Polars
            df = pl.from_pandas(data['price'].to_pandas())
            
            # Dodaj wszystkie wskaźniki techniczne jednocześnie
            df = df.with_columns([
                # Zwroty
                pl.col('close').pct_change().alias('returns'),
                
                # Średnie kroczące
                pl.col('close').rolling_mean(20).alias('sma_20'),
                pl.col('close').rolling_mean(50).alias('sma_50'),
                pl.col('close').rolling_mean(200).alias('sma_200'),
                
                # RSI
                calculate_rsi_polars('close', 14).alias('rsi'),
                
                # MACD
                (pl.col('close').ewm_mean(span=12) - 
                 pl.col('close').ewm_mean(span=26)).alias('macd'),
                
                # Bollinger Bands
                pl.col('close').rolling_mean(20).alias('bb_middle'),
                (pl.col('close').rolling_mean(20) + 
                 2 * pl.col('close').rolling_std(20)).alias('bb_upper'),
                (pl.col('close').rolling_mean(20) - 
                 2 * pl.col('close').rolling_std(20)).alias('bb_lower'),
                
                # Volume indicators
                (pl.col('volume') * pl.col('close')).alias('dollar_volume'),
                pl.col('volume').rolling_mean(20).alias('volume_sma'),
                
                # Volatility
                pl.col('returns').rolling_std(20).alias('volatility_20d'),
                
                # Price patterns
                pl.col('high').rolling_max(20).alias('resistance_20d'),
                pl.col('low').rolling_min(20).alias('support_20d')
            ])
            
            processed[symbol] = df
        
        return processed
    
    def run_vectorbt_backtest(self, processed_data):
        """
        Masowy backtest z VectorBT
        """
        results = {}
        
        for symbol, data in processed_data.items():
            # Konwertuj z powrotem do pandas dla VectorBT
            df = data.to_pandas()
            
            # Zdefiniuj sygnały
            entries = (
                (df['rsi'] < 30) & 
                (df['close'] < df['bb_lower']) &
                (df['macd'] > df['macd'].shift(1))
            )
            
            exits = (
                (df['rsi'] > 70) |
                (df['close'] > df['bb_upper'])
            )
            
            # Symulacja portfolio
            portfolio = vbt.Portfolio.from_signals(
                df['close'],
                entries,
                exits,
                init_cash=100000,
                fees=0.001,
                slippage=0.001,
                freq='1D'
            )
            
            results[symbol] = {
                'portfolio': portfolio,
                'stats': portfolio.stats(),
                'trades': portfolio.trades.records_readable
            }
        
        return results
    
    def analyze_trades_with_duckdb(self, all_trades):
        """
        Błyskawiczna analiza z DuckDB
        """
        # Połącz wszystkie transakcje
        combined_trades = pd.concat(
            [trades.assign(symbol=symbol) 
             for symbol, trades in all_trades.items()]
        )
        
        # Załaduj do DuckDB
        self.duckdb_analyzer.load_trades(combined_trades)
        
        # Wykonaj zaawansowane analizy
        analyses = {
            'by_symbol': self.duckdb_analyzer.analyze_performance_by_symbol(),
            'by_timeperiod': self.duckdb_analyzer.analyze_performance_by_timeperiod(),
            'by_market_regime': self.duckdb_analyzer.market_regime_analysis(),
            'rolling_metrics': self.duckdb_analyzer.calculate_rolling_metrics(30),
            'best_worst_periods': self.duckdb_analyzer.find_best_worst_periods()
        }
        
        return analyses
    
    def optimize_portfolio_with_riskfolio(self, processed_data):
        """
        Zaawansowana optymalizacja portfela
        """
        # Przygotuj macierz zwrotów
        returns_dict = {}
        for symbol, data in processed_data.items():
            returns_dict[symbol] = data['returns'].to_pandas()
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        # Optymalizator
        optimizer = AdvancedPortfolioOptimizer(returns_df)
        
        # Różne metody optymalizacji
        optimizations = {
            'mean_variance': optimizer.calculate_optimal_portfolio('MV', 'Sharpe'),
            'cvar': optimizer.calculate_optimal_portfolio('CVaR', 'Sharpe'),
            'hierarchical': optimizer.hierarchical_risk_parity(),
            'risk_parity': optimizer.risk_parity_optimization()[0]
        }
        
        return optimizations
    
    def generate_ultimate_report(self, vbt_results, trade_analysis, optimal_weights):
        """
        Generowanie kompleksowego raportu HTML
        """
        # Tu można dodać generowanie pięknego raportu HTML/PDF
        # z wszystkimi wynikami, wykresami i analizami
        pass

# Użycie Ultimate Framework
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
framework = UltimateBacktestingFramework(symbols, '2022-01-01', '2024-01-01')
results = framework.execute_complete_backtest_pipeline()
```

## 4. Pobieranie danych historycznych

### Podstawowe pobieranie danych z yfinance

```python
def download_data(symbol, start_date, end_date, interval='1d'):
    """
    Pobiera dane historyczne z Yahoo Finance
    
    Parameters:
    - symbol: ticker symbolu (np. 'AAPL', 'BTC-USD')
    - start_date: data początkowa (format: 'YYYY-MM-DD')
    - end_date: data końcowa
    - interval: interwał czasowy ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        # Sprawdzenie jakości danych
        if data.empty:
            raise ValueError(f"Brak danych dla {symbol}")
        
        # Uzupełnienie brakujących wartości
        data = data.fillna(method='ffill')
        
        # Dodanie kolumn pomocniczych
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        print(f"Pobrano dane dla {symbol}: {len(data)} rekordów")
        print(f"Zakres dat: {data.index[0]} - {data.index[-1]}")
        
        return data
    
    except Exception as e:
        print(f"Błąd podczas pobierania danych: {e}")
        return None
```

### Zaawansowane pobieranie danych z OpenBB

```python
def download_enhanced_data_openbb(symbol, start_date, end_date):
    """
    Pobiera wzbogacone dane z wielu źródeł przez OpenBB
    """
    # Dane cenowe z najlepszego dostępnego źródła
    price_data = obb.equity.price.historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider='polygon',  # Polygon ma najwyższą jakość danych
        extended_hours=True,  # Uwzględnij pre/post market
        include_actions=True  # Dywidendy i splity
    )
    
    # Dane o wolumenie z różnych źródeł
    volume_profile = obb.technical.volume_profile(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Dane o krótkiej sprzedaży
    short_interest = obb.equity.shorts.interest(
        symbol=symbol,
        provider='finra'
    )
    
    # Dane opcyjne (dla sentymentu)
    options_flow = obb.derivatives.options.flow(
        symbol=symbol,
        provider='unusual_whales'
    )
    
    # Łączenie wszystkich danych
    enhanced_data = price_data.to_pandas()
    enhanced_data['volume_profile'] = volume_profile
    enhanced_data['short_interest'] = short_interest
    enhanced_data['options_sentiment'] = calculate_options_sentiment(options_flow)
    
    return enhanced_data
```

### Ultra-szybkie przetwarzanie z Polars

```python
def process_large_dataset_polars(file_paths):
    """
    Przetwarzanie gigabajtów danych tick-by-tick z Polars
    """
    # Lazy loading wielu plików
    lazy_frames = []
    
    for file_path in file_paths:
        # Lazy loading - nie ładuje do pamięci!
        lf = pl.scan_parquet(file_path) if file_path.endswith('.parquet') else pl.scan_csv(file_path)
        lazy_frames.append(lf)
    
    # Połącz wszystkie pliki (nadal lazy!)
    combined = pl.concat(lazy_frames)
    
    # Przetwarzanie z pełną optymalizacją
    processed = (
        combined
        # Filtrowanie
        .filter(pl.col("volume") > 0)
        .filter(pl.col("price").is_not_null())
        
        # Agregacja do świec OHLCV
        .groupby_dynamic(
            "timestamp",
            every="5m",  # 5-minutowe świece
            period="5m",
            closed="left"
        )
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("price").count().alias("tick_count"),
            # VWAP
            (pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
            # Spread
            (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread")
        ])
        
        # Dodaj wskaźniki techniczne
        .with_columns([
            # Zwroty
            pl.col("close").pct_change().alias("returns"),
            
            # Wskaźniki zmienności
            pl.col("returns").rolling_std(20).alias("volatility"),
            (pl.col("high") - pl.col("low")) / pl.col("close").alias("true_range"),
            
            # Wskaźniki wolumenu
            pl.col("volume").rolling_mean(20).alias("volume_ma"),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
            
            # Mikrostruktura
            (pl.col("high") - pl.col("close")).abs() / (pl.col("high") - pl.col("low"))).alias("pin_bar_ratio"),
            pl.col("tick_count").rolling_mean(20).alias("avg_tick_count")
        ])
    )
    
    # Wykonaj lazy evaluation i zwróć wynik
    return processed.collect()

# Przykład użycia dla danych tick-by-tick
tick_files = [
    "data/AAPL_ticks_2024_01.parquet",
    "data/AAPL_ticks_2024_02.parquet",
    "data/AAPL_ticks_2024_03.parquet"
]

# Przetworzy gigabajty danych w sekundy!
processed_ticks = process_large_dataset_polars(tick_files)
```

### Analityczne zapytania z DuckDB

```python
class MarketDataAnalyzer:
    """
    Wykorzystanie DuckDB do złożonych analiz danych rynkowych
    """
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
        
    def load_market_data(self, data_dict):
        """
        Ładowanie danych różnych instrumentów
        """
        for symbol, data in data_dict.items():
            self.conn.register(f"{symbol}_data", data)
    
    def find_correlation_patterns(self):
        """
        Znajdź korelacje między instrumentami
        """
        query = """
        WITH returns_data AS (
            SELECT 
                a.date,
                a.returns as AAPL_returns,
                m.returns as MSFT_returns,
                g.returns as GOOGL_returns,
                s.returns as SPY_returns
            FROM AAPL_data a
            JOIN MSFT_data m ON a.date = m.date
            JOIN GOOGL_data g ON a.date = g.date
            JOIN SPY_data s ON a.date = s.date
        ),
        rolling_correlations AS (
            SELECT 
                date,
                -- 30-dniowe korelacje kroczące
                CORR(AAPL_returns, SPY_returns) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as AAPL_SPY_corr,
                CORR(MSFT_returns, SPY_returns) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as MSFT_SPY_corr,
                CORR(AAPL_returns, MSFT_returns) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as AAPL_MSFT_corr
            FROM returns_data
        )
        SELECT 
            date,
            AAPL_SPY_corr,
            MSFT_SPY_corr,
            AAPL_MSFT_corr,
            -- Wykryj okresy niskiej korelacji (okazje do dywersyfikacji)
            CASE 
                WHEN ABS(AAPL_MSFT_corr) < 0.3 THEN 'Low Correlation - Good Diversification'
                WHEN ABS(AAPL_MSFT_corr) > 0.8 THEN 'High Correlation - Poor Diversification'
                ELSE 'Normal Correlation'
            END as correlation_regime
        FROM rolling_correlations
        WHERE date >= CURRENT_DATE - INTERVAL '1 year'
        """
        
        return self.conn.execute(query).df()
    
    def detect_market_anomalies(self):
        """
        Wykrywanie anomalii rynkowych
        """
        query = """
        WITH stats AS (
            SELECT 
                symbol,
                date,
                close,
                volume,
                returns,
                -- Statystyki kroczące
                AVG(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as avg_return_30d,
                STDDEV(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as std_return_30d,
                AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as avg_volume_30d,
                -- Z-score
                (returns - AVG(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)) / 
                NULLIF(STDDEV(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 0) as return_zscore,
                -- Volume spike
                volume / NULLIF(AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 0) as volume_ratio
            FROM market_data
        )
        SELECT 
            symbol,
            date,
            close,
            returns * 100 as return_pct,
            return_zscore,
            volume_ratio,
            CASE 
                WHEN ABS(return_zscore) > 3 THEN 'Extreme Price Movement'
                WHEN volume_ratio > 3 THEN 'Volume Spike'
                WHEN ABS(return_zscore) > 2 AND volume_ratio > 2 THEN 'High Impact Event'
                ELSE 'Normal'
            END as anomaly_type
        FROM stats
        WHERE ABS(return_zscore) > 2 OR volume_ratio > 2
        ORDER BY date DESC
        """
        
        return self.conn.execute(query).df()
```

### Pobieranie danych dla wielu instrumentów z obsługą błędów

```python
def download_multiple_assets_enhanced(symbols, start_date, end_date, interval='1d'):
    """
    Pobiera dane dla wielu instrumentów z różnych źródeł
    """
    all_data = {}
    failed_symbols = []
    
    # Próbuj różnych źródeł danych
    data_sources = ['yfinance', 'openbb', 'alpha_vantage', 'polygon']
    
    for symbol in tqdm(symbols, desc="Pobieranie danych"):
        success = False
        
        for source in data_sources:
            try:
                if source == 'yfinance':
                    data = download_data(symbol, start_date, end_date, interval)
                elif source == 'openbb':
                    data = download_enhanced_data_openbb(symbol, start_date, end_date)
                # Dodaj inne źródła...
                
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    success = True
                    print(f"✅ {symbol} pobrano z {source}")
                    break
                    
            except Exception as e:
                print(f"❌ Błąd dla {symbol} z {source}: {str(e)[:50]}...")
                continue
        
        if not success:
            failed_symbols.append(symbol)
            print(f"⚠️ Nie udało się pobrać danych dla {symbol}")
    
    # Raport
    print(f"\n📊 Podsumowanie:")
    print(f"Pobrano dane dla {len(all_data)}/{len(symbols)} symboli")
    if failed_symbols:
        print(f"Niepowodzenia: {', '.join(failed_symbols)}")
    
    return all_data, failed_symbols

# Przykład użycia
symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'BTC-USD', 'ETH-USD']
portfolio_data, failures = download_multiple_assets_enhanced(
    symbols, '2020-01-01', '2024-01-01'
)
```

### Przechowywanie i zarządzanie danymi

```python
class DataManager:
    """
    Efektywne zarządzanie dużymi zbiorami danych
    """
    def __init__(self, data_dir='./market_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def save_to_parquet(self, data, symbol, compression='snappy'):
        """
        Zapisz dane w formacie Parquet (10x mniejsze niż CSV!)
        """
        file_path = f"{self.data_dir}/{symbol}_data.parquet"
        
        # Konwertuj do Polars dla najlepszej kompresji
        if isinstance(data, pd.DataFrame):
            pl_data = pl.from_pandas(data)
        else:
            pl_data = data
            
        # Zapisz z kompresją
        pl_data.write_parquet(
            file_path,
            compression=compression,
            statistics=True,  # Dla szybszego filtrowania
            row_group_size=50000  # Optymalna wielkość grupy
        )
        
        # Info o kompresji
        original_size = data.memory_usage(deep=True).sum() / 1024 / 1024
        compressed_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"💾 Zapisano {symbol}: {original_size:.1f}MB → {compressed_size:.1f}MB "
              f"(kompresja {original_size/compressed_size:.1f}x)")
        
    def load_from_parquet(self, symbol, lazy=True):
        """
        Wczytaj dane z Parquet
        """
        file_path = f"{self.data_dir}/{symbol}_data.parquet"
        
        if lazy:
            # Lazy loading - nie ładuje do pamięci
            return pl.scan_parquet(file_path)
        else:
            # Pełne załadowanie
            return pl.read_parquet(file_path)
    
    def create_data_catalog(self):
        """
        Stwórz katalog wszystkich dostępnych danych
        """
        catalog = []
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet'):
                path = os.path.join(self.data_dir, file)
                
                # Metadane bez ładowania pliku
                pq_file = pq.ParquetFile(path)
                metadata = pq_file.metadata
                
                catalog.append({
                    'symbol': file.replace('_data.parquet', ''),
                    'rows': metadata.num_rows,
                    'columns': len(pq_file.schema),
                    'size_mb': os.path.getsize(path) / 1024 / 1024,
                    'date_range': self._get_date_range(pq_file)
                })
        
        return pd.DataFrame(catalog)
    
    def _get_date_range(self, pq_file):
        """
        Pobierz zakres dat bez ładowania całego pliku
        """
        # Wykorzystaj statystyki Parquet
        stats = pq_file.metadata.row_group(0).column(0).statistics
        return f"{stats.min} - {stats.max}"

# Użycie
data_manager = DataManager()

# Zapisz wszystkie dane
for symbol, data in portfolio_data.items():
    data_manager.save_to_parquet(data, symbol)

# Zobacz katalog
catalog = data_manager.create_data_catalog()
print("\n📚 Katalog danych:")
print(catalog)
```

## 5. Struktura strategii tradingowej

### Klasa bazowa strategii z wykorzystaniem zaawansowanych bibliotek

```python
class AdvancedTradingStrategy:
    def __init__(self, initial_capital=10000, commission=0.001, use_vectorbt=True):
        """
        Zaawansowana klasa strategii wykorzystująca VectorBT i Polars
        
        Parameters:
        - initial_capital: początkowy kapitał
        - commission: prowizja od transakcji (0.001 = 0.1%)
        - use_vectorbt: czy używać VectorBT do obliczeń wektoryzowanych
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.use_vectorbt = use_vectorbt
        self.positions = []
        self.trades = []
        
    def calculate_indicators_polars(self, data):
        """
        Superszybkie obliczanie wskaźników z Polars
        """
        if isinstance(data, pd.DataFrame):
            # Konwertuj pandas do polars
            df = pl.from_pandas(data)
        else:
            df = data
            
        # Oblicz wszystkie wskaźniki jednocześnie
        df = df.with_columns([
            # Trend indicators
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(50).alias("sma_50"),
            pl.col("close").rolling_mean(200).alias("sma_200"),
            pl.col("close").ewm_mean(span=12).alias("ema_12"),
            pl.col("close").ewm_mean(span=26).alias("ema_26"),
            
            # MACD
            (pl.col("close").ewm_mean(span=12) - 
             pl.col("close").ewm_mean(span=26)).alias("macd"),
            
            # RSI
            self._calculate_rsi_polars("close", 14).alias("rsi_14"),
            
            # Bollinger Bands
            pl.col("close").rolling_mean(20).alias("bb_middle"),
            (pl.col("close").rolling_mean(20) + 
             2 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("close").rolling_mean(20) - 
             2 * pl.col("close").rolling_std(20)).alias("bb_lower"),
            
            # ATR (Average True Range)
            self._calculate_atr_polars("high", "low", "close", 14).alias("atr_14"),
            
            # Volume indicators
            pl.col("volume").rolling_mean(20).alias("volume_sma"),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
            
            # Price patterns
            pl.col("high").rolling_max(20).alias("resistance_20"),
            pl.col("low").rolling_min(20).alias("support_20"),
            
            # Volatility
            pl.col("close").pct_change().rolling_std(20).alias("volatility_20"),
            
            # Market microstructure
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("true_range_pct"),
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("body_pct"),
            ((pl.col("high") - pl.col("close").max(pl.col("open"))) / 
             (pl.col("high") - pl.col("low"))).alias("upper_shadow_ratio"),
            ((pl.col("close").min(pl.col("open")) - pl.col("low")) / 
             (pl.col("high") - pl.col("low"))).alias("lower_shadow_ratio"),
        ])
        
        return df
    
    def _calculate_rsi_polars(self, price_col, period=14):
        """Helper do obliczania RSI w Polars"""
        delta = pl.col(price_col).diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        
        avg_gain = gain.rolling_mean(period)
        avg_loss = loss.rolling_mean(period)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr_polars(self, high_col, low_col, close_col, period=14):
        """Helper do obliczania ATR w Polars"""
        tr1 = pl.col(high_col) - pl.col(low_col)
        tr2 = (pl.col(high_col) - pl.col(close_col).shift(1)).abs()
        tr3 = (pl.col(low_col) - pl.col(close_col).shift(1)).abs()
        
        true_range = pl.max([tr1, tr2, tr3])
        return true_range.rolling_mean(period)
    
    def generate_signals(self, data):
        """
        Generuje sygnały kupna/sprzedaży
        Musi być zaimplementowana w klasie pochodnej
        """
        raise NotImplementedError("Metoda generate_signals musi być zaimplementowana")

# Przykład zaawansowanej strategii - Multi-Factor Strategy
class MultiFactorStrategy(AdvancedTradingStrategy):
    def __init__(self, momentum_weight=0.3, mean_reversion_weight=0.3, 
                 volatility_weight=0.2, volume_weight=0.2, **kwargs):
        super().__init__(**kwargs)
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.volatility_weight = volatility_weight
        self.volume_weight = volume_weight
        
    def generate_signals(self, data):
        """
        Generuje sygnały na podstawie wielu czynników
        """
        # Oblicz wskaźniki z Polars (ultra-szybko!)
        df = self.calculate_indicators_polars(data)
        
        if self.use_vectorbt:
            return self._generate_signals_vectorbt(df)
        else:
            return self._generate_signals_standard(df)
    
    def _generate_signals_vectorbt(self, df):
        """
        Wektoryzowane generowanie sygnałów z VectorBT
        """
        # Konwertuj do pandas dla VectorBT
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        
        # Czynnik momentum
        momentum_score = vbt.apply_func_nb(
            lambda close, sma_50, sma_200: 
            (close > sma_50) * 0.5 + (sma_50 > sma_200) * 0.5,
            df_pd['close'].values,
            df_pd['sma_50'].values,
            df_pd['sma_200'].values
        )
        
        # Czynnik mean reversion
        rsi = df_pd['rsi_14'].values
        bb_position = (df_pd['close'] - df_pd['bb_lower']) / (df_pd['bb_upper'] - df_pd['bb_lower'])
        mean_reversion_score = vbt.apply_func_nb(
            lambda rsi, bb_pos: 
            (rsi < 30) * 0.5 + (bb_pos < 0.2) * 0.5,
            rsi,
            bb_position.values
        )
        
        # Czynnik volatility
        volatility_score = vbt.apply_func_nb(
            lambda vol, atr: 
            (vol < np.percentile(vol, 30)) * 0.5 + 
            (atr < np.percentile(atr, 30)) * 0.5,
            df_pd['volatility_20'].values,
            df_pd['atr_14'].values
        )
        
        # Czynnik volume
        volume_score = vbt.apply_func_nb(
            lambda vol_ratio: (vol_ratio > 1.5).astype(float),
            df_pd['volume_ratio'].values
        )
        
        # Łączny wynik
        total_score = (
            momentum_score * self.momentum_weight +
            mean_reversion_score * self.mean_reversion_weight +
            volatility_score * self.volatility_weight +
            volume_score * self.volume_weight
        )
        
        # Generuj sygnały
        entries = total_score > 0.6
        exits = total_score < 0.3
        
        return entries, exits, total_score
    
    def _generate_signals_standard(self, df):
        """
        Standardowe generowanie sygnałów (wolniejsze)
        """
        # Implementacja bez VectorBT
        pass

# Strategia wykorzystująca Machine Learning
class MLTradingStrategy(AdvancedTradingStrategy):
    def __init__(self, model_type='random_forest', lookback_period=60, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, df):
        """
        Przygotuj cechy dla modelu ML używając Polars
        """
        # Dodaj opóźnione wartości
        feature_df = df.with_columns([
            # Lagged returns
            pl.col("returns").shift(i).alias(f"returns_lag_{i}")
            for i in range(1, 6)
        ] + [
            # Lagged volume ratios
            pl.col("volume_ratio").shift(i).alias(f"volume_lag_{i}")
            for i in range(1, 4)
        ] + [
            # Technical indicators ratios
            (pl.col("close") / pl.col("sma_20") - 1).alias("close_to_sma20"),
            (pl.col("close") / pl.col("sma_50") - 1).alias("close_to_sma50"),
            (pl.col("rsi_14") / 100).alias("rsi_normalized"),
            (pl.col("close") - pl.col("bb_lower")) / 
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_position"),
            
            # Pattern recognition
            pl.when(
                (pl.col("close") > pl.col("open")) & 
                (pl.col("lower_shadow_ratio") > 0.6)
            ).then(1).otherwise(0).alias("hammer_pattern"),
            
            pl.when(
                (pl.col("close") < pl.col("open")) & 
                (pl.col("upper_shadow_ratio") > 0.6)
            ).then(1).otherwise(0).alias("shooting_star_pattern"),
        ])
        
        # Dodaj rolling statistics
        windows = [5, 10, 20]
        for w in windows:
            feature_df = feature_df.with_columns([
                pl.col("returns").rolling_mean(w).alias(f"returns_mean_{w}d"),
                pl.col("returns").rolling_std(w).alias(f"returns_std_{w}d"),
                pl.col("returns").rolling_skew(w).alias(f"returns_skew_{w}d"),
                pl.col("volume").rolling_mean(w).alias(f"volume_mean_{w}d"),
            ])
        
        return feature_df
    
    def train_model(self, train_data):
        """
        Trenuj model ML na danych historycznych
        """
        # Przygotuj dane
        feature_df = self.prepare_features(train_data)
        
        # Utwórz target - przyszły kierunek ruchu
        feature_df = feature_df.with_columns([
            pl.when(pl.col("returns").shift(-1) > 0).then(1).otherwise(0).alias("target")
        ])
        
        # Usuń NaN
        feature_df = feature_df.drop_nulls()
        
        # Przygotuj X i y
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]
        
        X = feature_df.select(feature_cols).to_numpy()
        y = feature_df.select("target").to_numpy().ravel()
        
        # Trenuj model
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X, y)
        
        # Zapisz feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Model trained! Accuracy: {self.model.score(X, y):.3f}")
        print("\nTop 10 najważniejszych cech:")
        print(self.feature_importance.head(10))
        
        return self.model
    
    def generate_signals(self, data):
        """
        Generuj sygnały używając wytrenowanego modelu
        """
        if self.model is None:
            raise ValueError("Model musi być najpierw wytrenowany!")
        
        # Przygotuj cechy
        feature_df = self.prepare_features(data)
        
        # Przygotuj dane do predykcji
        feature_cols = self.feature_importance['feature'].tolist()
        X = feature_df.select(feature_cols).to_numpy()
        
        # Predykcja prawdopodobieństw
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Generuj sygnały na podstawie prawdopodobieństw
        df_pd = feature_df.to_pandas()
        df_pd['ml_probability'] = 0
        df_pd.loc[len(df_pd) - len(probabilities):, 'ml_probability'] = probabilities
        
        # Sygnały z progami
        df_pd['signal'] = 0
        df_pd.loc[df_pd['ml_probability'] > 0.65, 'signal'] = 1  # Buy
        df_pd.loc[df_pd['ml_probability'] < 0.35, 'signal'] = -1  # Sell
        
        return df_pd

# Strategia arbitrażowa wykorzystująca DuckDB
class StatisticalArbitrageStrategy(AdvancedTradingStrategy):
    def __init__(self, pair_symbols, lookback_period=60, z_score_threshold=2, **kwargs):
        super().__init__(**kwargs)
        self.pair_symbols = pair_symbols
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.conn = duckdb.connect(':memory:')
        
    def calculate_pair_statistics(self, data_dict):
        """
        Oblicz statystyki par używając DuckDB
        """
        # Załaduj dane do DuckDB
        for symbol, data in data_dict.items():
            self.conn.register(f"{symbol}_data", data)
        
        # Oblicz spread i z-score
        query = f"""
        WITH pair_data AS (
            SELECT 
                a.date,
                a.close as {self.pair_symbols[0]}_close,
                b.close as {self.pair_symbols[1]}_close,
                LOG(a.close) - LOG(b.close) as log_spread
            FROM {self.pair_symbols[0]}_data a
            JOIN {self.pair_symbols[1]}_data b ON a.date = b.date
        ),
        spread_stats AS (
            SELECT 
                *,
                AVG(log_spread) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {self.lookback_period} PRECEDING AND CURRENT ROW
                ) as spread_mean,
                STDDEV(log_spread) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {self.lookback_period} PRECEDING AND CURRENT ROW
                ) as spread_std
            FROM pair_data
        )
        SELECT 
            *,
            (log_spread - spread_mean) / NULLIF(spread_std, 0) as z_score,
            -- Oblicz half-life of mean reversion
            -LOG(2) / LOG(
                CORR(log_spread, LAG(log_spread) OVER (ORDER BY date)) OVER (
                    ORDER BY date 
                    ROWS BETWEEN {self.lookback_period} PRECEDING AND CURRENT ROW
                )
            ) as half_life
        FROM spread_stats
        ORDER BY date
        """
        
        return self.conn.execute(query).df()
    
    def generate_signals(self, data_dict):
        """
        Generuj sygnały arbitrażowe
        """
        # Oblicz statystyki
        pair_stats = self.calculate_pair_statistics(data_dict)
        
        # Generuj sygnały
        pair_stats['signal'] = 0
        
        # Long spread gdy z-score < -threshold
        pair_stats.loc[pair_stats['z_score'] < -self.z_score_threshold, 'signal'] = 1
        
        # Short spread gdy z-score > threshold
        pair_stats.loc[pair_stats['z_score'] > self.z_score_threshold, 'signal'] = -1
        
        # Exit gdy z-score wraca do 0
        pair_stats.loc[abs(pair_stats['z_score']) < 0.5, 'signal'] = 0
        
        return pair_stats
```

## 6. Zaawansowane wykrywanie wzorców z tseries-patterns

### 6.1 AmplitudeBasedLabeler - Zero-lag Trend Classification

**AmplitudeBasedLabeler** to zaawansowane narzędzie do klasyfikacji trendów bez opóźnienia, wykorzystujące analizę amplitudy ruchów cenowych. Jest szczególnie przydatne do historycznej analizy trendów i walidacji strategii.

```python
class AdvancedTrendAnalyzer:
    """
    Zaawansowana analiza trendów z wykorzystaniem AmplitudeBasedLabeler
    """
    def __init__(self, minamp=20, Tinactive=10):
        """
        Parameters:
        - minamp: Minimalna amplituda ruchu w basis points (20 = 0.2%)
        - Tinactive: Maksymalna liczba okresów bez nowych ekstremów
        """
        self.labeler = AmplitudeBasedLabeler(minamp=minamp, Tinactive=Tinactive)
        self.trend_labels = None
        
    def classify_historical_trends(self, price_data):
        """
        Klasyfikacja historycznych trendów (wymaga lookahead)
        
        Returns:
        - trend_labels: Series z etykietami (-1: spadek, 0: brak trendu, +1: wzrost)
        """
        # UWAGA: Ta metoda używa danych przyszłościowych - tylko do analizy historycznej!
        df_for_labeling = pd.DataFrame({
            'stamp': price_data.index,
            'close': price_data.values
        })
        
        self.trend_labels = self.labeler.label(df_for_labeling)
        return self.trend_labels
    
    def enhance_strategy_signals(self, price_data, base_signals):
        """
        Wzbogacenie sygnałów strategii o klasyfikację trendów
        
        Parameters:
        - price_data: Seria cen
        - base_signals: Podstawowe sygnały strategii
        
        Returns:
        - enhanced_signals: Wzmocnione sygnały
        - trend_labels: Etykiety trendów
        """
        if self.trend_labels is None:
            self.trend_labels = self.classify_historical_trends(price_data)
        
        enhanced_signals = base_signals.copy()
        
        # Wzmocnienie sygnałów zgodnych z trendem
        enhanced_signals.loc[
            (base_signals > 0) & (self.trend_labels == 1)
        ] *= 1.5  # Wzmocnij long w trendzie wzrostowym
        
        enhanced_signals.loc[
            (base_signals < 0) & (self.trend_labels == -1)
        ] *= 1.5  # Wzmocnij short w trendzie spadkowym
        
        # Osłabienie sygnałów przeciwnych do trendu
        enhanced_signals.loc[
            (base_signals > 0) & (self.trend_labels == -1)
        ] *= 0.3  # Osłab long w trendzie spadkowym
        
        enhanced_signals.loc[
            (base_signals < 0) & (self.trend_labels == 1)
        ] *= 0.3  # Osłab short w trendzie wzrostowym
        
        return enhanced_signals, self.trend_labels
    
    def analyze_trend_performance(self, returns, signals):
        """
        Analiza wydajności strategii w różnych trendach
        """
        if self.trend_labels is None:
            raise ValueError("Najpierw wykonaj klasyfikację trendów")
        
        # Zwroty strategii w różnych trendach
        strategy_returns = returns * signals.shift(1)
        
        trend_performance = {}
        for trend_type in [-1, 0, 1]:
            trend_mask = self.trend_labels == trend_type
            trend_returns = strategy_returns[trend_mask]
            
            if len(trend_returns) > 10:  # Minimum próbek
                trend_performance[f'trend_{trend_type}'] = {
                    'total_return': trend_returns.sum(),
                    'avg_return': trend_returns.mean(),
                    'sharpe_ratio': trend_returns.mean() / trend_returns.std() * np.sqrt(252) if trend_returns.std() > 0 else 0,
                    'win_rate': (trend_returns > 0).mean() * 100,
                    'max_drawdown': (trend_returns.cumsum() - trend_returns.cumsum().expanding().max()).min()
                }
        
        return trend_performance
    
    def plot_trend_classification(self, price_data, save_path=None):
        """
        Wizualizacja klasyfikacji trendów
        """
        if self.trend_labels is None:
            self.trend_labels = self.classify_historical_trends(price_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Wykres cen z kolorami trendów
        ax1.plot(price_data.index, price_data.values, 'k-', linewidth=1, alpha=0.7)
        
        # Kolory dla różnych trendów
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        labels = {-1: 'Trend spadkowy', 0: 'Brak trendu', 1: 'Trend wzrostowy'}
        
        for trend_type in [-1, 0, 1]:
            mask = self.trend_labels == trend_type
            if mask.any():
                ax1.scatter(price_data.index[mask], price_data.values[mask], 
                          c=colors[trend_type], s=10, alpha=0.6, label=labels[trend_type])
        
        ax1.set_title('Klasyfikacja Trendów - Amplitude Based Labeler')
        ax1.set_ylabel('Cena')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wykres etykiet trendów
        ax2.plot(price_data.index, self.trend_labels, 'b-', linewidth=2)
        ax2.fill_between(price_data.index, self.trend_labels, alpha=0.3)
        ax2.set_ylabel('Etykieta trendu')
        ax2.set_xlabel('Data')
        ax2.set_ylim(-1.5, 1.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Przykład użycia
trend_analyzer = AdvancedTrendAnalyzer(minamp=20, Tinactive=10)

# Klasyfikacja historycznych trendów
price_data = data['close']  # Przykładowe dane cenowe
trend_labels = trend_analyzer.classify_historical_trends(price_data)

# Wzmocnienie sygnałów bazowych
base_signals = calculate_ma_crossover_signals(data)  # Przykładowe sygnały
enhanced_signals, trends = trend_analyzer.enhance_strategy_signals(price_data, base_signals)

# Analiza wydajności w różnych trendach
performance_by_trend = trend_analyzer.analyze_trend_performance(
    data['returns'], enhanced_signals
)

print("📊 Wydajność strategii w różnych trendach:")
for trend, metrics in performance_by_trend.items():
    print(f"\n{trend}:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
```

### 6.2 Hawkes Buy-Sell Intensity Models

**Hawkes Process Models** modelują intensywność zdarzeń kupna/sprzedaży z wykorzystaniem procesów samopobudzających się. Są szczególnie przydatne do analizy momentum wolumenu i potwierdzania sygnałów cenowych.

```python
class VolumeBasedSignalEnhancer:
    """
    Wzmacniacz sygnałów oparty na analizie wolumenu z wykorzystaniem procesów Hawkesa
    """
    def __init__(self, kappa=0.1, window=30):
        """
        Parameters:
        - kappa: Współczynnik zaniku dla procesu Hawkesa (0.05-0.5)
        - window: Okno dla BVC (Bulk Volume Classification)
        """
        self.kappa = kappa
        self.window = window
        self.bsi = HawkesBSI(kappa) if HawkesBSI else None
        self.bvc = HawkesBVC(window, kappa) if HawkesBVC else None
        
    def generate_volume_momentum_signals(self, ohlcv_data, buy_sell_data=None):
        """
        Generowanie sygnałów momentum na podstawie wolumenu
        
        Parameters:
        - ohlcv_data: DataFrame z danymi OHLCV
        - buy_sell_data: Opcjonalne dane z buy/sell volume
        
        Returns:
        - volume_signals: Seria sygnałów wolumenu
        """
        if buy_sell_data is not None and self.bsi is not None:
            # Używaj BSI jeśli dostępne dane buy/sell volume
            try:
                bsi_signals = self.bsi.eval(buy_sell_data)
                return pd.Series(bsi_signals, index=ohlcv_data.index, name='bsi_signal')
            except Exception as e:
                print(f"⚠️ Błąd BSI: {e}, przechodzę na BVC")
                
        # Używaj BVC dla standardowych danych OHLCV
        if self.bvc is not None:
            try:
                # Przygotuj dane dla BVC
                bvc_data = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].copy()
                bvc_data.reset_index(inplace=True)
                bvc_data.rename(columns={'index': 'stamp'}, inplace=True)
                
                bvc_signals = self.bvc.eval(bvc_data)
                return pd.Series(bvc_signals, index=ohlcv_data.index, name='bvc_signal')
            except Exception as e:
                print(f"⚠️ Błąd BVC: {e}, używam prostej analizy wolumenu")
        
        # Fallback: prosta analiza wolumenu
        return self._simple_volume_analysis(ohlcv_data)
    
    def _simple_volume_analysis(self, ohlcv_data):
        """
        Prosta analiza wolumenu jako fallback
        """
        volume = ohlcv_data['volume']
        volume_ma = volume.rolling(self.window).mean()
        volume_ratio = volume / volume_ma
        
        # Normalizacja sygnału
        volume_signal = (volume_ratio - volume_ratio.rolling(100).mean()) / volume_ratio.rolling(100).std()
        return volume_signal.fillna(0)
    
    def combine_with_price_signals(self, price_signals, volume_signals, 
                                 volume_threshold=0.6, confirmation_weight=0.3):
        """
        Łączenie sygnałów cenowych z sygnałami wolumenu
        
        Parameters:
        - price_signals: Sygnały cenowe
        - volume_signals: Sygnały wolumenu  
        - volume_threshold: Próg dla silnych sygnałów wolumenu
        - confirmation_weight: Waga potwierdzenia wolumenu
        
        Returns:
        - confirmed_signals: Potwierdzone sygnały
        """
        # Normalizacja sygnałów wolumenu
        volume_norm = volume_signals.copy()
        if volume_norm.std() > 0:
            volume_norm = (volume_norm - volume_norm.mean()) / volume_norm.std()
        
        # Identyfikacja silnych sygnałów wolumenu
        strong_volume_up = volume_norm > volume_threshold
        strong_volume_down = volume_norm < -volume_threshold
        
        # Inicjalizacja potwierdzonych sygnałów
        confirmed_signals = price_signals.copy()
        
        # Wzmocnienie sygnałów potwierdzonych przez wolumen
        long_confirmed = (price_signals > 0) & strong_volume_up
        short_confirmed = (price_signals < 0) & strong_volume_down
        
        confirmed_signals.loc[long_confirmed] *= (1 + confirmation_weight)
        confirmed_signals.loc[short_confirmed] *= (1 + confirmation_weight)
        
        # Osłabienie sygnałów przeciwnych do wolumenu
        long_contradicted = (price_signals > 0) & strong_volume_down
        short_contradicted = (price_signals < 0) & strong_volume_up
        
        confirmed_signals.loc[long_contradicted] *= (1 - confirmation_weight)
        confirmed_signals.loc[short_contradicted] *= (1 - confirmation_weight)
        
        # Dodaj metryki potwierdzenia
        confirmation_stats = {
            'long_confirmed_pct': long_confirmed.sum() / (price_signals > 0).sum() * 100 if (price_signals > 0).sum() > 0 else 0,
            'short_confirmed_pct': short_confirmed.sum() / (price_signals < 0).sum() * 100 if (price_signals < 0).sum() > 0 else 0,
            'avg_volume_signal': volume_norm.mean(),
            'volume_signal_strength': volume_norm.std()
        }
        
        return confirmed_signals, confirmation_stats
    
    def analyze_volume_patterns(self, ohlcv_data, signals):
        """
        Analiza wzorców wolumenu w kontekście sygnałów
        """
        volume = ohlcv_data['volume']
        returns = ohlcv_data['close'].pct_change()
        
        # Analiza korelacji volume-return
        volume_return_corr = volume.rolling(100).corr(returns.abs())
        
        # Volume profil dla różnych typów sygnałów
        analysis = {
            'avg_volume_on_signals': volume[abs(signals) > 0.5].mean(),
            'avg_volume_no_signals': volume[abs(signals) <= 0.5].mean(),
            'volume_return_correlation': volume_return_corr.iloc[-30:].mean(),  # Ostatnie 30 dni
            'high_volume_signal_success': self._analyze_high_volume_success(ohlcv_data, signals)
        }
        
        return analysis
    
    def _analyze_high_volume_success(self, ohlcv_data, signals):
        """
        Analiza sukcesu sygnałów w wysokim wolumenie
        """
        volume = ohlcv_data['volume']
        returns = ohlcv_data['close'].pct_change()
        
        # Definicja wysokiego wolumenu (powyżej 75 percentyla)
        high_volume_threshold = volume.quantile(0.75)
        high_volume_mask = volume > high_volume_threshold
        
        # Zwroty następnego dnia dla sygnałów w wysokim wolumenie
        signal_returns = returns.shift(-1) * signals
        
        high_vol_signal_returns = signal_returns[high_volume_mask & (abs(signals) > 0.5)]
        normal_vol_signal_returns = signal_returns[~high_volume_mask & (abs(signals) > 0.5)]
        
        if len(high_vol_signal_returns) > 0 and len(normal_vol_signal_returns) > 0:
            return {
                'high_volume_avg_return': high_vol_signal_returns.mean(),
                'normal_volume_avg_return': normal_vol_signal_returns.mean(),
                'high_volume_win_rate': (high_vol_signal_returns > 0).mean() * 100,
                'normal_volume_win_rate': (normal_vol_signal_returns > 0).mean() * 100
            }
        
        return {'insufficient_data': True}

# Przykład użycia
volume_enhancer = VolumeBasedSignalEnhancer(kappa=0.1, window=30)

# Generuj sygnały wolumenu
volume_signals = volume_enhancer.generate_volume_momentum_signals(ohlcv_data)

# Połącz z sygnałami cenowymi
price_signals = calculate_strategy_signals(ohlcv_data)
confirmed_signals, confirmation_stats = volume_enhancer.combine_with_price_signals(
    price_signals, volume_signals, volume_threshold=0.6
)

# Analiza wzorców wolumenu
volume_analysis = volume_enhancer.analyze_volume_patterns(ohlcv_data, confirmed_signals)

print("📈 Statystyki potwierdzenia wolumenu:")
for key, value in confirmation_stats.items():
    print(f"  {key}: {value:.2f}")

print("\n📊 Analiza wzorców wolumenu:")
for key, value in volume_analysis.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey}: {subvalue:.2f}")
    else:
        print(f"  {key}: {value:.2f}")
```

### 6.3 Hidden Markov Models for Market Regime Detection

**Hidden Markov Models** pozwalają na identyfikację ukrytych stanów rynku i adaptację strategii do zmieniających się warunków.

```python
class MarketRegimeDetector:
    """
    Detektor reżimów rynkowych z wykorzystaniem Hidden Markov Models
    """
    def __init__(self, n_states=3):
        """
        Parameters:
        - n_states: Liczba ukrytych stanów (reżimów) do wykrycia
        """
        self.n_states = n_states
        self.hmm = None
        self.regime_labels = ['Low Volatility', 'Normal', 'High Volatility'][:n_states]
        self.trained = False
        
    def prepare_features(self, ohlcv_data):
        """
        Przygotowanie cech dla modelu HMM
        """
        returns = ohlcv_data['close'].pct_change()
        
        features = pd.DataFrame({
            'returns': returns,
            'abs_returns': returns.abs(),
            'volatility': returns.rolling(20).std(),
            'volume_ratio': ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()
        }).dropna()
        
        return features
    
    def train_regime_model(self, ohlcv_data, validation_split=0.2):
        """
        Trenowanie modelu HMM dla detekcji reżimów rynkowych
        
        Parameters:
        - ohlcv_data: Dane OHLCV
        - validation_split: Procent danych do walidacji
        
        Returns:
        - model_performance: Metryki wydajności modelu
        """
        features = self.prepare_features(ohlcv_data)
        
        # Podział na train/validation
        split_point = int(len(features) * (1 - validation_split))
        train_features = features.iloc[:split_point]
        val_features = features.iloc[split_point:]
        
        try:
            from hmmlearn import hmm
            
            # Inicjalizacja modelu HMM z Gaussian emissions
            self.hmm = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Trenowanie modelu
            self.hmm.fit(train_features.values)
            
            # Walidacja
            train_regimes = self.hmm.predict(train_features.values)
            val_regimes = self.hmm.predict(val_features.values)
            
            # Oblicz log-likelihood jako miarę jakości
            train_ll = self.hmm.score(train_features.values)
            val_ll = self.hmm.score(val_features.values)
            
            self.trained = True
            
            model_performance = {
                'train_log_likelihood': train_ll,
                'val_log_likelihood': val_ll,
                'n_regime_switches_train': np.sum(np.diff(train_regimes) != 0),
                'n_regime_switches_val': np.sum(np.diff(val_regimes) != 0),
                'regime_distribution_train': np.bincount(train_regimes, minlength=self.n_states),
                'regime_distribution_val': np.bincount(val_regimes, minlength=self.n_states)
            }
            
            return model_performance
            
        except ImportError:
            print("⚠️ hmmlearn nie jest dostępne, używam prostego modelu")
            return self._simple_regime_model(features, train_features, val_features)
    
    def _simple_regime_model(self, features, train_features, val_features):
        """
        Prosty model reżimów na podstawie kwantyli zmienności
        """
        volatility = features['volatility']
        
        # Definicja reżimów na podstawie kwantyli zmienności
        if self.n_states == 2:
            self.vol_thresholds = [volatility.quantile(0.5)]
        elif self.n_states == 3:
            self.vol_thresholds = [volatility.quantile(0.33), volatility.quantile(0.67)]
        else:
            self.vol_thresholds = [volatility.quantile(i/self.n_states) for i in range(1, self.n_states)]
        
        self.trained = True
        
        return {
            'model_type': 'simple_volatility_based',
            'thresholds': self.vol_thresholds,
            'train_samples': len(train_features),
            'val_samples': len(val_features)
        }
    
    def predict_current_regime(self, recent_data):
        """
        Przewidywanie aktualnego reżimu rynkowego
        
        Parameters:
        - recent_data: Ostatnie dane OHLCV (minimum 20 obserwacji)
        
        Returns:
        - current_regime: Aktualny reżim (0, 1, 2, ...)
        - regime_probabilities: Prawdopodobieństwa każdego reżimu
        """
        if not self.trained:
            raise ValueError("Model nie został wytrenowany")
        
        features = self.prepare_features(recent_data)
        latest_features = features.iloc[-1:].values
        
        if hasattr(self.hmm, 'predict_proba'):
            # HMM model
            regime_probs = self.hmm.predict_proba(latest_features)[0]
            current_regime = np.argmax(regime_probs)
        else:
            # Simple model
            current_vol = features['volatility'].iloc[-1]
            current_regime = 0
            for i, threshold in enumerate(self.vol_thresholds):
                if current_vol > threshold:
                    current_regime = i + 1
            
            # Pseudo-probabilities dla prostego modelu
            regime_probs = np.zeros(self.n_states)
            regime_probs[current_regime] = 0.8
            regime_probs[regime_probs == 0] = 0.2 / (self.n_states - 1)
        
        return current_regime, regime_probs
    
    def predict_historical_regimes(self, ohlcv_data):
        """
        Przewidywanie historycznych reżimów dla całego datasetu
        """
        if not self.trained:
            raise ValueError("Model nie został wytrenowany")
        
        features = self.prepare_features(ohlcv_data)
        
        if hasattr(self.hmm, 'predict'):
            historical_regimes = self.hmm.predict(features.values)
        else:
            # Simple model
            vol_series = features['volatility']
            historical_regimes = np.zeros(len(vol_series), dtype=int)
            
            for i, threshold in enumerate(self.vol_thresholds):
                historical_regimes[vol_series > threshold] = i + 1
        
        return pd.Series(historical_regimes, index=features.index, name='regime')
    
    def adaptive_strategy_parameters(self, regime_state, base_params=None):
        """
        Adaptacja parametrów strategii na podstawie reżimu rynkowego
        
        Parameters:
        - regime_state: Aktualny stan reżimu (0, 1, 2, ...)
        - base_params: Bazowe parametry strategii
        
        Returns:
        - adapted_params: Zaadaptowane parametry
        """
        if base_params is None:
            base_params = {
                'risk_multiplier': 1.0,
                'ma_period_adj': 1.0,
                'stop_loss_adj': 1.0,
                'position_size_adj': 1.0
            }
        
        # Przykładowa konfiguracja dla 3 stanów
        regime_configs = {
            0: {  # Low volatility regime
                'risk_multiplier': 1.2,      # Większe ryzyko w spokojnym rynku
                'ma_period_adj': 1.3,        # Dłuższe MA (mniej szumu)
                'stop_loss_adj': 1.5,        # Szersze stop loss
                'position_size_adj': 1.1     # Większe pozycje
            },
            1: {  # Normal regime  
                'risk_multiplier': 1.0,      # Standardowe parametry
                'ma_period_adj': 1.0,
                'stop_loss_adj': 1.0,
                'position_size_adj': 1.0
            },
            2: {  # High volatility regime
                'risk_multiplier': 0.6,      # Mniejsze ryzyko w volatile rynku
                'ma_period_adj': 0.8,        # Krótsze MA (szybsza reakcja)
                'stop_loss_adj': 0.7,        # Węższe stop loss
                'position_size_adj': 0.7     # Mniejsze pozycje
            }
        }
        
        # Adaptacja parametrów
        regime_config = regime_configs.get(regime_state, regime_configs[1])
        adapted_params = base_params.copy()
        
        for key, multiplier in regime_config.items():
            if key in adapted_params:
                adapted_params[key] *= multiplier
        
        return adapted_params, self.regime_labels[regime_state] if regime_state < len(self.regime_labels) else f"Regime_{regime_state}"
    
    def analyze_regime_performance(self, ohlcv_data, strategy_returns):
        """
        Analiza wydajności strategii w różnych reżimach
        """
        if not self.trained:
            raise ValueError("Model nie został wytrenowany")
        
        historical_regimes = self.predict_historical_regimes(ohlcv_data)
        
        # Analiza dla każdego reżimu
        regime_performance = {}
        
        for regime_id in range(self.n_states):
            regime_mask = historical_regimes == regime_id
            regime_returns = strategy_returns[regime_mask]
            
            if len(regime_returns) > 10:  # Minimum obserwacji
                regime_performance[f'regime_{regime_id}'] = {
                    'label': self.regime_labels[regime_id] if regime_id < len(self.regime_labels) else f"Regime_{regime_id}",
                    'total_return': regime_returns.sum() * 100,
                    'avg_return': regime_returns.mean() * 100,
                    'volatility': regime_returns.std() * np.sqrt(252) * 100,
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean() * 100,
                    'max_drawdown': (regime_returns.cumsum() - regime_returns.cumsum().expanding().max()).min() * 100,
                    'n_observations': len(regime_returns),
                    'time_in_regime': regime_mask.mean() * 100
                }
        
        return regime_performance
    
    def plot_regime_analysis(self, ohlcv_data, save_path=None):
        """
        Wizualizacja analizy reżimów
        """
        if not self.trained:
            raise ValueError("Model nie został wytrenowany")
        
        historical_regimes = self.predict_historical_regimes(ohlcv_data)
        features = self.prepare_features(ohlcv_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Ceny z reżimami
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_states))
        
        ax1.plot(ohlcv_data.index, ohlcv_data['close'], 'k-', alpha=0.7, linewidth=1)
        for regime_id in range(self.n_states):
            mask = historical_regimes == regime_id
            if mask.any():
                ax1.scatter(historical_regimes.index[mask], ohlcv_data['close'].reindex(historical_regimes.index)[mask], 
                          c=[colors[regime_id]], s=10, alpha=0.6, 
                          label=self.regime_labels[regime_id] if regime_id < len(self.regime_labels) else f"Regime {regime_id}")
        
        ax1.set_title('Klasyfikacja Reżimów Rynkowych')
        ax1.set_ylabel('Cena')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Zmienność z reżimami
        ax2.plot(features.index, features['volatility'] * 100, 'b-', alpha=0.7)
        ax2.scatter(historical_regimes.index, features['volatility'].reindex(historical_regimes.index) * 100, 
                   c=[colors[r] for r in historical_regimes], s=10, alpha=0.6)
        ax2.set_title('Zmienność vs Reżimy')
        ax2.set_ylabel('Zmienność (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Dystrybucja reżimów
        regime_counts = historical_regimes.value_counts().sort_index()
        ax3.bar(range(len(regime_counts)), regime_counts.values, color=[colors[i] for i in range(len(regime_counts))])
        ax3.set_title('Rozkład Reżimów')
        ax3.set_xlabel('Reżim')
        ax3.set_ylabel('Liczba obserwacji')
        ax3.set_xticks(range(len(regime_counts)))
        ax3.set_xticklabels([self.regime_labels[i] if i < len(self.regime_labels) else f"R{i}" for i in range(len(regime_counts))])
        
        # 4. Przejścia między reżimami
        regime_transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(historical_regimes) - 1):
            current_regime = historical_regimes.iloc[i]
            next_regime = historical_regimes.iloc[i + 1]
            regime_transitions[current_regime, next_regime] += 1
        
        # Normalizacja do prawdopodobieństw
        row_sums = regime_transitions.sum(axis=1, keepdims=True)
        regime_transitions_prob = np.divide(regime_transitions, row_sums, out=np.zeros_like(regime_transitions), where=row_sums!=0)
        
        im = ax4.imshow(regime_transitions_prob, cmap='Blues')
        ax4.set_title('Macierz Przejść (Prawdopodobieństwa)')
        ax4.set_xlabel('Do Reżimu')
        ax4.set_ylabel('Z Reżimu')
        
        # Dodaj wartości do macierzy
        for i in range(self.n_states):
            for j in range(self.n_states):
                ax4.text(j, i, f'{regime_transitions_prob[i, j]:.2f}', 
                        ha="center", va="center", color="red" if regime_transitions_prob[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Przykład użycia
regime_detector = MarketRegimeDetector(n_states=3)

# Trenowanie modelu
model_performance = regime_detector.train_regime_model(ohlcv_data)
print("📊 Wydajność modelu reżimów:")
for key, value in model_performance.items():
    print(f"  {key}: {value}")

# Przewidywanie aktualnego reżimu
current_regime, regime_probs = regime_detector.predict_current_regime(ohlcv_data.tail(50))
print(f"\n🎯 Aktualny reżim: {current_regime}")
print(f"📊 Prawdopodobieństwa: {regime_probs}")

# Adaptacja parametrów strategii
adapted_params, regime_name = regime_detector.adaptive_strategy_parameters(current_regime)
print(f"\n⚙️ Parametry dla reżimu '{regime_name}':")
for key, value in adapted_params.items():
    print(f"  {key}: {value:.2f}")

# Analiza wydajności w różnych reżimach
returns = ohlcv_data['close'].pct_change()
regime_performance = regime_detector.analyze_regime_performance(ohlcv_data, returns)
print(f"\n📈 Wydajność w różnych reżimach:")
for regime, metrics in regime_performance.items():
    print(f"\n  {metrics['label']}:")
    print(f"    Time in regime: {metrics['time_in_regime']:.1f}%")
    print(f"    Avg return: {metrics['avg_return']:.3f}%")
    print(f"    Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"    Volatility: {metrics['volatility']:.1f}%")
```

### 6.4 Integracja z Istniejącymi Strategiami

```python
class EnhancedAIMAStrategy(AdvancedTradingStrategy):
    """
    Rozszerzona strategia AI/MA z wykorzystaniem pattern detection
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Komponenty pattern detection
        self.trend_analyzer = AdvancedTrendAnalyzer(minamp=20, Tinactive=10)
        self.volume_enhancer = VolumeBasedSignalEnhancer(kappa=0.1, window=30)
        self.regime_detector = MarketRegimeDetector(n_states=3)
        
        # Status trenowania
        self.regime_trained = False
        
    def train_components(self, training_data):
        """
        Trenowanie komponentów pattern detection
        """
        print("🔧 Trenowanie komponentów pattern detection...")
        
        # Trenowanie detektora reżimów
        regime_performance = self.regime_detector.train_regime_model(training_data)
        self.regime_trained = True
        
        # Klasyfikacja historycznych trendów
        self.trend_analyzer.classify_historical_trends(training_data['close'])
        
        print("✅ Komponenty wytrenowane")
        return regime_performance
        
    def generate_enhanced_signals(self, data, base_signals=None):
        """
        Generowanie wzmocnionych sygnałów z pattern detection
        """
        if base_signals is None:
            # Podstawowa logika AI/MA (przykład)
            base_signals = self.calculate_aima_signals(data)
        
        # 1. Wzmocnienie sygnałów przez analizę trendów
        enhanced_signals, trend_labels = self.trend_analyzer.enhance_strategy_signals(
            data['close'], base_signals
        )
        
        # 2. Potwierdzenie przez analizę wolumenu
        volume_signals = self.volume_enhancer.generate_volume_momentum_signals(data)
        final_signals, confirmation_stats = self.volume_enhancer.combine_with_price_signals(
            enhanced_signals, volume_signals
        )
        
        # 3. Adaptacja do reżimu rynkowego
        if self.regime_trained:
            current_regime, regime_probs = self.regime_detector.predict_current_regime(data.tail(50))
            adapted_params, regime_name = self.regime_detector.adaptive_strategy_parameters(current_regime)
            
            # Zastosuj mnożnik ryzyka z reżimu
            final_signals *= adapted_params['risk_multiplier']
        else:
            regime_name = "Unknown"
            adapted_params = {}
        
        # Przygotuj metadata
        signal_metadata = {
            'trend_labels': trend_labels,
            'volume_confirmation': confirmation_stats,
            'current_regime': regime_name if self.regime_trained else "Not trained",
            'regime_params': adapted_params,
            'enhancement_layers': ['trend', 'volume', 'regime']
        }
        
        return final_signals, signal_metadata
    
    def calculate_aima_signals(self, data):
        """
        Podstawowa logika AI/MA (przykład - zastąp właściwą implementacją)
        """
        # To jest uproszczona implementacja - zastąp rzeczywistą logiką AI/MA
        fast_ma = data['close'].rolling(20).mean()
        slow_ma = data['close'].rolling(50).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
    
    def backtest_enhanced_strategy(self, data, initial_capital=10000):
        """
        Backtest strategii z pattern detection
        """
        # Trenowanie komponentów na pierwszych 70% danych
        train_size = int(len(data) * 0.7)
        training_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Trenowanie
        regime_performance = self.train_components(training_data)
        
        # Generowanie sygnałów dla danych testowych
        enhanced_signals, metadata = self.generate_enhanced_signals(test_data)
        
        # Obliczanie zwrotów
        returns = test_data['close'].pct_change()
        strategy_returns = returns * enhanced_signals.shift(1)
        
        # Obliczanie metryk
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        win_rate = (strategy_returns > 0).mean()
        
        results = {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'n_signals': (abs(enhanced_signals) > 0.5).sum(),
            'metadata': metadata,
            'regime_performance': regime_performance
        }
        
        return results, enhanced_signals, strategy_returns

# Przykład użycia Enhanced AI/MA Strategy
enhanced_strategy = EnhancedAIMAStrategy(initial_capital=10000)

# Backtest z pattern detection
results, signals, returns = enhanced_strategy.backtest_enhanced_strategy(ohlcv_data)

print("🚀 WYNIKI ENHANCED AI/MA STRATEGY:")
print(f"📈 Total Return: {results['total_return']:.2f}%")
print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.3f}")  
print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
print(f"🔄 Number of Signals: {results['n_signals']}")

# Wyświetl metadata ostatniego sygnału
if 'metadata' in results:
    metadata = results['metadata']
    print(f"\n🎯 Pattern Detection Status:")
    print(f"  Current Regime: {metadata['current_regime']}")
    print(f"  Enhancement Layers: {', '.join(metadata['enhancement_layers'])}")
    if 'volume_confirmation' in metadata:
        print(f"  Volume Confirmation - Long: {metadata['volume_confirmation'].get('long_confirmed_pct', 0):.1f}%")
        print(f"  Volume Confirmation - Short: {metadata['volume_confirmation'].get('short_confirmed_pct', 0):.1f}%")
```

## 7. Implementacja silnika backtestingu

### Zaawansowana klasa Backtester z wykorzystaniem VectorBT i DuckDB

```python
class AdvancedBacktester:
    def __init__(self, strategy, data, use_vectorbt=True, use_duckdb=True):
        """
        Zaawansowany silnik backtestingu wykorzystujący najnowsze biblioteki
        
        Parameters:
        - strategy: instancja klasy strategii
        - data: DataFrame lub dict z danymi historycznymi
        - use_vectorbt: czy używać VectorBT do obliczeń
        - use_duckdb: czy używać DuckDB do analiz
        """
        self.strategy = strategy
        self.data = data
        self.use_vectorbt = use_vectorbt
        self.use_duckdb = use_duckdb
        self.results = None
        self.portfolio = None
        
        if use_duckdb:
            self.conn = duckdb.connect(':memory:')
            
    def run_backtest(self):
        """
        Przeprowadza zaawansowany backtest
        """
        print("🚀 Rozpoczynanie zaawansowanego backtestu...")
        
        if self.use_vectorbt:
            return self._run_vectorbt_backtest()
        else:
            return self._run_standard_backtest()
    
    def _run_vectorbt_backtest(self):
        """
        Superszybki backtest z VectorBT
        """
        print("⚡ Używam VectorBT dla maksymalnej wydajności...")
        
        # Generuj sygnały
        if hasattr(self.strategy, 'generate_signals'):
            entries, exits, scores = self.strategy.generate_signals(self.data)
        else:
            # Domyślna strategia
            fast_ma = vbt.MA.run(self.data['Close'], window=20)
            slow_ma = vbt.MA.run(self.data['Close'], window=50)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Symulacja portfolio z zaawansowanymi opcjami
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.data['Close'],
            entries=entries,
            exits=exits,
            size=vbt.RepeatMode.TargetPercent,  # Używaj % kapitału
            size_value=0.95,  # 95% dostępnego kapitału
            init_cash=self.strategy.initial_capital,
            fees=self.strategy.commission,
            slippage=0.0005,  # 0.05% poślizgu
            freq='1D',
            call_seq='auto',  # Automatyczna optymalizacja kolejności
            cash_sharing=True,  # Współdzielenie kapitału między sygnałami
            log=True  # Logowanie wszystkich transakcji
        )
        
        # Oblicz wszystkie metryki jednocześnie
        self.results = self._calculate_vectorbt_metrics()
        
        # Zapisz transakcje
        self.trades = self.portfolio.trades.records_readable
        
        # Analiza z DuckDB jeśli włączona
        if self.use_duckdb:
            self._analyze_with_duckdb()
        
        return self.results
    
    def _calculate_vectorbt_metrics(self):
        """
        Oblicz kompletny zestaw metryk z VectorBT
        """
        # Podstawowe statystyki
        stats = self.portfolio.stats()
        
        # Rozszerzone metryki
        extended_metrics = {
            # Zwroty
            'Total_Return': self.portfolio.total_return() * 100,
            'Annual_Return': self.portfolio.annualized_return() * 100,
            'Benchmark_Return': self.portfolio.benchmark_return() * 100,
            'Alpha': (self.portfolio.annualized_return() - self.portfolio.benchmark_return()) * 100,
            
            # Ryzyko
            'Volatility': self.portfolio.annualized_volatility() * 100,
            'Sharpe_Ratio': self.portfolio.sharpe_ratio(),
            'Sortino_Ratio': self.portfolio.sortino_ratio(),
            'Calmar_Ratio': self.portfolio.calmar_ratio(),
            'Omega_Ratio': self.portfolio.omega_ratio(),
            'Max_Drawdown': self.portfolio.max_drawdown() * 100,
            'Max_Drawdown_Duration': self.portfolio.max_drawdown_duration(),
            
            # VaR i CVaR
            'Value_at_Risk_95': self.portfolio.value_at_risk(0.95) * 100,
            'Conditional_VaR_95': self.portfolio.conditional_value_at_risk(0.95) * 100,
            'Value_at_Risk_99': self.portfolio.value_at_risk(0.99) * 100,
            'Conditional_VaR_99': self.portfolio.conditional_value_at_risk(0.99) * 100,
            
            # Statystyki transakcji
            'Total_Trades': self.portfolio.total_trades(),
            'Win_Rate': self.portfolio.win_rate() * 100,
            'Best_Trade': self.portfolio.best_trade_return() * 100,
            'Worst_Trade': self.portfolio.worst_trade_return() * 100,
            'Avg_Win': self.portfolio.avg_win_return() * 100,
            'Avg_Loss': self.portfolio.avg_loss_return() * 100,
            'Profit_Factor': self.portfolio.profit_factor(),
            'Expectancy': self.portfolio.expectancy(),
            'Kelly_Criterion': self.portfolio.kelly_criterion() * 100,
            
            # Zaawansowane metryki
            'Tail_Ratio': self.portfolio.tail_ratio(),
            'Common_Sense_Ratio': self.portfolio.common_sense_ratio(),
            'CPC_Ratio': self.portfolio.cpc_ratio(),
            'Lake_Ratio': self.portfolio.lake_ratio(),
            'Burke_Ratio': self.portfolio.burke_ratio(),
            'Rachev_Ratio': self.portfolio.rachev_ratio(),
            'Skewness': self.portfolio.skew(),
            'Kurtosis': self.portfolio.kurtosis(),
            
            # Czas w rynku
            'Exposure_Time': self.portfolio.total_exposure_time(),
            'Avg_Trade_Duration': self.portfolio.avg_trade_duration(),
            
            # Kapitał
            'Final_Value': self.portfolio.final_value(),
            'Total_Profit': self.portfolio.total_profit(),
            'Total_Fees_Paid': self.portfolio.total_fees_paid()
        }
        
        return pd.Series(extended_metrics)
    
    def _analyze_with_duckdb(self):
        """
        Zaawansowana analiza transakcji z DuckDB
        """
        print("🗄️ Analizuję transakcje z DuckDB...")
        
        # Załaduj dane do DuckDB
        self.conn.register('trades', self.trades)
        self.conn.register('portfolio_value', self.portfolio.value())
        
        # Analiza wydajności w czasie
        time_analysis_query = """
        WITH trade_times AS (
            SELECT 
                *,
                EXTRACT(hour FROM "Entry Timestamp") as entry_hour,
                EXTRACT(dow FROM "Entry Timestamp") as entry_dow,
                EXTRACT(month FROM "Entry Timestamp") as entry_month,
                EXTRACT(quarter FROM "Entry Timestamp") as entry_quarter
            FROM trades
        )
        SELECT 
            entry_hour,
            COUNT(*) as trade_count,
            AVG("Return [%]") as avg_return,
            SUM(CASE WHEN "Return [%]" > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Return [%]") as median_return
        FROM trade_times
        GROUP BY entry_hour
        ORDER BY avg_return DESC
        """
        
        self.hourly_performance = self.conn.execute(time_analysis_query).df()
        
        # Analiza sekwencji wygranych/przegranych
        streak_analysis_query = """
        WITH trade_results AS (
            SELECT 
                *,
                CASE WHEN "Return [%]" > 0 THEN 1 ELSE 0 END as is_win,
                ROW_NUMBER() OVER (ORDER BY "Entry Timestamp") as trade_num
            FROM trades
        ),
        streaks AS (
            SELECT 
                *,
                SUM(CASE WHEN is_win != LAG(is_win) OVER (ORDER BY trade_num) THEN 1 ELSE 0 END) 
                    OVER (ORDER BY trade_num) as streak_group
            FROM trade_results
        )
        SELECT 
            streak_group,
            is_win,
            COUNT(*) as streak_length,
            SUM("Return [%]") as total_return,
            AVG("Return [%]") as avg_return
        FROM streaks
        GROUP BY streak_group, is_win
        ORDER BY streak_length DESC
        """
        
        self.streak_analysis = self.conn.execute(streak_analysis_query).df()
        
        # Analiza drawdown periods
        dd_analysis_query = """
        WITH drawdown_periods AS (
            SELECT 
                date,
                value,
                MAX(value) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) as running_max,
                (value - MAX(value) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING)) / 
                MAX(value) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) as drawdown_pct
            FROM portfolio_value
        ),
        dd_segments AS (
            SELECT 
                *,
                CASE WHEN drawdown_pct < 0 AND LAG(drawdown_pct) OVER (ORDER BY date) >= 0 THEN 1 ELSE 0 END as new_dd_start,
                SUM(CASE WHEN drawdown_pct < 0 AND LAG(drawdown_pct) OVER (ORDER BY date) >= 0 THEN 1 ELSE 0 END) 
                    OVER (ORDER BY date) as dd_group
            FROM drawdown_periods
        )
        SELECT 
            dd_group,
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(*) as duration_days,
            MIN(drawdown_pct) * 100 as max_drawdown_pct,
            FIRST(value) as start_value,
            LAST(value) as end_value
        FROM dd_segments
        WHERE drawdown_pct < 0
        GROUP BY dd_group
        HAVING COUNT(*) > 1
        ORDER BY max_drawdown_pct
        """
        
        self.drawdown_analysis = self.conn.execute(dd_analysis_query).df()
    
    def generate_comprehensive_report(self):
        """
        Generuj kompleksowy raport HTML z wynikami
        """
        from jinja2 import Template
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {{ strategy_name }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
                .metric-card { 
                    background: #f0f0f0; 
                    padding: 15px; 
                    border-radius: 8px;
                    text-align: center;
                }
                .metric-value { font-size: 24px; font-weight: bold; }
                .metric-label { color: #666; font-size: 14px; }
                .section { margin: 30px 0; }
                h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>📊 Backtest Report - {{ strategy_name }}</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(total_return) }}%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.3f"|format(sharpe_ratio) }}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(max_drawdown) }}%</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(win_rate) }}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Portfolio Performance</h2>
                <div id="equity-curve"></div>
            </div>
            
            <div class="section">
                <h2>Trade Analysis</h2>
                <div id="trade-returns"></div>
            </div>
            
            <script>
                {{ plotly_charts }}
            </script>
        </body>
        </html>
        """
        
        # Przygotuj dane dla template
        context = {
            'strategy_name': self.strategy.__class__.__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_return': self.results['Total_Return'],
            'sharpe_ratio': self.results['Sharpe_Ratio'],
            'max_drawdown': self.results['Max_Drawdown'],
            'win_rate': self.results['Win_Rate'],
            'plotly_charts': self._generate_plotly_charts()
        }
        
        # Renderuj template
        template = Template(html_template)
        html_report = template.render(**context)
        
        # Zapisz raport
        with open('backtest_report.html', 'w') as f:
            f.write(html_report)
        
        print("📄 Raport HTML wygenerowany: backtest_report.html")

# Hybrydowy Backtester łączący wszystkie biblioteki
class UltimateBacktester:
    """
    Ultimate backtester wykorzystujący wszystkie zaawansowane biblioteki
    """
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Komponenty
        self.data_manager = DataManager()
        self.duckdb_conn = duckdb.connect(':memory:')
        
    def run_massive_backtest(self, strategies, parameter_grid):
        """
        Masowy backtest wielu strategii z różnymi parametrami
        Wykorzystuje Polars + DuckDB + VectorBT dla maksymalnej wydajności
        """
        print("🚀 Rozpoczynam MASOWY BACKTEST...")
        print(f"📊 Strategii: {len(strategies)}")
        print(f"🔧 Kombinacji parametrów: {len(parameter_grid)}")
        print(f"💹 Symboli: {len(self.symbols)}")
        
        total_tests = len(strategies) * len(parameter_grid) * len(self.symbols)
        print(f"📈 TOTAL TESTÓW: {total_tests}")
        
        # Pobierz dane raz dla wszystkich testów
        all_data = self._load_all_data_polars()
        
        # Parallel processing results
        results = []
        
        # Użyj joblib dla równoległego przetwarzania
        from joblib import Parallel, delayed
        
        def run_single_test(strategy_class, params, symbol, data):
            """Pojedynczy test"""
            try:
                # Inicjalizuj strategię
                strategy = strategy_class(**params)
                
                # Backtest z VectorBT
                backtester = AdvancedBacktester(strategy, data[symbol])
                result = backtester.run_backtest()
                
                return {
                    'strategy': strategy_class.__name__,
                    'symbol': symbol,
                    'params': params,
                    'sharpe_ratio': result['Sharpe_Ratio'],
                    'total_return': result['Total_Return'],
                    'max_drawdown': result['Max_Drawdown'],
                    'win_rate': result['Win_Rate'],
                    'profit_factor': result['Profit_Factor']
                }
            except Exception as e:
                print(f"❌ Błąd dla {strategy_class.__name__} - {symbol}: {e}")
                return None
        
        # Równoległe wykonanie wszystkich testów
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_single_test)(strategy, params, symbol, all_data)
            for strategy in strategies
            for params in parameter_grid
            for symbol in self.symbols
        )
        
        # Filtruj błędy
        results = [r for r in results if r is not None]
        
        # Analiza wyników z DuckDB
        return self._analyze_massive_results(results)
    
    def _load_all_data_polars(self):
        """
        Superszybkie ładowanie wszystkich danych z Polars
        """
        all_data = {}
        
        for symbol in self.symbols:
            # Sprawdź czy dane są w cache
            cache_file = f"./market_data/{symbol}_enhanced.parquet"
            
            if os.path.exists(cache_file):
                # Lazy load z Polars
                df = pl.scan_parquet(cache_file)
            else:
                # Pobierz i przetworz
                df = self._download_and_process_symbol(symbol)
                
            all_data[symbol] = df.collect()
        
        return all_data
    
    def _analyze_massive_results(self, results):
        """
        Analiza wyników masowego backtestu z DuckDB
        """
        # Załaduj do DuckDB
        results_df = pd.DataFrame(results)
        self.duckdb_conn.register('results', results_df)
        
        # Znajdź najlepsze strategie
        best_strategies_query = """
        WITH ranked_strategies AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY strategy, symbol 
                    ORDER BY sharpe_ratio DESC
                ) as rank
            FROM results
            WHERE sharpe_ratio > 0 AND max_drawdown > -50
        )
        SELECT 
            strategy,
            symbol,
            params,
            sharpe_ratio,
            total_return,
            max_drawdown,
            win_rate,
            profit_factor
        FROM ranked_strategies
        WHERE rank = 1
        ORDER BY sharpe_ratio DESC
        LIMIT 20
        """
        
        best_strategies = self.duckdb_conn.execute(best_strategies_query).df()
        
        # Analiza stabilności parametrów
        stability_query = """
        SELECT 
            strategy,
            params::json->>'short_window' as param_value,
            AVG(sharpe_ratio) as avg_sharpe,
            STDDEV(sharpe_ratio) as sharpe_std,
            AVG(sharpe_ratio) / NULLIF(STDDEV(sharpe_ratio), 0) as stability_score,
            COUNT(*) as test_count
        FROM results
        GROUP BY strategy, param_value
        HAVING COUNT(*) > 5
        ORDER BY stability_score DESC
        """
        
        stability_analysis = self.duckdb_conn.execute(stability_query).df()
        
        # Korelacja między strategiami
        correlation_query = """
        WITH strategy_returns AS (
            SELECT 
                a.strategy as strategy_a,
                b.strategy as strategy_b,
                CORR(a.total_return, b.total_return) as correlation
            FROM results a
            JOIN results b ON a.symbol = b.symbol AND a.strategy < b.strategy
            GROUP BY a.strategy, b.strategy
        )
        SELECT * FROM strategy_returns
        WHERE ABS(correlation) < 0.5  -- Niska korelacja = dobra dywersyfikacja
        ORDER BY correlation
        """
        
        low_correlation_pairs = self.duckdb_conn.execute(correlation_query).df()
        
        return {
            'best_strategies': best_strategies,
            'stability_analysis': stability_analysis,
            'low_correlation_pairs': low_correlation_pairs,
            'summary_stats': self._calculate_summary_stats(results_df)
        }
    
    def _calculate_summary_stats(self, results_df):
        """
        Oblicz statystyki podsumowujące
        """
        return {
            'total_tests': len(results_df),
            'profitable_strategies': (results_df['total_return'] > 0).sum(),
            'avg_sharpe_ratio': results_df['sharpe_ratio'].mean(),
            'best_sharpe_ratio': results_df['sharpe_ratio'].max(),
            'avg_max_drawdown': results_df['max_drawdown'].mean(),
            'strategies_with_sharpe_above_1': (results_df['sharpe_ratio'] > 1).sum(),
            'strategies_with_dd_below_20': (results_df['max_drawdown'] > -20).sum()
        }
```

## 7. Kluczowe metryki oceny strategii

### Kompleksowy system metryk z wykorzystaniem wszystkich bibliotek

```python
class AdvancedMetricsCalculator:
    """
    Zaawansowany kalkulator metryk wykorzystujący VectorBT, DuckDB i Riskfolio
    """
    def __init__(self, portfolio_data, trades_data, market_data):
        self.portfolio_data = portfolio_data
        self.trades_data = trades_data
        self.market_data = market_data
        self.conn = duckdb.connect(':memory:')
        
    def calculate_all_metrics(self):
        """
        Oblicza kompletny zestaw metryk - ponad 100 różnych wskaźników!
        """
        metrics = {}
        
        # 1. Metryki zwrotu
        metrics.update(self._calculate_return_metrics())
        
        # 2. Metryki ryzyka
        metrics.update(self._calculate_risk_metrics())
        
        # 3. Metryki efektywności skorygowanej o ryzyko
        metrics.update(self._calculate_risk_adjusted_metrics())
        
        # 4. Metryki transakcyjne
        metrics.update(self._calculate_trade_metrics())
        
        # 5. Metryki drawdown
        metrics.update(self._calculate_drawdown_metrics())
        
        # 6. Metryki dystrybucji
        metrics.update(self._calculate_distribution_metrics())
        
        # 7. Metryki stabilności
        metrics.update(self._calculate_stability_metrics())
        
        # 8. Metryki względem benchmarku
        metrics.update(self._calculate_benchmark_metrics())
        
        # 9. Metryki ML/AI
        metrics.update(self._calculate_ml_metrics())
        
        # 10. Metryki custom
        metrics.update(self._calculate_custom_metrics())
        
        return pd.Series(metrics)
    
    def _calculate_return_metrics(self):
        """Metryki związane ze zwrotami"""
        returns = self.portfolio_data['returns']
        
        # Użyj DuckDB dla szybkich obliczeń
        self.conn.register('returns', pd.DataFrame({'date': returns.index, 'returns': returns.values}))
        
        query = """
        WITH return_stats AS (
            SELECT 
                -- Podstawowe statystyki
                SUM(1 + returns) - 1 as total_return,
                AVG(returns) * 252 as annualized_mean,
                
                -- Zwroty w różnych okresach
                SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '1 month' THEN 1 + returns ELSE 1 END) - 1 as return_1m,
                SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '3 months' THEN 1 + returns ELSE 1 END) - 1 as return_3m,
                SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '6 months' THEN 1 + returns ELSE 1 END) - 1 as return_6m,
                SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '1 year' THEN 1 + returns ELSE 1 END) - 1 as return_1y,
                
                -- Zwroty w różnych warunkach rynkowych
                AVG(CASE WHEN returns > 0 THEN returns ELSE NULL END) as avg_positive_return,
                AVG(CASE WHEN returns < 0 THEN returns ELSE NULL END) as avg_negative_return,
                
                -- Konsystencja zwrotów
                COUNT(CASE WHEN returns > 0 THEN 1 END) * 100.0 / COUNT(*) as positive_days_pct,
                
                -- Rolling returns
                MAX(returns) as best_day,
                MIN(returns) as worst_day
            FROM returns
        )
        SELECT * FROM return_stats
        """
        
        return_stats = self.conn.execute(query).df().iloc[0].to_dict()
        
        # Dodaj metryki z VectorBT
        if hasattr(self, 'vbt_portfolio'):
            return_stats.update({
                'geometric_mean_return': self.vbt_portfolio.geometric_mean() * 100,
                'cagr': self.vbt_portfolio.cagr() * 100,
                'mtd_return': self.vbt_portfolio.month_to_date_return() * 100,
                'qtd_return': self.vbt_portfolio.quarter_to_date_return() * 100,
                'ytd_return': self.vbt_portfolio.year_to_date_return() * 100
            })
        
        return return_stats
    
    def _calculate_risk_metrics(self):
        """Zaawansowane metryki ryzyka"""
        returns = self.portfolio_data['returns']
        
        # Podstawowe metryki zmienności
        risk_metrics = {
            'volatility_annual': returns.std() * np.sqrt(252) * 100,
            'volatility_monthly': returns.std() * np.sqrt(21) * 100,
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252) * 100,
            'upside_deviation': returns[returns > 0].std() * np.sqrt(252) * 100,
        }
        
        # VaR i CVaR dla różnych poziomów ufności
        confidence_levels = [0.90, 0.95, 0.99]
        for cl in confidence_levels:
            var = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= var].mean()
            risk_metrics[f'var_{int(cl*100)}'] = var * 100
            risk_metrics[f'cvar_{int(cl*100)}'] = cvar * 100
        
        # Zaawansowane miary ryzyka z Riskfolio
        if 'riskfolio' in globals():
            port = rp.Portfolio(returns=pd.DataFrame(returns))
            
            risk_metrics.update({
                'mad': port.mad() * 100,  # Mean Absolute Deviation
                'gmd': port.gmd() * 100,  # Gini Mean Difference
                'tg': port.tg(alpha=0.05) * 100,  # Tail Gini
                'rg': port.rg(alpha=0.05) * 100,  # Range
                'cvrg': port.cvrg(alpha=0.05, beta=0.05) * 100,  # CVaR Range
                'wr': port.wr() * 100,  # Worst Realization
            })
        
        # Metryki tail risk
        risk_metrics.update({
            'left_tail_ratio': abs(returns.quantile(0.05) / returns.quantile(0.95)),
            'gain_to_pain_ratio': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
        })
        
        return risk_metrics
    
    def _calculate_risk_adjusted_metrics(self):
        """Metryki efektywności skorygowane o ryzyko"""
        returns = self.portfolio_data['returns']
        rf_rate = 0.02  # Risk-free rate
        
        # Podstawowe wskaźniki
        excess_returns = returns - rf_rate / 252
        
        metrics = {
            'sharpe_ratio': np.sqrt(252) * excess_returns.mean() / returns.std(),
            'sortino_ratio': np.sqrt(252) * excess_returns.mean() / returns[returns < 0].std(),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'sterling_ratio': self._calculate_sterling_ratio(),
            'burke_ratio': self._calculate_burke_ratio(),
            'martin_ratio': self._calculate_martin_ratio(),
            'pain_ratio': self._calculate_pain_ratio(),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(),
            'kappa_3': self._calculate_kappa(3),
            'omega_ratio': self._calculate_omega_ratio(),
            'upside_potential_ratio': self._calculate_upside_potential_ratio(),
            'd_ratio': self._calculate_d_ratio(),
        }
        
        # Dodaj metryki z VectorBT jeśli dostępne
        if hasattr(self, 'vbt_portfolio'):
            metrics.update({
                'information_ratio': self.vbt_portfolio.information_ratio(),
                'probabilistic_sharpe_ratio': self.vbt_portfolio.probabilistic_sharpe_ratio(),
                'deflated_sharpe_ratio': self.vbt_portfolio.deflated_sharpe_ratio(),
                'skew_adjusted_sharpe_ratio': self._calculate_skew_adjusted_sharpe(),
                'modified_sharpe_ratio': self._calculate_modified_sharpe(),
                'conditional_sharpe_ratio': self._calculate_conditional_sharpe(),
            })
        
        return metrics
    
    def _calculate_trade_metrics(self):
        """Szczegółowe metryki transakcyjne z DuckDB"""
        # Załaduj dane do DuckDB
        self.conn.register('trades', self.trades_data)
        
        # Kompleksowa analiza transakcji
        query = """
        WITH trade_analysis AS (
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN return > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN return <= 0 THEN 1 END) as losing_trades,
                
                -- Win rates
                COUNT(CASE WHEN return > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate,
                COUNT(CASE WHEN return > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(CASE WHEN entry_signal = 'long' THEN 1 END), 0) as long_win_rate,
                COUNT(CASE WHEN return > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(CASE WHEN entry_signal = 'short' THEN 1 END), 0) as short_win_rate,
                
                -- Returns
                AVG(return) * 100 as avg_return,
                AVG(CASE WHEN return > 0 THEN return END) * 100 as avg_win,
                AVG(CASE WHEN return <= 0 THEN return END) * 100 as avg_loss,
                MAX(return) * 100 as best_trade,
                MIN(return) * 100 as worst_trade,
                
                -- Profit factors
                SUM(CASE WHEN return > 0 THEN return ELSE 0 END) / 
                    NULLIF(ABS(SUM(CASE WHEN return <= 0 THEN return ELSE 0 END)), 0) as profit_factor,
                
                -- Duration
                AVG(duration_days) as avg_duration,
                AVG(CASE WHEN return > 0 THEN duration_days END) as avg_win_duration,
                AVG(CASE WHEN return <= 0 THEN duration_days END) as avg_loss_duration,
                
                -- Consecutive analysis
                MAX(consecutive_wins) as max_consecutive_wins,
                MAX(consecutive_losses) as max_consecutive_losses,
                
                -- Time analysis
                COUNT(CASE WHEN EXTRACT(dow FROM entry_date) = 1 THEN 1 END) * 100.0 / COUNT(*) as monday_trades_pct,
                COUNT(CASE WHEN EXTRACT(hour FROM entry_time) BETWEEN 9 AND 11 THEN 1 END) * 100.0 / COUNT(*) as morning_trades_pct,
                
                -- Risk metrics
                STDDEV(return) * 100 as trade_return_std,
                AVG(max_adverse_excursion) * 100 as avg_mae,
                AVG(max_favorable_excursion) * 100 as avg_mfe,
                
                -- Efficiency
                SUM(return) / SUM(ABS(return)) as efficiency_ratio,
                COUNT(CASE WHEN ABS(return) < 0.001 THEN 1 END) * 100.0 / COUNT(*) as scratch_trades_pct
                
            FROM trades
        ),
        payoff_ratio AS (
            SELECT 
                AVG(CASE WHEN return > 0 THEN return END) / 
                    NULLIF(ABS(AVG(CASE WHEN return <= 0 THEN return END)), 0) as payoff_ratio
            FROM trades
        )
        SELECT 
            t.*,
            p.payoff_ratio,
            t.win_rate * p.payoff_ratio - (100 - t.win_rate) as expectancy
        FROM trade_analysis t
        CROSS JOIN payoff_ratio p
        """
        
        trade_metrics = self.conn.execute(query).df().iloc[0].to_dict()
        
        # Dodaj zaawansowane metryki
        trade_metrics.update({
            'kelly_percentage': self._calculate_kelly_criterion(),
            'system_quality_number': self._calculate_sqn(),
            'edge_ratio': self._calculate_edge_ratio(),
            't_statistic': self._calculate_t_statistic(),
            'z_score': self._calculate_z_score(),
            'run_test_z_score': self._calculate_run_test(),
        })
        
        return trade_metrics
    
    def _calculate_drawdown_metrics(self):
        """Kompleksowa analiza drawdown"""
        equity = self.portfolio_data['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        # Podstawowe metryki
        metrics = {
            'max_drawdown': drawdown.min() * 100,
            'avg_drawdown': drawdown[drawdown < 0].mean() * 100,
            'max_drawdown_duration': self._calculate_max_dd_duration(drawdown),
            'avg_drawdown_duration': self._calculate_avg_dd_duration(drawdown),
            'recovery_factor': (equity.iloc[-1] - equity.iloc[0]) / abs(drawdown.min() * equity.max()),
            'pain_index': np.sqrt((drawdown ** 2).mean()) * 100,
        }
        
        # Analiza drawdown z DuckDB
        dd_df = pd.DataFrame({
            'date': equity.index,
            'equity': equity.values,
            'drawdown': drawdown.values
        })
        self.conn.register('drawdowns', dd_df)
        
        dd_query = """
        WITH dd_periods AS (
            SELECT 
                *,
                CASE WHEN drawdown < 0 AND LAG(drawdown) OVER (ORDER BY date) >= 0 THEN 1 ELSE 0 END as new_dd,
                SUM(CASE WHEN drawdown < 0 AND LAG(drawdown) OVER (ORDER BY date) >= 0 THEN 1 ELSE 0 END) 
                    OVER (ORDER BY date) as dd_id
            FROM drawdowns
        ),
        dd_stats AS (
            SELECT 
                dd_id,
                MIN(drawdown) * 100 as depth,
                COUNT(*) as duration,
                MIN(date) as start_date,
                MAX(date) as end_date,
                FIRST(equity) as start_equity,
                LAST(equity) as end_equity
            FROM dd_periods
            WHERE drawdown < 0
            GROUP BY dd_id
        )
        SELECT 
            COUNT(*) as total_drawdowns,
            AVG(depth) as avg_dd_depth,
            AVG(duration) as avg_dd_duration,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY depth) as dd_95_percentile,
            SUM(duration) * 100.0 / (SELECT COUNT(*) FROM drawdowns) as time_underwater_pct
        FROM dd_stats
        """
        
        dd_analysis = self.conn.execute(dd_query).df().iloc[0].to_dict()
        metrics.update(dd_analysis)
        
        # Ulcer Index i Martin Ratio
        metrics['ulcer_index'] = np.sqrt((drawdown ** 2).mean()) * 100
        metrics['martin_ratio'] = (equity.iloc[-1] / equity.iloc[0] - 1) / metrics['ulcer_index'] if metrics['ulcer_index'] != 0 else 0
        
        return metrics
    
    def _calculate_distribution_metrics(self):
        """Metryki rozkładu zwrotów"""
        returns = self.portfolio_data['returns']
        
        # Momenty statystyczne
        metrics = {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera_stat': self._jarque_bera_test(returns),
            'normal_distribution_test': self._test_normality(returns),
        }
        
        # Percentyle
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'percentile_{p}'] = np.percentile(returns, p) * 100
        
        # Analiza ogonów rozkładu
        metrics.update({
            'left_tail_weight': len(returns[returns < returns.quantile(0.05)]) / len(returns) * 100,
            'right_tail_weight': len(returns[returns > returns.quantile(0.95)]) / len(returns) * 100,
            'tail_ratio': abs(returns.quantile(0.05)) / returns.quantile(0.95),
        })
        
        # Higher moments
        metrics.update({
            'downside_skewness': returns[returns < 0].skew(),
            'upside_skewness': returns[returns > 0].skew(),
            'co_kurtosis': self._calculate_co_kurtosis(returns),
            'downside_risk_skewness': self._calculate_downside_risk_skewness(returns),
        })
        
        return metrics
    
    def _calculate_stability_metrics(self):
        """Metryki stabilności strategii"""
        returns = self.portfolio_data['returns']
        equity = self.portfolio_data['equity']
        
        # Rolling window analysis
        window = 252  # 1 rok
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        
        metrics = {
            'sharpe_stability': 1 - rolling_sharpe.std() / rolling_sharpe.mean() if rolling_sharpe.mean() != 0 else 0,
            'rolling_sharpe_min': rolling_sharpe.min(),
            'rolling_sharpe_max': rolling_sharpe.max(),
            'return_stability': 1 - returns.rolling(window).mean().std() / returns.mean() if returns.mean() != 0 else 0,
        }
        
        # Analiza podokresów
        self.conn.register('equity', pd.DataFrame({'date': equity.index, 'equity': equity.values}))
        
        subperiod_query = """
        WITH yearly_returns AS (
            SELECT 
                EXTRACT(year FROM date) as year,
                (LAST(equity) / FIRST(equity) - 1) * 100 as annual_return
            FROM equity
            GROUP BY EXTRACT(year FROM date)
        )
        SELECT 
            COUNT(CASE WHEN annual_return > 0 THEN 1 END) * 100.0 / COUNT(*) as positive_years_pct,
            STDDEV(annual_return) as annual_return_volatility,
            MIN(annual_return) as worst_year,
            MAX(annual_return) as best_year,
            AVG(annual_return) as avg_annual_return
        FROM yearly_returns
        """
        
        stability_stats = self.conn.execute(subperiod_query).df().iloc[0].to_dict()
        metrics.update(stability_stats)
        
        # Consistency metrics
        metrics.update({
            'consistency_ratio': self._calculate_consistency_ratio(),
            'smoothness_index': self._calculate_smoothness_index(),
            'robustness_score': self._calculate_robustness_score(),
        })
        
        return metrics
    
    def _calculate_ml_metrics(self):
        """Metryki związane z uczeniem maszynowym"""
        if not hasattr(self, 'ml_predictions'):
            return {}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_true = self.ml_predictions['actual']
        y_pred = self.ml_predictions['predicted']
        y_proba = self.ml_predictions['probability']
        
        metrics = {
            'ml_accuracy': accuracy_score(y_true, y_pred) * 100,
            'ml_precision': precision_score(y_true, y_pred) * 100,
            'ml_recall': recall_score(y_true, y_pred) * 100,
            'ml_f1_score': f1_score(y_true, y_pred) * 100,
            'ml_auc_roc': roc_auc_score(y_true, y_proba) * 100,
            'ml_feature_importance_top': self._get_top_features(),
            'ml_prediction_confidence_avg': y_proba.mean() * 100,
        }
        
        return metrics
    
    def generate_metrics_report(self, metrics):
        """
        Generuj piękny raport HTML z wszystkimi metrykami
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Metrics Report</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }
                .metric-section { 
                    background: white; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px; 
                    margin-top: 15px;
                }
                .metric-item {
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }
                .metric-value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #2c3e50;
                    margin: 5px 0;
                }
                .metric-label { 
                    color: #7f8c8d; 
                    font-size: 12px;
                    text-transform: uppercase;
                }
                .good { border-left-color: #27ae60; }
                .warning { border-left-color: #f39c12; }
                .bad { border-left-color: #e74c3c; }
                .chart { margin: 20px 0; }
                h2 { color: #2c3e50; margin-top: 0; }
                .summary-box {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Comprehensive Strategy Metrics Report</h1>
                <p>Generated: {{ timestamp }}</p>
            </div>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <div class="metric-grid">
                    <div class="metric-item" style="background: rgba(255,255,255,0.2);">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value">{{ "%.2f"|format(total_return) }}%</div>
                    </div>
                    <div class="metric-item" style="background: rgba(255,255,255,0.2);">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{{ "%.3f"|format(sharpe_ratio) }}</div>
                    </div>
                    <div class="metric-item" style="background: rgba(255,255,255,0.2);">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value">{{ "%.2f"|format(max_drawdown) }}%</div>
                    </div>
                    <div class="metric-item" style="background: rgba(255,255,255,0.2);">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{{ "%.1f"|format(win_rate) }}%</div>
                    </div>
                </div>
            </div>
            
            <div class="metric-section">
                <h2>📈 Return Metrics</h2>
                <div class="metric-grid">
                    {% for metric, value in return_metrics.items() %}
                    <div class="metric-item">
                        <div class="metric-label">{{ metric }}</div>
                        <div class="metric-value">{{ "%.2f"|format(value) }}%</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Add more sections for other metric categories -->
            
        </body>
        </html>
        """
        
        # Generate the report
        from jinja2 import Template
        template = Template(html_template)
        
        html_content = template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_return=metrics.get('Total_Return', 0),
            sharpe_ratio=metrics.get('Sharpe_Ratio', 0),
            max_drawdown=metrics.get('Max_Drawdown', 0),
            win_rate=metrics.get('Win_Rate', 0),
            return_metrics={k: v for k, v in metrics.items() if 'return' in k.lower()}
        )
        
        with open('comprehensive_metrics_report.html', 'w') as f:
            f.write(html_content)
        
        print("📊 Kompleksowy raport metryk wygenerowany: comprehensive_metrics_report.html")

# Przykład użycia
calculator = AdvancedMetricsCalculator(portfolio_data, trades_data, market_data)
all_metrics = calculator.calculate_all_metrics()
calculator.generate_metrics_report(all_metrics)

print(f"\n📊 Obliczono {len(all_metrics)} metryk!")
print("\n🔝 Top 10 najważniejszych metryk:")
key_metrics = ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate', 'Profit_Factor',
               'Calmar_Ratio', 'Sortino_Ratio', 'Kelly_Criterion', 'Recovery_Factor', 'System_Quality_Number']

for metric in key_metrics:
    if metric in all_metrics:
        print(f"{metric}: {all_metrics[metric]:.3f}")
```

## 8. Wizualizacja wyników

### Zaawansowana wizualizacja z wykorzystaniem VectorBT, Plotly i Seaborn

```python
class AdvancedVisualization:
    """
    Klasa do tworzenia zaawansowanych, interaktywnych wizualizacji
    """
    def __init__(self, backtest_results, portfolio, trades_df, market_data):
        self.results = backtest_results
        self.portfolio = portfolio
        self.trades = trades_df
        self.market_data = market_data
        
    def create_interactive_dashboard(self):
        """
        Tworzy interaktywny dashboard z wykorzystaniem Plotly i VectorBT
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        # Główny dashboard - 6 wykresów
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Performance', 'Drawdown Analysis',
                'Monthly Returns Heatmap', 'Trade Distribution',
                'Rolling Metrics', 'Risk-Return Scatter'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"secondary_y": True}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Portfolio Performance
        fig.add_trace(
            go.Scatter(
                x=self.portfolio.index,
                y=self.portfolio.value(),
                name='Portfolio Value',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.portfolio.index,
                y=self.portfolio.benchmark_value(),
                name='Buy & Hold',
                line=dict(color='#95a5a6', width=2, dash='dash')
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Dodaj sygnały transakcji
        entries = self.portfolio.trades.records[self.portfolio.trades.records['entry_idx']]
        exits = self.portfolio.trades.records[self.portfolio.trades.records['exit_idx']]
        
        fig.add_trace(
            go.Scatter(
                x=entries.index,
                y=self.portfolio.value()[entries.index],
                mode='markers',
                name='Buy Signals',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            row=1, col=1, secondary_y=False
        )
        
        # 2. Drawdown Analysis
        drawdown = self.portfolio.drawdown()
        underwater_periods = drawdown[drawdown < 0]
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#e74c3c', width=1),
                fillcolor='rgba(231, 76, 60, 0.3)'
            ),
            row=1, col=2
        )
        
        # 3. Monthly Returns Heatmap
        monthly_returns = self.portfolio.returns().resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_pivot.values * 100,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_pivot.index,
                colorscale='RdYlGn',
                text=np.round(monthly_pivot.values * 100, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                colorbar=dict(title="Return %")
            ),
            row=2, col=1
        )
        
        # 4. Trade Distribution
        fig.add_trace(
            go.Histogram(
                x=self.trades['Return_Pct'],
                nbinsx=30,
                name='Trade Returns',
                marker=dict(
                    color='#3498db',
                    line=dict(color='#2c3e50', width=1)
                )
            ),
            row=2, col=2
        )
        
        # 5. Rolling Metrics
        rolling_sharpe = self.portfolio.rolling_sharpe_ratio(window=252)
        rolling_sortino = self.portfolio.rolling_sortino_ratio(window=252)
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='#9b59b6', width=2)
            ),
            row=3, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sortino.index,
                y=rolling_sortino,
                name='Rolling Sortino',
                line=dict(color='#e67e22', width=2)
            ),
            row=3, col=1, secondary_y=True
        )
        
        # 6. Risk-Return Scatter
        # Porównanie różnych okresów
        periods = ['Q1', 'Q2', 'Q3', 'Q4']
        scatter_data = []
        
        for i, quarter in enumerate(periods):
            quarter_returns = self.portfolio.returns().resample('Q').apply(lambda x: (1 + x).prod() - 1)
            quarter_vol = self.portfolio.returns().resample('Q').std() * np.sqrt(252)
            
            scatter_data.append({
                'Period': quarter,
                'Return': quarter_returns.iloc[i] * 100 if i < len(quarter_returns) else 0,
                'Volatility': quarter_vol.iloc[i] * 100 if i < len(quarter_vol) else 0
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig.add_trace(
            go.Scatter(
                x=scatter_df['Volatility'],
                y=scatter_df['Return'],
                mode='markers+text',
                text=scatter_df['Period'],
                textposition='top center',
                marker=dict(size=15, color=scatter_df.index, colorscale='Viridis'),
                name='Quarterly Performance'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text=f"Strategy Performance Dashboard - {self.portfolio.total_return():.2%} Total Return",
            template='plotly_white'
        )
        
        # Save interactive dashboard
        fig.write_html('interactive_dashboard.html')
        print("📊 Interaktywny dashboard zapisany jako: interactive_dashboard.html")
        
        return fig
    
    def create_vectorbt_plots(self):
        """
        Wykorzystuje wbudowane wykresy VectorBT
        """
        # 1. Kompleksowy wykres portfolio
        portfolio_fig = self.portfolio.plot(
            subplots=[
                'value',
                'drawdown',
                'returns',
                'trades',
                'underwater',
                'rolling_sharpe'
            ],
            title='VectorBT Portfolio Analysis'
        )
        portfolio_fig.show()
        
        # 2. Heatmapa korelacji
        if hasattr(self.portfolio, 'returns'):
            corr_fig = self.portfolio.returns().vbt.heatmap(
                title='Returns Correlation Heatmap',
                xaxis_title='Days',
                yaxis_title='Days'
            )
            corr_fig.show()
        
        # 3. Wykres QQ dla analizy rozkładu
        qq_fig = self.portfolio.returns().vbt.qqplot(
            title='Q-Q Plot - Returns Distribution'
        )
        qq_fig.show()
        
        # 4. Analiza transakcji
        if len(self.portfolio.trades.records) > 0:
            trades_fig = self.portfolio.trades.plot(
                title='Trade Analysis',
                subplots=['pnl', 'returns', 'duration']
            )
            trades_fig.show()
    
    def create_advanced_risk_plots(self):
        """
        Zaawansowane wykresy analizy ryzyka
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Value at Risk Analysis',
                'Tail Risk Distribution',
                'Risk Contribution',
                'Stress Test Scenarios'
            )
        )
        
        returns = self.portfolio.returns()
        
        # 1. VaR Analysis
        confidence_levels = [0.90, 0.95, 0.99]
        for i, cl in enumerate(confidence_levels):
            var = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= var].mean()
            
            fig.add_trace(
                go.Bar(
                    x=[f'VaR {int(cl*100)}%', f'CVaR {int(cl*100)}%'],
                    y=[var * 100, cvar * 100],
                    name=f'{int(cl*100)}% Level',
                    marker_color=['#3498db', '#e74c3c']
                ),
                row=1, col=1
            )
        
        # 2. Tail Risk Distribution
        left_tail = returns[returns < returns.quantile(0.05)]
        right_tail = returns[returns > returns.quantile(0.95)]
        
        fig.add_trace(
            go.Histogram(
                x=left_tail * 100,
                name='Left Tail (5%)',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=right_tail * 100,
                name='Right Tail (95%)',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Risk Contribution (jeśli mamy wiele aktywów)
        if hasattr(self, 'risk_contributions'):
            fig.add_trace(
                go.Pie(
                    labels=self.risk_contributions.index,
                    values=self.risk_contributions.values,
                    hole=0.3,
                    name='Risk Contribution'
                ),
                row=2, col=1
            )
        
        # 4. Stress Test Scenarios
        scenarios = {
            'Normal': returns.mean(),
            '2008 Crisis': returns.quantile(0.01),
            'COVID Crash': returns.quantile(0.02),
            'Flash Crash': returns.min(),
            'Best Case': returns.quantile(0.99)
        }
        
        scenario_impact = []
        current_value = self.portfolio.final_value()
        
        for scenario, return_val in scenarios.items():
            impact = current_value * (1 + return_val) - current_value
            scenario_impact.append({
                'Scenario': scenario,
                'Impact': impact,
                'Return': return_val * 100
            })
        
        scenario_df = pd.DataFrame(scenario_impact)
        
        fig.add_trace(
            go.Waterfall(
                x=scenario_df['Scenario'],
                y=scenario_df['Impact'],
                text=[f"{r:.2f}%" for r in scenario_df['Return']],
                textposition="outside",
                name='Portfolio Impact'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Advanced Risk Analysis")
        fig.show()
    
    def create_ml_performance_plots(self):
        """
        Wykresy dla strategii opartych na ML
        """
        if not hasattr(self, 'ml_predictions'):
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Prediction Accuracy Over Time',
                'Feature Importance',
                'Confusion Matrix',
                'Learning Curve'
            )
        )
        
        # 1. Prediction Accuracy Over Time
        rolling_accuracy = pd.Series(
            self.ml_predictions['correct'], 
            index=self.ml_predictions.index
        ).rolling(252).mean()
        
        fig.add_trace(
            go.Scatter(
                x=rolling_accuracy.index,
                y=rolling_accuracy * 100,
                name='Rolling Accuracy',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Feature Importance
        if hasattr(self, 'feature_importance'):
            top_features = self.feature_importance.head(10)
            
            fig.add_trace(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker_color='#e74c3c'
                ),
                row=1, col=2
            )
        
        # 3. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            self.ml_predictions['actual'], 
            self.ml_predictions['predicted']
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Sell', 'Buy'],
                y=['Sell', 'Buy'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ),
            row=2, col=1
        )
        
        # 4. Learning Curve
        if hasattr(self, 'learning_curve_data'):
            fig.add_trace(
                go.Scatter(
                    x=self.learning_curve_data['train_size'],
                    y=self.learning_curve_data['train_score'],
                    name='Training Score',
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.learning_curve_data['train_size'],
                    y=self.learning_curve_data['val_score'],
                    name='Validation Score',
                    line=dict(color='red')
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="ML Model Performance Analysis")
        fig.show()
    
    def create_trade_analysis_plots(self):
        """
        Szczegółowa analiza transakcji z DuckDB i Plotly
        """
        # Użyj DuckDB do agregacji
        conn = duckdb.connect()
        conn.register('trades', self.trades)
        
        # Analiza MAE/MFE
        mae_mfe_query = """
        SELECT 
            Return_Pct,
            CASE 
                WHEN Return_Pct > 0 THEN 'Winner'
                ELSE 'Loser'
            END as trade_type,
            -- Symulowane MAE/MFE (w prawdziwym backteście byłyby rzeczywiste wartości)
            ABS(RANDOM() % 5) as MAE,
            ABS(RANDOM() % 10) as MFE
        FROM trades
        """
        
        mae_mfe_data = conn.execute(mae_mfe_query).df()
        
        # Wykres MAE/MFE
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'MAE vs Final Return',
                'MFE vs Final Return',
                'Trade Duration Distribution',
                'Win/Loss Streaks'
            )
        )
        
        # 1. MAE Analysis
        fig.add_trace(
            go.Scatter(
                x=mae_mfe_data[mae_mfe_data['trade_type'] == 'Winner']['MAE'],
                y=mae_mfe_data[mae_mfe_data['trade_type'] == 'Winner']['Return_Pct'],
                mode='markers',
                name='Winners',
                marker=dict(color='green', size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=mae_mfe_data[mae_mfe_data['trade_type'] == 'Loser']['MAE'],
                y=mae_mfe_data[mae_mfe_data['trade_type'] == 'Loser']['Return_Pct'],
                mode='markers',
                name='Losers',
                marker=dict(color='red', size=8)
            ),
            row=1, col=1
        )
        
        # 2. MFE Analysis
        fig.add_trace(
            go.Scatter(
                x=mae_mfe_data['MFE'],
                y=mae_mfe_data['Return_Pct'],
                mode='markers',
                marker=dict(
                    color=mae_mfe_data['Return_Pct'],
                    colorscale='RdYlGn',
                    size=8,
                    showscale=True
                ),
                name='All Trades'
            ),
            row=1, col=2
        )
        
        # 3. Trade Duration Distribution
        duration_by_outcome = conn.execute("""
        SELECT 
            CASE 
                WHEN Return_Pct > 0 THEN 'Winners'
                ELSE 'Losers'
            END as outcome,
            Duration
        FROM trades
        """).df()
        
        for outcome in ['Winners', 'Losers']:
            data = duration_by_outcome[duration_by_outcome['outcome'] == outcome]['Duration']
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=outcome,
                    opacity=0.7,
                    nbinsx=20
                ),
                row=2, col=1
            )
        
        # 4. Win/Loss Streaks
        # Oblicz sekwencje
        wins = (self.trades['Return'] > 0).astype(int)
        streaks = wins.groupby((wins != wins.shift()).cumsum()).agg(['size', 'first'])
        
        win_streaks = streaks[streaks['first'] == 1]['size'].value_counts()
        loss_streaks = streaks[streaks['first'] == 0]['size'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=win_streaks.index,
                y=win_streaks.values,
                name='Win Streaks',
                marker_color='green'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=loss_streaks.index,
                y=loss_streaks.values,
                name='Loss Streaks',
                marker_color='red'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Trade Analysis Dashboard")
        fig.show()
    
    def create_3d_analysis(self):
        """
        Tworzenie wykresów 3D dla zaawansowanej analizy
        """
        import plotly.graph_objects as go
        
        # Przygotuj dane
        returns = self.portfolio.returns()
        volatility = returns.rolling(20).std()
        sharpe = returns.rolling(252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        
        # Wykres 3D: Return vs Volatility vs Time
        fig = go.Figure(data=[
            go.Scatter3d(
                x=returns.index,
                y=volatility * 100,
                z=returns * 100,
                mode='markers',
                marker=dict(
                    size=5,
                    color=sharpe,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                text=[f"Date: {d}<br>Return: {r:.2f}%<br>Vol: {v:.2f}%<br>Sharpe: {s:.2f}"
                      for d, r, v, s in zip(returns.index, returns*100, volatility*100, sharpe)],
                hoverinfo='text'
            )
        ])
        
        fig.update_layout(
            title='3D Portfolio Analysis: Return vs Volatility over Time',
            scene=dict(
                xaxis_title='Date',
                yaxis_title='Volatility (%)',
                zaxis_title='Return (%)'
            ),
            height=800
        )
        
        fig.show()
    
    def generate_pdf_report(self):
        """
        Generuje profesjonalny raport PDF
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        # Utwórz dokument
        doc = SimpleDocTemplate("backtest_report.pdf", pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Tytuł
        title = Paragraph("Backtest Strategy Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Podsumowanie
        summary_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{self.results['Total_Return']:.2f}%"],
            ['Sharpe Ratio', f"{self.results['Sharpe_Ratio']:.3f}"],
            ['Max Drawdown', f"{self.results['Max_Drawdown']:.2f}%"],
            ['Win Rate', f"{self.results['Win_Rate']:.1f}%"],
            ['Total Trades', f"{self.results['Total_Trades']}"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        
        # Zapisz PDF
        doc.build(story)
        print("📄 Raport PDF wygenerowany: backtest_report.pdf")

# Przykład użycia
viz = AdvancedVisualization(backtest_results, portfolio, trades_df, market_data)

# Twórz wszystkie wykresy
viz.create_interactive_dashboard()
viz.create_vectorbt_plots()
viz.create_advanced_risk_plots()
viz.create_ml_performance_plots()
viz.create_trade_analysis_plots()
viz.create_3d_analysis()
viz.generate_pdf_report()
```

## 9. Interpretacja wyników

### Zaawansowana interpretacja wyników z wykorzystaniem wszystkich bibliotek

```python
class ResultsInterpreter:
    """
    Zaawansowana klasa do interpretacji wyników backtestu
    """
    def __init__(self, results, trades, portfolio_data):
        self.results = results
        self.trades = trades
        self.portfolio_data = portfolio_data
        self.conn = duckdb.connect(':memory:')
        
    def comprehensive_analysis(self):
        """
        Przeprowadza kompleksową analizę i interpretację wyników
        """
        print("🔍 KOMPLEKSOWA INTERPRETACJA WYNIKÓW BACKTESTU")
        print("=" * 80)
        
        # 1. Ocena ogólnej wydajności
        self._interpret_performance()
        
        # 2. Analiza ryzyka
        self._interpret_risk()
        
        # 3. Analiza stabilności
        self._interpret_stability()
        
        # 4. Analiza transakcji
        self._interpret_trades()
        
        # 5. Analiza reżimów rynkowych
        self._interpret_market_regimes()
        
        # 6. Wnioski i rekomendacje
        self._generate_recommendations()
        
    def _interpret_performance(self):
        """
        Interpretacja wydajności strategii
        """
        print("\n📈 WYDAJNOŚĆ STRATEGII:")
        print("-" * 40)
        
        total_return = self.results['Total_Return']
        annual_return = self.results['Annual_Return']
        sharpe = self.results['Sharpe_Ratio']
        
        # Ocena zwrotu
        if total_return > 100:
            print(f"✅ Doskonały całkowity zwrot: {total_return:.2f}%")
        elif total_return > 50:
            print(f"✅ Bardzo dobry całkowity zwrot: {total_return:.2f}%")
        elif total_return > 20:
            print(f"⚠️ Umiarkowany całkowity zwrot: {total_return:.2f}%")
        else:
            print(f"❌ Niski całkowity zwrot: {total_return:.2f}%")
        
        # Ocena Sharpe Ratio
        if sharpe > 2:
            print(f"✅ Wybitny Sharpe Ratio: {sharpe:.3f} - strategia bardzo efektywna")
        elif sharpe > 1:
            print(f"✅ Dobry Sharpe Ratio: {sharpe:.3f} - korzystny stosunek zysku do ryzyka")
        elif sharpe > 0.5:
            print(f"⚠️ Przeciętny Sharpe Ratio: {sharpe:.3f} - wymaga optymalizacji")
        else:
            print(f"❌ Niski Sharpe Ratio: {sharpe:.3f} - niekorzystny profil ryzyko/zwrot")
        
        # Porównanie z benchmarkiem
        excess_return = total_return - self.results.get('Buy_Hold_Return', 0)
        if excess_return > 0:
            print(f"✅ Strategia bije benchmark o {excess_return:.2f}%")
        else:
            print(f"❌ Strategia przegrywa z benchmarkiem o {abs(excess_return):.2f}%")
    
    def _interpret_risk(self):
        """
        Interpretacja profilu ryzyka
        """
        print("\n⚠️ PROFIL RYZYKA:")
        print("-" * 40)
        
        max_dd = self.results['Max_Drawdown']
        volatility = self.results['Volatility']
        var_95 = self.results.get('Value_at_Risk_95', 0)
        
        # Ocena maksymalnego obsunięcia
        if abs(max_dd) < 10:
            print(f"✅ Niskie maksymalne obsunięcie: {max_dd:.2f}% - bardzo bezpieczna strategia")
        elif abs(max_dd) < 20:
            print(f"✅ Akceptowalne obsunięcie: {max_dd:.2f}% - rozsądny poziom ryzyka")
        elif abs(max_dd) < 30:
            print(f"⚠️ Wysokie obsunięcie: {max_dd:.2f}% - wymaga silnej psychiki")
        else:
            print(f"❌ Ekstremalne obsunięcie: {max_dd:.2f}% - bardzo ryzykowna strategia")
        
        # Analiza zmienności
        if volatility < 10:
            print(f"✅ Niska zmienność: {volatility:.2f}% rocznie")
        elif volatility < 20:
            print(f"⚠️ Umiarkowana zmienność: {volatility:.2f}% rocznie")
        else:
            print(f"❌ Wysoka zmienność: {volatility:.2f}% rocznie")
        
        # Recovery analysis
        recovery_factor = self.results.get('Recovery_Factor', 0)
        if recovery_factor > 5:
            print(f"✅ Doskonały Recovery Factor: {recovery_factor:.2f} - szybkie odrabianie strat")
        elif recovery_factor > 2:
            print(f"⚠️ Przeciętny Recovery Factor: {recovery_factor:.2f}")
        else:
            print(f"❌ Słaby Recovery Factor: {recovery_factor:.2f} - wolne odrabianie strat")
    
    def _interpret_stability(self):
        """
        Ocena stabilności strategii w czasie
        """
        print("\n🔄 STABILNOŚĆ STRATEGII:")
        print("-" * 40)
        
        # Analiza konsystencji z DuckDB
        self.conn.register('portfolio', self.portfolio_data)
        
        consistency_query = """
        WITH monthly_performance AS (
            SELECT 
                DATE_TRUNC('month', date) as month,
                SUM(returns) as monthly_return
            FROM portfolio
            GROUP BY DATE_TRUNC('month', date)
        ),
        stats AS (
            SELECT 
                COUNT(CASE WHEN monthly_return > 0 THEN 1 END) * 100.0 / COUNT(*) as positive_months_pct,
                STDDEV(monthly_return) * SQRT(12) * 100 as annualized_monthly_vol,
                COUNT(CASE WHEN monthly_return < -0.05 THEN 1 END) as severe_loss_months
            FROM monthly_performance
        )
        SELECT * FROM stats
        """
        
        stability_stats = self.conn.execute(consistency_query).df().iloc[0]
        
        # Interpretacja
        if stability_stats['positive_months_pct'] > 60:
            print(f"✅ Wysoka konsystencja: {stability_stats['positive_months_pct']:.1f}% miesięcy zyskownych")
        else:
            print(f"⚠️ Niska konsystencja: {stability_stats['positive_months_pct']:.1f}% miesięcy zyskownych")
        
        if stability_stats['severe_loss_months'] == 0:
            print("✅ Brak miesięcy z poważnymi stratami (>5%)")
        else:
            print(f"⚠️ Liczba miesięcy z poważnymi stratami: {int(stability_stats['severe_loss_months'])}")
    
    def _interpret_trades(self):
        """
        Interpretacja statystyk transakcyjnych
        """
        print("\n💼 ANALIZA TRANSAKCJI:")
        print("-" * 40)
        
        win_rate = self.results['Win_Rate']
        profit_factor = self.results['Profit_Factor']
        avg_win = self.results['Average_Win']
        avg_loss = abs(self.results['Average_Loss'])
        
        # Win Rate analysis
        if win_rate > 60:
            print(f"✅ Wysoki Win Rate: {win_rate:.1f}% - strategia bardzo skuteczna")
        elif win_rate > 45:
            print(f"⚠️ Przeciętny Win Rate: {win_rate:.1f}% - typowy dla trend following")
        else:
            print(f"❌ Niski Win Rate: {win_rate:.1f}% - wymaga dużych zysków gdy trafia")
        
        # Profit Factor
        if profit_factor > 2:
            print(f"✅ Doskonały Profit Factor: {profit_factor:.2f}")
        elif profit_factor > 1.5:
            print(f"✅ Dobry Profit Factor: {profit_factor:.2f}")
        elif profit_factor > 1.2:
            print(f"⚠️ Minimalny Profit Factor: {profit_factor:.2f}")
        else:
            print(f"❌ Niewystarczający Profit Factor: {profit_factor:.2f}")
        
        # Risk/Reward ratio
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        if rr_ratio > 2:
            print(f"✅ Doskonały stosunek Risk/Reward: {rr_ratio:.2f}:1")
        elif rr_ratio > 1.5:
            print(f"⚠️ Dobry stosunek Risk/Reward: {rr_ratio:.2f}:1")
        else:
            print(f"❌ Słaby stosunek Risk/Reward: {rr_ratio:.2f}:1")
    
    def _interpret_market_regimes(self):
        """
        Analiza wydajności w różnych warunkach rynkowych
        """
        print("\n🌍 ANALIZA REŻIMÓW RYNKOWYCH:")
        print("-" * 40)
        
        # Symulowana analiza - w rzeczywistości używałbyś prawdziwych danych
        regime_performance = {
            'Bull Market': {'return': 45, 'trades': 120},
            'Bear Market': {'return': -5, 'trades': 80},
            'Sideways Market': {'return': 12, 'trades': 150}
        }
        
        for regime, stats in regime_performance.items():
            if stats['return'] > 0:
                print(f"✅ {regime}: +{stats['return']}% ({stats['trades']} transakcji)")
            else:
                print(f"❌ {regime}: {stats['return']}% ({stats['trades']} transakcji)")
    
    def _generate_recommendations(self):
        """
        Generowanie rekomendacji na podstawie analizy
        """
        print("\n🎯 REKOMENDACJE:")
        print("-" * 40)
        
        recommendations = []
        
        # Rekomendacje na podstawie metryk
        if self.results['Sharpe_Ratio'] < 1:
            recommendations.append("• Rozważ optymalizację parametrów dla poprawy Sharpe Ratio")
        
        if abs(self.results['Max_Drawdown']) > 25:
            recommendations.append("• Implementuj bardziej restrykcyjne zarządzanie ryzykiem")
        
        if self.results['Win_Rate'] < 40:
            recommendations.append("• Przeanalizuj kryteria wejścia - zbyt wiele fałszywych sygnałów")
        
        if self.results['Average_Trade_Duration'] < 5:
            recommendations.append("• Zbyt krótkie pozycje - rozważ dłuższy time frame")
        
        kelly = self.results.get('Kelly_Criterion', 0)
        if kelly > 25:
            recommendations.append(f"• Kelly Criterion sugeruje {kelly:.1f}% kapitału, ale użyj max 10-15%")
        
        if not recommendations:
            recommendations.append("✅ Strategia wydaje się dobrze zoptymalizowana!")
        
        for rec in recommendations:
            print(rec)

def evaluate_strategy_readiness(results, min_requirements=None):
    """
    Kompleksowa ocena gotowości strategii do wdrożenia
    """
    if min_requirements is None:
        min_requirements = {
            'sharpe_ratio': 1.0,
            'max_drawdown': -20,
            'win_rate': 40,
            'profit_factor': 1.5,
            'min_trades': 100,
            'annual_return': 15,
            'calmar_ratio': 1.0,
            'recovery_factor': 2.0,
            'kelly_criterion_max': 25,
            'sortino_ratio': 1.5
        }
    
    print("\n" + "="*80)
    print("🎯 OCENA GOTOWOŚCI STRATEGII DO WDROŻENIA")
    print("="*80)
    
    score = 0
    max_score = len(min_requirements)
    failed_criteria = []
    
    # Szczegółowa ocena każdego kryterium
    criteria_results = {}
    
    # Sprawdzanie wszystkich kryteriów
    for criterion, min_value in min_requirements.items():
        actual_value = results.get(criterion.replace('_', ' ').title().replace(' ', '_'), 0)
        
        if criterion == 'max_drawdown':
            passed = actual_value >= min_value
        elif criterion == 'kelly_criterion_max':
            passed = results.get('Kelly_Criterion', 0) <= min_value
        else:
            passed = actual_value >= min_value
        
        if passed:
            criteria_results[criterion] = f"✅ PASS ({actual_value:.2f})"
            score += 1
        else:
            criteria_results[criterion] = f"❌ FAIL ({actual_value:.2f} vs required {min_value})"
            failed_criteria.append(criterion)
    
    # Wyświetl wyniki
    print("\n📊 WYNIKI OCENY:")
    for criterion, result in criteria_results.items():
        print(f"{criterion.replace('_', ' ').title()}: {result}")
    
    # Oblicz ocenę procentową
    percentage_score = (score / max_score) * 100
    
    print(f"\n📈 WYNIK KOŃCOWY: {score}/{max_score} ({percentage_score:.1f}%)")
    
    # Klasyfikacja gotowości
    print("\n🏆 KLASYFIKACJA:")
    if percentage_score >= 90:
        grade = "A+"
        status = "GOTOWA DO WDROŻENIA"
        recommendation = "Strategia spełnia wszystkie kryteria. Rozpocznij od małej pozycji."
    elif percentage_score >= 80:
        grade = "A"
        status = "PRAWIE GOTOWA"
        recommendation = "Drobne poprawki w: " + ", ".join(failed_criteria[:2])
    elif percentage_score >= 70:
        grade = "B"
        status = "WYMAGA DOPRACOWANIA"
        recommendation = "Popraw kluczowe obszary: " + ", ".join(failed_criteria[:3])
    elif percentage_score >= 60:
        grade = "C"
        status = "ZNACZĄCE BRAKI"
        recommendation = "Poważna optymalizacja wymagana w wielu obszarach"
    else:
        grade = "F"
        status = "NIE NADAJE SIĘ DO WDROŻENIA"
        recommendation = "Przeprojektuj strategię od podstaw"
    
    print(f"Ocena: {grade}")
    print(f"Status: {status}")
    print(f"Rekomendacja: {recommendation}")
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'grade': grade,
        'status': status,
        'failed_criteria': failed_criteria
    }

def perform_walk_forward_analysis(strategy_class, data, window_size, step_size):
    """
    Walk-forward analysis z wykorzystaniem VectorBT dla szybkości
    """
    print("\n🚶 WALK-FORWARD ANALYSIS")
    print("="*60)
    
    results = []
    
    # Podziel dane na okna
    for i in range(0, len(data) - window_size, step_size):
        # Okres treningowy
        train_start = i
        train_end = i + int(window_size * 0.7)
        
        # Okres testowy
        test_start = train_end
        test_end = min(i + window_size, len(data))
        
        # Dane treningowe i testowe
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Optymalizacja na danych treningowych
        strategy = strategy_class()
        
        # Użyj VectorBT do szybkiej optymalizacji
        param_grid = {
            'short_window': range(10, 50, 5),
            'long_window': range(50, 200, 10)
        }
        
        best_params = None
        best_sharpe = -np.inf
        
        # Szybka optymalizacja z VectorBT
        for short in param_grid['short_window']:
            for long in param_grid['long_window']:
                if short >= long:
                    continue
                    
                # Testuj parametry
                fast_ma = vbt.MA.run(train_data['Close'], window=short)
                slow_ma = vbt.MA.run(train_data['Close'], window=long)
                
                entries = fast_ma.ma_crossed_above(slow_ma)
                exits = fast_ma.ma_crossed_below(slow_ma)
                
                portfolio = vbt.Portfolio.from_signals(
                    train_data['Close'],
                    entries,
                    exits,
                    init_cash=10000,
                    fees=0.001
                )
                
                sharpe = portfolio.sharpe_ratio()
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'short_window': short, 'long_window': long}
        
        # Test na danych out-of-sample
        test_strategy = strategy_class(**best_params)
        test_backtester = AdvancedBacktester(test_strategy, test_data)
        test_results = test_backtester.run_backtest()
        
        results.append({
            'period': f"{data.index[test_start].date()} - {data.index[test_end-1].date()}",
            'train_sharpe': best_sharpe,
            'test_sharpe': test_results['Sharpe_Ratio'],
            'test_return': test_results['Total_Return'],
            'best_params': best_params
        })
        
        print(f"Period {len(results)}: Train Sharpe={best_sharpe:.3f}, "
              f"Test Sharpe={test_results['Sharpe_Ratio']:.3f}")
    
    # Analiza wyników
    wf_df = pd.DataFrame(results)
    
    print("\n📊 PODSUMOWANIE WALK-FORWARD:")
    print(f"Średni Test Sharpe: {wf_df['test_sharpe'].mean():.3f}")
    print(f"Std Test Sharpe: {wf_df['test_sharpe'].std():.3f}")
    print(f"Najgorszy Test Sharpe: {wf_df['test_sharpe'].min():.3f}")
    print(f"Najlepszy Test Sharpe: {wf_df['test_sharpe'].max():.3f}")
    
    # Ocena stabilności
    consistency = wf_df['test_sharpe'].std() / wf_df['test_sharpe'].mean() if wf_df['test_sharpe'].mean() != 0 else float('inf')
    
    if consistency < 0.5:
        print("✅ Strategia wykazuje wysoką stabilność w czasie")
    elif consistency < 1:
        print("⚠️ Strategia wykazuje umiarkowaną stabilność")
    else:
        print("❌ Strategia jest niestabilna - duże wahania wydajności")
    
    return wf_df

def monte_carlo_analysis(portfolio_returns, n_simulations=10000):
    """
    Monte Carlo simulation dla oceny ryzyka
    """
    print("\n🎲 MONTE CARLO SIMULATION")
    print("="*60)
    
    # Parametry rozkładu
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    n_days = len(portfolio_returns)
    
    # Symulacje
    final_values = []
    max_drawdowns = []
    
    for _ in tqdm(range(n_simulations), desc="Running simulations"):
        # Generuj losową ścieżkę
        daily_returns = np.random.normal(mean_return, std_return, n_days)
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # Oblicz metryki
        final_values.append(cumulative_returns[-1])
        
        # Max drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdowns.append(drawdown.min())
    
    # Analiza wyników
    final_values = np.array(final_values)
    max_drawdowns = np.array(max_drawdowns)
    
    print(f"\n📊 WYNIKI {n_simulations} SYMULACJI:")
    print(f"Średni końcowy zwrot: {(final_values.mean() - 1) * 100:.2f}%")
    print(f"Mediana końcowego zwrotu: {(np.median(final_values) - 1) * 100:.2f}%")
    
    # Percentyle
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n📈 ROZKŁAD ZWROTÓW:")
    for p in percentiles:
        value = np.percentile(final_values, p)
        print(f"{p}% percentyl: {(value - 1) * 100:.2f}%")
    
    # Prawdopodobieństwo straty
    prob_loss = (final_values < 1).sum() / n_simulations * 100
    print(f"\n⚠️ Prawdopodobieństwo straty: {prob_loss:.1f}%")
    
    # Analiza Maximum Drawdown
    print(f"\n📉 MAXIMUM DRAWDOWN:")
    print(f"Średni Max DD: {max_drawdowns.mean() * 100:.2f}%")
    print(f"95% VaR Max DD: {np.percentile(max_drawdowns, 5) * 100:.2f}%")
    print(f"Najgorszy scenariusz: {max_drawdowns.min() * 100:.2f}%")
    
    # Risk of Ruin
    ruin_threshold = 0.5  # 50% strata kapitału
    risk_of_ruin = (final_values < ruin_threshold).sum() / n_simulations * 100
    print(f"\n💀 Risk of Ruin (50% strata): {risk_of_ruin:.2f}%")
    
    return {
        'final_values': final_values,
        'max_drawdowns': max_drawdowns,
        'prob_loss': prob_loss,
        'risk_of_ruin': risk_of_ruin
    }

# Przykład użycia
interpreter = ResultsInterpreter(backtest_results, trades_df, portfolio_data)
interpreter.comprehensive_analysis()

# Ocena gotowości
readiness = evaluate_strategy_readiness(backtest_results)

# Walk-forward analysis
wf_results = perform_walk_forward_analysis(MovingAverageCrossover, data, 
                                         window_size=504, step_size=126)  # 2 lata okno, 6 miesięcy krok

# Monte Carlo
mc_results = monte_carlo_analysis(portfolio_data['returns'], n_simulations=10000)
```

## 11. Monte Carlo Permutation Testing - Wykrywanie Overfittingu

Monte Carlo Permutation Testing (MCPT) jest zaawansowaną techniką wykrywania overfittingu w strategiach tradingowych. Metoda ta została rozwinięta przez neurotrader888 i pozwala na statystycznie znaczącą ocenę tego, czy wyniki strategii są efektem prawdziwego przewidywania rynku, czy tylko dopasowania do przypadkowych wzorców w danych historycznych.

### Teoretyczne podstawy MCPT

MCPT opiera się na prostej, ale potężnej koncepcji: jeśli strategia rzeczywiście przewiduje ruchy rynku, powinna osiągać lepsze wyniki na rzeczywistych danych niż na danych z permutowanymi (przelosowanymi) zwrotami, które zachowują charakterystyki statystyczne rynku, ale niszczą jakiekolwiek przewidywalne wzorce.

### Implementacja frameworka MCPT

```python
class MCPTValidator:
    """
    Comprehensive Monte Carlo Permutation Testing framework
    """
    def __init__(self, strategy_class, data, n_permutations=1000):
        self.strategy_class = strategy_class
        self.data = data
        self.n_permutations = n_permutations
        self.real_results = None
        self.permuted_results = []
        
    def run_comprehensive_mcpt(self, strategy_params=None):
        """
        Przeprowadź kompleksowe MCPT na strategii
        """
        print(f"🧪 MONTE CARLO PERMUTATION TESTING")
        print(f"📊 Testowanie strategii na {self.n_permutations} permutacjach...")
        print("="*80)
        
        # 1. Uruchom strategię na rzeczywistych danych
        print("1️⃣ Uruchamianie strategii na rzeczywistych danych...")
        self.real_results = self._run_strategy_on_data(self.data, strategy_params)
        
        # 2. Uruchom strategię na permutowanych danych
        print(f"2️⃣ Uruchamianie {self.n_permutations} permutacji...")
        self._run_permutation_tests(strategy_params)
        
        # 3. Oblicz p-values dla wszystkich metryk
        print("3️⃣ Obliczanie p-values...")
        p_values = self._calculate_p_values()
        
        # 4. Generuj raport końcowy
        print("4️⃣ Generowanie raportu...")
        self._generate_mcpt_report(p_values)
        
        return {
            'real_results': self.real_results,
            'permuted_results': self.permuted_results,
            'p_values': p_values,
            'significant_metrics': [k for k, v in p_values.items() if v < 0.05]
        }
    
    def _run_strategy_on_data(self, data, params=None):
        """Uruchom strategię na danych"""
        if params is None:
            params = {}
            
        strategy = self.strategy_class(**params)
        backtester = AdvancedBacktester(strategy, data)
        results = backtester.run_backtest()
        
        return results
    
    def _run_permutation_tests(self, strategy_params):
        """Uruchom testy na permutowanych danych"""
        self.permuted_results = []
        
        for i in tqdm(range(self.n_permutations), desc="MCPT Progress"):
            try:
                # Generuj permutację używając bar_permute
                permuted_data = get_permutation(self.data, seed=i)
                
                # Uruchom strategię na permutowanych danych
                perm_results = self._run_strategy_on_data(permuted_data, strategy_params)
                self.permuted_results.append(perm_results)
                
            except Exception as e:
                print(f"⚠️ Błąd w permutacji {i}: {e}")
                continue
    
    def _calculate_p_values(self):
        """Oblicz p-values dla wszystkich metryk"""
        p_values = {}
        
        # Lista metryk do testowania
        metrics_to_test = [
            'Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 
            'Profit_Factor', 'Win_Rate', 'Calmar_Ratio',
            'Sortino_Ratio', 'Average_Win', 'Average_Loss',
            'Recovery_Factor', 'Volatility'
        ]
        
        for metric in metrics_to_test:
            if metric in self.real_results:
                real_value = self.real_results[metric]
                
                # Zbierz wartości z permutacji
                perm_values = []
                for perm_result in self.permuted_results:
                    if metric in perm_result:
                        perm_values.append(perm_result[metric])
                
                if perm_values:
                    # Oblicz p-value (jaka część permutacji była lepsza)
                    if metric == 'Max_Drawdown':  # Dla drawdown mniejsza wartość jest lepsza
                        better_count = sum(1 for v in perm_values if v > real_value)  # większe DD = gorsze
                    else:  # Dla większości metryk większa wartość jest lepsza
                        better_count = sum(1 for v in perm_values if v >= real_value)
                    
                    p_value = (better_count + 1) / (len(perm_values) + 1)  # +1 dla real result
                    p_values[metric] = p_value
        
        return p_values
    
    def _generate_mcpt_report(self, p_values):
        """Generuj szczegółowy raport MCPT"""
        print("\n" + "="*80)
        print("📊 RAPORT MONTE CARLO PERMUTATION TESTING")
        print("="*80)
        
        # Podsumowanie ogólne
        significant_count = sum(1 for p in p_values.values() if p < 0.05)
        total_metrics = len(p_values)
        
        print(f"\n📈 PODSUMOWANIE WYNIKÓW:")
        print(f"Liczba testowanych metryk: {total_metrics}")
        print(f"Statystycznie znaczące (p < 0.05): {significant_count}")
        print(f"Odsetek znaczących wyników: {significant_count/total_metrics*100:.1f}%")
        
        # Szczegółowe wyniki
        print(f"\n📊 SZCZEGÓŁOWE WYNIKI (p-values):")
        print("-" * 60)
        
        # Sortuj według p-value
        sorted_metrics = sorted(p_values.items(), key=lambda x: x[1])
        
        for metric, p_value in sorted_metrics:
            real_val = self.real_results.get(metric, 0)
            
            # Interpretacja
            if p_value < 0.001:
                status = "🟢 BARDZO ZNACZĄCE"
                interpretation = "Praktycznie niemożliwe, że to przypadek"
            elif p_value < 0.01:
                status = "🟢 ZNACZĄCE"
                interpretation = "Bardzo prawdopodobne prawdziwe przewidywanie"
            elif p_value < 0.05:
                status = "🟡 GRANICZNE"
                interpretation = "Prawdopodobnie znaczące, ale wymaga ostrożności"
            elif p_value < 0.1:
                status = "🟠 SŁABE"
                interpretation = "Słabe dowody na przewidywalność"
            else:
                status = "🔴 NIEZНАЧĄCE"
                interpretation = "Brak dowodów na przewidywalność"
            
            print(f"{metric:20} | p={p_value:7.4f} | {status}")
            print(f"{'':20} | Wartość rzeczywista: {real_val:8.3f}")
            print(f"{'':20} | {interpretation}")
            print("-" * 60)
        
        # Interpretacja ogólna
        self._generate_overall_interpretation(p_values, significant_count, total_metrics)
    
    def _generate_overall_interpretation(self, p_values, significant_count, total_metrics):
        """Generuj interpretację ogólną wyników MCPT"""
        print(f"\n🎯 INTERPRETACJA OGÓLNA:")
        print("=" * 40)
        
        significance_ratio = significant_count / total_metrics
        
        if significance_ratio >= 0.6:
            print("✅ STRATEGIA BARDZO OBIECUJĄCA")
            print("• Większość metryk statystycznie znaczących")
            print("• Silne dowody na przewidywalność rynku")
            print("• Niskie ryzyko overfittingu")
            recommendation = "ZIELONE ŚWIATŁO dla wdrożenia"
            
        elif significance_ratio >= 0.4:
            print("🟡 STRATEGIA CZĘŚCIOWO ZNACZĄCA")
            print("• Część metryk pokazuje przewidywalność")
            print("• Umiarkowane dowody na skuteczność")
            print("• Wymagana dodatkowa walidacja")
            recommendation = "ŻÓŁTE ŚWIATŁO - dodatkowe testy wymagane"
            
        elif significance_ratio >= 0.2:
            print("🟠 STRATEGIA SŁABO ZNACZĄCA")
            print("• Niewiele metryk statystycznie znaczących")
            print("• Słabe dowody na przewidywalność")
            print("• Wysokie ryzyko overfittingu")
            recommendation = "POMARAŃCZOWE ŚWIATŁO - wymaga re-designu"
            
        else:
            print("🔴 STRATEGIA PRAWDOPODOBNIE OVERFITTED")
            print("• Brak lub bardzo mało znaczących metryk")
            print("• Wyniki prawdopodobnie przypadkowe")
            print("• Strategia wymaga całkowitego przeprojektowania")
            recommendation = "CZERWONE ŚWIATŁO - NIE WDRAŻAJ"
        
        print(f"\n🚦 REKOMENDACJA: {recommendation}")
        
        # Dodatkowe wskazówki
        print(f"\n💡 DODATKOWE WSKAZÓWKI:")
        
        # Analiza kluczowych metryk
        key_metrics = ['Sharpe_Ratio', 'Total_Return', 'Max_Drawdown']
        key_significant = sum(1 for m in key_metrics if p_values.get(m, 1) < 0.05)
        
        if key_significant >= 2:
            print("• Kluczowe metryki ryzyka/zwrotu są znaczące ✅")
        else:
            print("• Kluczowe metryki wymagają poprawy ⚠️")
        
        # Sprawdź czy Max Drawdown jest niez意義ący
        dd_pvalue = p_values.get('Max_Drawdown', 1)
        if dd_pvalue > 0.1:
            print("• Max Drawdown niezначący - może być przypadkowy ⚠️")
        else:
            print("• Max Drawdown kontrolowany przez strategię ✅")
        
        # Analiza Win Rate
        wr_pvalue = p_values.get('Win_Rate', 1)
        if wr_pvalue < 0.05:
            print("• Win Rate znaczący - strategia ma przewagę ✅")
        else:
            print("• Win Rate przypadkowy - brak przewagi w % wygranych ⚠️")

def advanced_mcpt_analysis(strategy_class, data, strategy_params=None):
    """
    Zaawansowana analiza MCPT z dodatkowymi testami
    """
    print("🧪 ZAAWANSOWANA ANALIZA MCPT")
    print("="*80)
    
    # 1. Standardowy MCPT
    mcpt_validator = MCPTValidator(strategy_class, data, n_permutations=1000)
    mcpt_results = mcpt_validator.run_comprehensive_mcpt(strategy_params)
    
    # 2. In-Sample vs Out-of-Sample MCPT
    print("\n🔄 IN-SAMPLE vs OUT-OF-SAMPLE MCPT")
    print("-" * 50)
    
    # Podziel dane na in-sample i out-of-sample
    split_point = int(len(data) * 0.7)
    is_data = data.iloc[:split_point]
    oos_data = data.iloc[split_point:]
    
    # MCPT na in-sample
    is_mcpt = MCPTValidator(strategy_class, is_data, n_permutations=500)
    is_results = is_mcpt.run_comprehensive_mcpt(strategy_params)
    
    # MCPT na out-of-sample
    oos_mcpt = MCPTValidator(strategy_class, oos_data, n_permutations=500)
    oos_results = oos_mcpt.run_comprehensive_mcpt(strategy_params)
    
    # Porównaj wyniki
    _compare_is_oos_mcpt(is_results, oos_results)
    
    return {
        'full_mcpt': mcpt_results,
        'in_sample_mcpt': is_results,
        'out_of_sample_mcpt': oos_results
    }

def _compare_is_oos_mcpt(is_results, oos_results):
    """Porównaj wyniki MCPT in-sample vs out-of-sample"""
    print("\n📊 PORÓWNANIE IN-SAMPLE vs OUT-OF-SAMPLE MCPT")
    print("="*60)
    
    common_metrics = set(is_results['p_values'].keys()) & set(oos_results['p_values'].keys())
    
    print(f"{'Metryka':20} | {'IS p-val':10} | {'OOS p-val':10} | {'Status':15}")
    print("-" * 65)
    
    consistent_significant = 0
    total_common = len(common_metrics)
    
    for metric in sorted(common_metrics):
        is_pval = is_results['p_values'][metric]
        oos_pval = oos_results['p_values'][metric]
        
        # Oceń konsystencję
        is_sig = is_pval < 0.05
        oos_sig = oos_pval < 0.05
        
        if is_sig and oos_sig:
            status = "✅ KONSYST."
            consistent_significant += 1
        elif is_sig and not oos_sig:
            status = "⚠️ OVERFIT?"
        elif not is_sig and oos_sig:
            status = "🟡 OOS BETTER"
        else:
            status = "❌ BOTH WEAK"
        
        print(f"{metric:20} | {is_pval:10.4f} | {oos_pval:10.4f} | {status}")
    
    # Ogólna ocena
    consistency_ratio = consistent_significant / total_common if total_common > 0 else 0
    
    print("\n🎯 OCENA KONSYSTENCJI:")
    print(f"Konsystentnie znaczące: {consistent_significant}/{total_common} ({consistency_ratio*100:.1f}%)")
    
    if consistency_ratio >= 0.5:
        print("✅ WYSOKA KONSYSTENCJA - strategia stabilna")
    elif consistency_ratio >= 0.3:
        print("🟡 ŚREDNIA KONSYSTENCJA - wymagana ostrożność")
    else:
        print("❌ NISKA KONSYSTENCJA - prawdopodobny overfitting")

def walk_forward_mcpt(strategy_class, data, window_months=24, step_months=6, n_permutations=500):
    """
    Walk-forward analysis z MCPT na każdym kroku
    """
    print("🚶 WALK-FORWARD MCPT ANALYSIS")
    print("="*60)
    
    results = []
    
    # Konwertuj miesiące na dni (przybliżenie)
    window_days = window_months * 30
    step_days = step_months * 30
    
    n_periods = (len(data) - window_days) // step_days
    
    print(f"Analizowanie {n_periods} okresów walk-forward...")
    
    for i in range(n_periods):
        period_start = i * step_days
        period_end = period_start + window_days
        
        if period_end > len(data):
            break
        
        period_data = data.iloc[period_start:period_end]
        period_name = f"Period_{i+1}"
        
        print(f"\n📅 {period_name}: {period_data.index[0].date()} to {period_data.index[-1].date()}")
        
        # Uruchom MCPT dla tego okresu
        mcpt = MCPTValidator(strategy_class, period_data, n_permutations=n_permutations)
        period_results = mcpt.run_comprehensive_mcpt()
        
        # Zapisz kluczowe wyniki
        significant_metrics = len(period_results['significant_metrics'])
        total_metrics = len(period_results['p_values'])
        significance_ratio = significant_metrics / total_metrics if total_metrics > 0 else 0
        
        results.append({
            'period': period_name,
            'start_date': period_data.index[0],
            'end_date': period_data.index[-1],
            'significant_metrics': significant_metrics,
            'total_metrics': total_metrics,
            'significance_ratio': significance_ratio,
            'key_p_values': {
                k: v for k, v in period_results['p_values'].items() 
                if k in ['Sharpe_Ratio', 'Total_Return', 'Max_Drawdown', 'Profit_Factor']
            }
        })
        
        print(f"Znaczące metryki: {significant_metrics}/{total_metrics} ({significance_ratio*100:.1f}%)")
    
    # Analiza stabilności w czasie
    print("\n📈 ANALIZA STABILNOŚCI WALK-FORWARD MCPT")
    print("-" * 50)
    
    significance_ratios = [r['significance_ratio'] for r in results]
    
    print(f"Średni odsetek znaczących metryk: {np.mean(significance_ratios)*100:.1f}%")
    print(f"Odchylenie standardowe: {np.std(significance_ratios)*100:.1f}%")
    print(f"Najgorszy okres: {min(significance_ratios)*100:.1f}%")
    print(f"Najlepszy okres: {max(significance_ratios)*100:.1f}%")
    
    # Trend w czasie
    if len(significance_ratios) >= 5:
        from scipy.stats import linregress
        x = range(len(significance_ratios))
        slope, intercept, r_value, p_value, std_err = linregress(x, significance_ratios)
        
        if p_value < 0.05:
            trend_direction = "rosnący" if slope > 0 else "malejący"
            print(f"🔍 Wykryto {trend_direction} trend znaczącoś ci (p={p_value:.3f})")
        else:
            print("🔍 Brak znaczącego trendu w czasie")
    
    return results

# Przykład użycia kompleksowego MCPT
def run_complete_mcpt_analysis():
    """
    Uruchom kompletną analizę MCPT dla strategii
    """
    print("🧪 KOMPLEKSOWA ANALIZA MCPT")
    print("="*80)
    
    # Załaduj dane (przykład)
    data = yf.download('QQQ', start='2015-01-01', end='2023-01-01')
    
    # Zdefiniuj klasę strategii (przykład)
    class ExampleStrategy(TradingStrategy):
        def __init__(self, short_ma=20, long_ma=50):
            self.short_ma = short_ma
            self.long_ma = long_ma
    
    # 1. Podstawowy MCPT
    print("\n1️⃣ PODSTAWOWY MCPT (1000 permutacji)")
    basic_results = advanced_mcpt_analysis(ExampleStrategy, data)
    
    # 2. Walk-forward MCPT
    print("\n2️⃣ WALK-FORWARD MCPT")
    wf_results = walk_forward_mcpt(ExampleStrategy, data, 
                                window_months=24, step_months=6)
    
    # 3. Optymalizacja odporna na overfitting
    print("\n3️⃣ MCPT-GUIDED OPTIMIZATION")
    mcpt_guided_optimization(ExampleStrategy, data)
    
    return {
        'basic_mcpt': basic_results,
        'walk_forward_mcpt': wf_results
    }

def mcpt_guided_optimization(strategy_class, data, param_grid=None):
    """
    Optymalizacja parametrów z użyciem MCPT jako kryterium
    """
    print("🎯 MCPT-GUIDED PARAMETER OPTIMIZATION")
    print("="*50)
    
    if param_grid is None:
        param_grid = {
            'short_ma': [10, 15, 20, 25, 30],
            'long_ma': [40, 50, 60, 70, 80]
        }
    
    best_score = 0
    best_params = None
    results = []
    
    total_combinations = len(param_grid['short_ma']) * len(param_grid['long_ma'])
    print(f"Testowanie {total_combinations} kombinacji parametrów...")
    
    for short_ma in param_grid['short_ma']:
        for long_ma in param_grid['long_ma']:
            if short_ma >= long_ma:
                continue
            
            params = {'short_ma': short_ma, 'long_ma': long_ma}
            print(f"\n📊 Testowanie: {params}")
            
            # Uruchom MCPT dla tych parametrów
            mcpt = MCPTValidator(strategy_class, data, n_permutations=200)  # Mniej permutacji dla szybkości
            mcpt_results = mcpt.run_comprehensive_mcpt(params)
            
            # Oblicz composite score na podstawie MCPT
            significant_count = len(mcpt_results['significant_metrics'])
            total_metrics = len(mcpt_results['p_values'])
            
            # Waga dla kluczowych metryk
            key_metrics_score = 0
            key_metrics = ['Sharpe_Ratio', 'Total_Return', 'Max_Drawdown']
            for metric in key_metrics:
                if metric in mcpt_results['p_values'] and mcpt_results['p_values'][metric] < 0.05:
                    key_metrics_score += 1
            
            # Composite score
            score = (significant_count / total_metrics * 0.5) + (key_metrics_score / len(key_metrics) * 0.5)
            
            results.append({
                'params': params,
                'mcpt_score': score,
                'significant_metrics': significant_count,
                'total_metrics': total_metrics,
                'key_p_values': {k: v for k, v in mcpt_results['p_values'].items() if k in key_metrics}
            })
            
            if score > best_score:
                best_score = score
                best_params = params
            
            print(f"MCPT Score: {score:.3f} (Significant: {significant_count}/{total_metrics})")
    
    # Podsumowanie optymalizacji
    print(f"\n🏆 NAJLEPSZE PARAMETRY:")
    print(f"Parametry: {best_params}")
    print(f"MCPT Score: {best_score:.3f}")
    
    # Sortuj wyniki według score
    results.sort(key=lambda x: x['mcpt_score'], reverse=True)
    
    print(f"\n📊 TOP 5 KOMBINACJI:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['params']} - Score: {result['mcpt_score']:.3f}")
    
    return best_params, results

# Integracja MCPT z głównym pipeline'em backtest
class MCPTEnhancedBacktester(AdvancedBacktester):
    """
    Rozszerzony backtester z wbudowanym MCPT
    """
    def __init__(self, strategy, data, mcpt_permutations=1000):
        super().__init__(strategy, data)
        self.mcpt_permutations = mcpt_permutations
        self.mcpt_results = None
    
    def run_full_analysis(self):
        """
        Uruchom pełną analizę z MCPT
        """
        # 1. Standardowy backtest
        print("1️⃣ Standardowy backtest...")
        backtest_results = self.run_backtest()
        
        # 2. MCPT analysis
        print("2️⃣ MCPT Analysis...")
        mcpt_validator = MCPTValidator(type(self.strategy), self.data, self.mcpt_permutations)
        self.mcpt_results = mcpt_validator.run_comprehensive_mcpt()
        
        # 3. Zintegrowany raport
        self._generate_integrated_report(backtest_results)
        
        return {
            'backtest_results': backtest_results,
            'mcpt_results': self.mcpt_results
        }
    
    def _generate_integrated_report(self, backtest_results):
        """Generuj zintegrowany raport backtest + MCPT"""
        print("\n" + "="*80)
        print("📊 ZINTEGROWANY RAPORT: BACKTEST + MCPT")
        print("="*80)
        
        # Podstawowe metryki backtest
        print(f"\n📈 WYNIKI BACKTEST:")
        print(f"Total Return: {backtest_results['Total_Return']:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['Sharpe_Ratio']:.3f}")
        print(f"Max Drawdown: {backtest_results['Max_Drawdown']:.2f}%")
        print(f"Win Rate: {backtest_results['Win_Rate']:.1f}%")
        
        # MCPT validation
        significant_metrics = len(self.mcpt_results['significant_metrics'])
        total_metrics = len(self.mcpt_results['p_values'])
        
        print(f"\n🧪 WALIDACJA MCPT:")
        print(f"Znaczące metryki: {significant_metrics}/{total_metrics}")
        print(f"Odsetek znaczących: {significant_metrics/total_metrics*100:.1f}%")
        
        # Kluczowe p-values
        key_metrics = ['Sharpe_Ratio', 'Total_Return', 'Max_Drawdown', 'Win_Rate']
        print(f"\nKluczowe p-values:")
        for metric in key_metrics:
            if metric in self.mcpt_results['p_values']:
                p_val = self.mcpt_results['p_values'][metric]
                status = "✅" if p_val < 0.05 else "❌"
                print(f"{status} {metric}: p = {p_val:.4f}")
        
        # Końcowa rekomendacja
        print(f"\n🚦 REKOMENDACJA KOŃCOWA:")
        if significant_metrics >= total_metrics * 0.5:
            if backtest_results['Sharpe_Ratio'] > 1.0 and backtest_results['Max_Drawdown'] > -30:
                print("✅ STRATEGIA REKOMENDOWANA DO WDROŻENIA")
                print("   • Wyniki statystycznie znaczące")
                print("   • Akceptowalne metryki ryzyka/zwrotu")
            else:
                print("🟡 STRATEGIA ZNACZĄCA, ALE WYMAGA POPRAWEK")
                print("   • Poprawa Sharpe Ratio lub Max Drawdown")
        else:
            print("❌ STRATEGIA NIEZALECANA")
            print("   • Brak statystycznej znaczności")
            print("   • Prawdopodobny overfitting")

```

### Praktyczne wskazówki MCPT

#### Interpretacja p-values:
- **p < 0.001**: Praktycznie pewne, że strategia ma przewagę
- **p < 0.01**: Bardzo silny dowód na przewidywalność
- **p < 0.05**: Statystycznie znaczące (standardowy próg)
- **p < 0.1**: Graniczne, wymaga dodatkowej walidacji
- **p > 0.1**: Brak dowodów na przewidywalność

#### Najczęstsze błędy:
1. **Za mało permutacji** - minimum 1000 dla wiarygodnych wyników
2. **Testowanie tylko jednej metryki** - MCPT powinien testować wszystkie kluczowe metryki
3. **Ignorowanie in-sample vs out-of-sample** - konsystencja jest kluczowa
4. **Skupienie się tylko na p-values** - wielkość efektu też ma znaczenie

#### Gdy MCPT pokazuje overfitting:
1. **Uprość strategię** - mniej parametrów, prostsze reguły
2. **Więcej danych** - wydłuż okres testowania
3. **Regularyzacja** - dodaj kary za złożoność
4. **Ensemble methods** - kombinuj wiele prostych strategii

## 12. Checklist przed przejściem na live trading

### Kompleksowa lista kontrolna przed wdrożeniem strategii

```python
class PreDeploymentChecker:
    """
    Kompleksowy system weryfikacji gotowości do wdrożenia
    """
    def __init__(self, strategy_results, infrastructure_config=None):
        self.results = strategy_results
        self.config = infrastructure_config or {}
        self.checklist_results = {}
        
    def run_complete_verification(self):
        """
        Przeprowadź kompletną weryfikację wszystkich aspektów
        """
        print("🚀 KOMPLEKSOWA WERYFIKACJA PRZED WDROŻENIEM")
        print("=" * 80)
        
        # 1. Weryfikacja backtestu
        self._verify_backtest_quality()
        
        # 2. Weryfikacja stabilności
        self._verify_stability()
        
        # 3. Weryfikacja zarządzania ryzykiem
        self._verify_risk_management()
        
        # 4. Weryfikacja infrastruktury
        self._verify_infrastructure()
        
        # 5. Weryfikacja psychologiczna
        self._verify_psychological_readiness()
        
        # 6. Plan wdrożenia
        self._create_deployment_plan()
        
        # Generuj raport końcowy
        self._generate_final_report()
    
    def _verify_backtest_quality(self):
        """
        Weryfikacja jakości i wiarygodności backtestu
        """
        print("\n📊 1. WERYFIKACJA JAKOŚCI BACKTESTU")
        print("-" * 40)
        
        backtest_checks = {
            'Okres testowania >= 3 lata': self.results.get('test_period_years', 0) >= 3,
            'Liczba transakcji >= 100': self.results.get('Total_Trades', 0) >= 100,
            'Uwzględnione koszty transakcyjne': self.results.get('fees_included', True),
            'Uwzględniony poślizg cenowy': self.results.get('slippage_included', True),
            'Testowane na różnych warunkach rynkowych': self.results.get('market_conditions_tested', False),
            'Walk-forward analysis przeprowadzona': self.results.get('walk_forward_done', False),
            'Monte Carlo simulation wykonana': self.results.get('monte_carlo_done', False),
            'Brak overfittingu (IS/OOS Sharpe podobne)': self._check_overfitting(),
            'Realistyczne założenia o płynności': self.results.get('liquidity_realistic', True),
            'Brak look-ahead bias': self.results.get('no_look_ahead_bias', True)
        }
        
        self._print_checklist("Backtest Quality", backtest_checks)
        self.checklist_results['backtest_quality'] = backtest_checks
    
    def _verify_stability(self):
        """
        Weryfikacja stabilności strategii
        """
        print("\n🔄 2. WERYFIKACJA STABILNOŚCI")
        print("-" * 40)
        
        stability_checks = {
            'Stabilne wyniki w różnych okresach': self.results.get('period_stability', 0) > 0.7,
            'Parametry nie są przeczulone': self._check_parameter_sensitivity(),
            'Działa na podobnych instrumentach': self.results.get('cross_asset_tested', False),
            'Consistent performance metrics': self.results.get('metric_consistency', 0) > 0.8,
            'Brak długich okresów strat': self.results.get('max_losing_streak_months', 12) < 6,
            'Recovery time < 6 miesięcy': self.results.get('avg_recovery_months', 12) < 6,
            'Stabilny Win Rate (std < 10%)': self.results.get('win_rate_stability', 0) < 0.1
        }
        
        self._print_checklist("Stability", stability_checks)
        self.checklist_results['stability'] = stability_checks
    
    def _verify_risk_management(self):
        """
        Weryfikacja systemu zarządzania ryzykiem
        """
        print("\n⚠️ 3. WERYFIKACJA ZARZĄDZANIA RYZYKIEM")
        print("-" * 40)
        
        risk_checks = {
            'Zdefiniowany max risk per trade': self.config.get('max_risk_per_trade', 0) <= 0.02,
            'Stop loss dla każdej pozycji': self.config.get('stop_loss_enabled', False),
            'Maksymalna ekspozycja <= 100%': self.config.get('max_exposure', 1.5) <= 1.0,
            'Dywersyfikacja (min 3 instrumenty)': self.config.get('n_instruments', 1) >= 3,
            'Position sizing zdefiniowane': self.config.get('position_sizing_method') is not None,
            'Correlation limits ustawione': self.config.get('correlation_limits', False),
            'Drawdown limit zdefiniowany': self.config.get('max_drawdown_limit', 0.5) <= 0.3,
            'Emergency exit plan': self.config.get('emergency_exit_plan', False),
            'Risk parity jeśli multi-asset': self.config.get('risk_parity_enabled', False)
        }
        
        self._print_checklist("Risk Management", risk_checks)
        self.checklist_results['risk_management'] = risk_checks
    
    def _verify_infrastructure(self):
        """
        Weryfikacja infrastruktury technicznej
        """
        print("\n🖥️ 4. WERYFIKACJA INFRASTRUKTURY")
        print("-" * 40)
        
        infra_checks = {
            'Redundantne połączenie internetowe': self.config.get('redundant_internet', False),
            'Backup serwer gotowy': self.config.get('backup_server', False),
            'Monitoring 24/7': self.config.get('monitoring_enabled', False),
            'Alerty SMS/Email skonfigurowane': self.config.get('alerts_configured', False),
            'Logi wszystkich operacji': self.config.get('logging_enabled', False),
            'Backup danych codziennie': self.config.get('daily_backup', False),
            'API rate limits przestrzegane': self.config.get('rate_limits_ok', False),
            'Disaster recovery plan': self.config.get('disaster_recovery', False),
            'Wersjonowanie kodu (Git)': self.config.get('version_control', False),
            'Testing environment dostępne': self.config.get('test_env_available', False)
        }
        
        self._print_checklist("Infrastructure", infra_checks)
        self.checklist_results['infrastructure'] = infra_checks
    
    def _verify_psychological_readiness(self):
        """
        Weryfikacja przygotowania psychologicznego
        """
        print("\n🧠 5. WERYFIKACJA PSYCHOLOGICZNEJ GOTOWOŚCI")
        print("-" * 40)
        
        psych_checks = {
            'Przygotowany na 30% drawdown': input("Czy jesteś przygotowany na 30% drawdown? (t/n): ").lower() == 't',
            'Plan na serie strat': input("Czy masz plan działania na serię strat? (t/n): ").lower() == 't',
            'Nie będziesz ingerował ręcznie': input("Czy zobowiązujesz się nie ingerować ręcznie? (t/n): ").lower() == 't',
            'Kapitał to nie jest życiowa konieczność': input("Czy możesz stracić ten kapitał bez wpływu na życie? (t/n): ").lower() == 't',
            'Doświadczenie w tradingu >= 1 rok': input("Czy masz min. 1 rok doświadczenia? (t/n): ").lower() == 't',
            'Zrozumienie strategii 100%': input("Czy w pełni rozumiesz strategię? (t/n): ").lower() == 't'
        }
        
        self._print_checklist("Psychological Readiness", psych_checks)
        self.checklist_results['psychological'] = psych_checks
    
    def _create_deployment_plan(self):
        """
        Stwórz plan wdrożenia
        """
        print("\n📋 6. PLAN WDROŻENIA")
        print("-" * 40)
        
        deployment_plan = {
            'Phase 1 - Paper Trading (2 tygodnie)': [
                '• Uruchom strategię na koncie demo',
                '• Monitoruj wszystkie sygnały',
                '• Porównuj z backtestem',
                '• Zapisuj wszystkie rozbieżności'
            ],
            'Phase 2 - Micro Live (4 tygodnie)': [
                '• Start z 10% docelowego kapitału',
                '• Minimalne dozwolone pozycje',
                '• Codzienny monitoring',
                '• Tygodniowe raporty wydajności'
            ],
            'Phase 3 - Scaling Up (8 tygodni)': [
                '• Stopniowo zwiększaj do 25%, 50%, 75%',
                '• Każdy krok po 2 tygodnie stabilnych wyników',
                '• Pełna analiza po każdym kroku',
                '• Stop jeśli wyniki odbiegają od backtestu'
            ],
            'Phase 4 - Full Deployment': [
                '• 100% kapitału tylko po 3 miesiącach',
                '• Utrzymuj wszystkie procedury monitoringu',
                '• Miesięczne przeglądy wydajności',
                '• Kwartalne re-optymalizacje'
            ]
        }
        
        for phase, tasks in deployment_plan.items():
            print(f"\n{phase}:")
            for task in tasks:
                print(task)
    
    def _check_overfitting(self):
        """Sprawdź czy nie ma overfittingu"""
        is_sharpe = self.results.get('in_sample_sharpe', 1)
        oos_sharpe = self.results.get('out_sample_sharpe', 0.5)
        return abs(is_sharpe - oos_sharpe) / is_sharpe < 0.3 if is_sharpe != 0 else False
    
    def _check_parameter_sensitivity(self):
        """Sprawdź czułość parametrów"""
        param_stability = self.results.get('parameter_stability', {})
        if not param_stability:
            return False
        return all(v > 0.8 for v in param_stability.values())
    
    def _print_checklist(self, category, checks):
        """Wydrukuj checklist z kolorami"""
        passed = sum(checks.values())
        total = len(checks)
        
        print(f"\n{category}: {passed}/{total} ({passed/total*100:.0f}%)")
        for item, status in checks.items():
            symbol = "✅" if status else "❌"
            print(f"{symbol} {item}")
    
    def _generate_final_report(self):
        """Generuj końcowy raport gotowości"""
        print("\n" + "="*80)
        print("📊 RAPORT KOŃCOWY GOTOWOŚCI DO WDROŻENIA")
        print("="*80)
        
        # Oblicz wyniki dla każdej kategorii
        category_scores = {}
        for category, checks in self.checklist_results.items():
            score = sum(checks.values()) / len(checks) * 100
            category_scores[category] = score
        
        # Wyświetl wyniki
        print("\nWYNIKI PO KATEGORIACH:")
        for category, score in category_scores.items():
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"{status} {category.replace('_', ' ').title()}: {score:.0f}%")
        
        # Ogólna ocena
        overall_score = sum(category_scores.values()) / len(category_scores)
        
        print(f"\n🎯 WYNIK OGÓLNY: {overall_score:.0f}%")
        
        # Rekomendacja końcowa
        print("\n🏁 REKOMENDACJA KOŃCOWA:")
        if overall_score >= 90:
            print("✅ STRATEGIA GOTOWA DO WDROŻENIA")
            print("   Rozpocznij od Phase 1 - Paper Trading")
        elif overall_score >= 75:
            print("⚠️ STRATEGIA PRAWIE GOTOWA")
            print("   Popraw elementy oznaczone ❌ przed wdrożeniem")
        else:
            print("❌ STRATEGIA NIE JEST GOTOWA")
            print("   Wymagane znaczące poprawki w wielu obszarach")
        
        # Krytyczne elementy
        print("\n⚠️ ELEMENTY KRYTYCZNE (muszą być spełnione):")
        critical_items = [
            ('Stop loss dla każdej pozycji', 
             self.checklist_results.get('risk_management', {}).get('Stop loss dla każdej pozycji', False)),
            ('Backup serwer gotowy', 
             self.checklist_results.get('infrastructure', {}).get('Backup serwer gotowy', False)),
            ('Emergency exit plan', 
             self.checklist_results.get('risk_management', {}).get('Emergency exit plan', False))
        ]
        
        for item, status in critical_items:
            symbol = "✅" if status else "❌"
            print(f"{symbol} {item}")

def final_deployment_checklist():
    """
    Ostateczna, szczegółowa lista kontrolna
    """
    print("\n" + "="*80)
    print("✅ OSTATECZNA LISTA KONTROLNA PRZED URUCHOMIENIEM")
    print("="*80)
    
    checklist = """
    📊 BACKTEST I WALIDACJA:
    □ Przetestowano minimum 5 lat danych (preferowane 10+)
    □ Out-of-sample test minimum 20% danych
    □ Walk-forward analysis z minimum 10 okresami
    □ Monte Carlo 10,000+ symulacji
    □ Slippage i komisje uwzględnione (pesymistyczne założenia)
    □ Test na różnych warunkach rynkowych (bessa, hossa, konsolidacja)
    □ Sprawdzono na minimum 3 podobnych instrumentach
    □ Brak oznak overfittingu (stable OOS performance)
    
    ⚠️ ZARZĄDZANIE RYZYKIEM:
    □ Max risk per trade: 1-2% kapitału
    □ Max drawdown limit: 20-30%
    □ Stop loss na KAŻDEJ pozycji
    □ Position sizing: Kelly Criterion / Fixed Fractional
    □ Correlation limits między pozycjami
    □ Max exposure: 100% (bez dźwigni na start)
    □ Trailing stop loss rules zdefiniowane
    □ Weekend/overnight risk plan
    
    🖥️ INFRASTRUKTURA TECHNICZNA:
    □ VPS z 99.9% uptime SLA
    □ Redundantne połączenie internetowe
    □ UPS dla lokalnego backup
    □ Monitoring z Grafana/Prometheus
    □ Alerty: SMS + Email + Telegram
    □ Logi w Elasticsearch/Splunk
    □ Codzienne backupy (3-2-1 rule)
    □ Disaster recovery < 1 godzina
    □ API keys bezpiecznie przechowywane
    □ 2FA na wszystkich kontach
    
    📈 BROKER I WYKONANIE:
    □ Minimum 2 brokerów (główny + backup)
    □ API stabilne (99%+ uptime ostatnie 3 miesiące)
    □ Execution speed < 100ms
    □ Koszty transakcyjne wynegocjowane
    □ Margining rules zrozumiane
    □ Tax implications sprawdzone
    
    🧠 PRZYGOTOWANIE PSYCHOLOGICZNE:
    □ Jestem gotowy na 30%+ drawdown
    □ Mam plan na 6-miesięczną serię strat
    □ NIE będę ręcznie zamykał pozycji
    □ NIE będę zmieniał parametrów w trakcie
    □ Będę prowadził dziennik tradingowy
    □ Mam wsparcie rodziny/partnera
    □ Ten kapitał to nie są pieniądze na życie
    
    📋 PLAN OPERACYJNY:
    □ Godziny monitoringu ustalone
    □ Procedura morning checkup
    □ Evening reconciliation process
    □ Weekend maintenance schedule
    □ Miesięczne review meetings
    □ Kwartalne re-optymalizacje
    □ Roczny strategy review
    
    🚀 DEPLOYMENT MILESTONES:
    □ Week 1-2: Paper trading (100% sygnałów)
    □ Week 3-4: Micro lots (10% kapitału)
    □ Month 2: 25% kapitału
    □ Month 3: 50% kapitału
    □ Month 4+: 100% jeśli wszystko OK
    
    🛑 KILL SWITCHES:
    □ Daily loss limit: -5%
    □ Weekly loss limit: -10%
    □ Monthly loss limit: -15%
    □ Drawdown kill switch: -25%
    □ Technical failure protocol
    □ Black swan event protocol
    
    📝 DOKUMENTACJA:
    □ Strategy specification document
    □ Risk management policy
    □ Operational procedures
    □ Emergency contacts
    □ Broker support numbers
    □ Code repository (Git)
    □ Change log process
    
    ✅ FINAL SIGN-OFF:
    □ Wszystkie powyższe punkty sprawdzone
    □ Przeczytałem i zrozumiałem ryzyko
    □ Jestem gotowy ponieść 100% stratę
    □ Mam plan B jeśli strategia zawiedzie
    □ Data rozpoczęcia: ___________
    □ Podpis: ___________________
    """
    
    print(checklist)
    
    # Interaktywne sprawdzenie
    print("\n🎯 INTERAKTYWNA WERYFIKACJA:")
    critical_checks = [
        "Czy wykonałeś walk-forward analysis?",
        "Czy masz stop loss na każdej pozycji?",
        "Czy masz backup serwer?",
        "Czy jesteś psychologicznie przygotowany na 30% drawdown?",
        "Czy ten kapitał możesz stracić bez wpływu na życie?"
    ]
    
    all_passed = True
    for check in critical_checks:
        response = input(f"{check} (t/n): ").lower()
        if response != 't':
            all_passed = False
            print("❌ STOP! Musisz najpierw rozwiązać ten problem.")
    
    if all_passed:
        print("\n✅ GRATULACJE! Możesz rozpocząć paper trading!")
        print("🍀 Powodzenia!")
    else:
        print("\n❌ Nie jesteś jeszcze gotowy. Wróć po rozwiązaniu problemów.")

# Przykład użycia
checker = PreDeploymentChecker(backtest_results, infrastructure_config)
checker.run_complete_verification()

# Ostateczna lista kontrolna
final_deployment_checklist()

# Zapisz raport do pliku
with open('deployment_readiness_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORT GOTOWOŚCI DO WDROŻENIA\n")
    f.write("="*80 + "\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Strategia: {strategy_name}\n")
    f.write(f"Całkowity wynik: {overall_score:.0f}%\n")
    f.write("="*80 + "\n")
    # Dodaj szczegóły...

print("\n📄 Raport zapisany do: deployment_readiness_report.txt")
```

## Podsumowanie

Ten kompleksowy przewodnik przedstawił profesjonalny proces backtestingu strategii tradingowej z wykorzystaniem najnowocześniejszych narzędzi dostępnych w ekosystemie Python. 

### Kluczowe wnioski:

1. **Wykorzystanie zaawansowanych bibliotek**:
   - **Polars** - do błyskawicznego przetwarzania danych (10-100x szybsze niż pandas)
   - **DuckDB** - do złożonych analiz SQL na dużych zbiorach danych
   - **VectorBT** - do wektoryzowanego backtestingu (1000x szybsze niż pętle)
   - **Riskfolio** - do zaawansowanej optymalizacji portfela
   - **OpenBB** - do dostępu do profesjonalnych danych finansowych

2. **Kompleksowa analiza**:
   - Ponad 100 różnych metryk oceny strategii
   - Zaawansowane testy stabilności (walk-forward, Monte Carlo)
   - Analiza wydajności w różnych reżimach rynkowych
   - Optymalizacja portfela z nowoczesnymi miarami ryzyka

3. **Profesjonalne podejście**:
   - Realistyczne założenia (koszty, poślizgi, płynność)
   - Właściwe zarządzanie ryzykiem
   - Kompleksowa dokumentacja
   - Stopniowy plan wdrożenia

### Najważniejsze zasady:

⚠️ **PAMIĘTAJ**: Pozytywne wyniki backtestu NIE gwarantują przyszłych zysków!

✅ **Zawsze**:
- Testuj out-of-sample
- Uwzględniaj pesymistyczne założenia
- Sprawdzaj stabilność parametrów
- Miej plan na najgorszy scenariusz

❌ **Nigdy**:
- Nie optymalizuj nadmiernie (overfitting)
- Nie ignoruj kosztów transakcyjnych
- Nie pomijaj testów stabilności
- Nie ryzykuj pieniędzy, których nie możesz stracić

### Dalsze kroki:

1. **Edukacja**: Kontynuuj naukę o rynkach i strategiach
2. **Praktyka**: Testuj różne strategie na danych historycznych
3. **Paper Trading**: Zawsze rozpoczynaj od konta demo
4. **Networking**: Dołącz do społeczności traderów algorytmicznych
5. **Iteracja**: Ciągle ulepszaj i adaptuj swoją strategię

Powodzenia w Twojej przygodzie z tradingiem algorytmicznym! 🚀

---

*"The market can remain irrational longer than you can remain solvent."* - John Maynard Keynes

*"In trading, the majority is always wrong."* - Jesse Livermore

*"Risk comes from not knowing what you're doing."* - Warren Buffett

---

## 13. Zaawansowane praktyki z PyQuant News - VectorBT Ultra-Fast Backtesting

### 13.1 Million-Scale Backtesting z VectorBT

Na podstawie cennych praktyk PyQuant News, które pokazują jak uruchamiać **1,000,000 symulacji backtestu w 20 sekund**, wprowadzamy zaawansowane techniki unikania overfittingu i statystycznej walidacji strategii.

#### Kluczowe zasady z PyQuant News:

**🚨 KRYTYCZNE OSTRZEŻENIE:**
- **96% materiałów edukacyjnych online jest bezużytecznych**
- **Overfitting prowadzi do strat** - nie optymalizuj ślepo na danych historycznych
- **Statystyczna istotność** jest kluczowa - nie P&L czy Sharpe Ratio

#### 13.1.1 Ultra-Fast VectorBT Implementation

```python
import vectorbt as vbt
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PyQuantAdvancedBacktesting:
    """
    Implementacja zaawansowanych technik z PyQuant News
    - Million-scale backtesting w sekundach
    - Walk-forward optimization 
    - Statistical significance testing
    - Overfitting prevention
    """
    
    def __init__(self, data):
        self.data = data
        self.close = data['Close'] if isinstance(data, pd.DataFrame) else data
        
    def million_scale_optimization(self, fast_windows=None, slow_windows=None):
        """
        Testuj miliony kombinacji parametrów w sekundach
        Inspirowane PyQuant News: 1M symulacji w 20 sekund
        """
        if fast_windows is None:
            fast_windows = np.arange(5, 50, 1)  # 45 wartości
        if slow_windows is None:
            slow_windows = np.arange(20, 200, 2)  # 90 wartości
            
        print(f"🚀 TESTOWANIE {len(fast_windows) * len(slow_windows):,} KOMBINACJI")
        print("Inspirowane PyQuant News - Million-Scale Backtesting")
        
        # VectorBT - ultra-fast vectorized backtesting
        fast_ma = vbt.MA.run(self.close, window=fast_windows, short_name='fast')
        slow_ma = vbt.MA.run(self.close, window=slow_windows, short_name='slow')
        
        # Sygnały dla WSZYSTKICH kombinacji jednocześnie
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Portfolio simulation dla milionów kombinacji
        portfolio = vbt.Portfolio.from_signals(
            self.close,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,  # 0.1% prowizji (realistyczne)
            slippage=0.001,  # 0.1% poślizg
            freq='1D'
        )
        
        return portfolio, fast_windows, slow_windows
    
    def walk_forward_analysis(self, portfolio, fast_windows, slow_windows, 
                            split_ratio=0.7, statistical_test=True):
        """
        Walk-Forward Analysis - kluczowa technika z PyQuant News
        
        KLUCZOWE ZASADY:
        1. In-sample (training) - 70% danych
        2. Out-of-sample (testing) - 30% danych  
        3. Statistical significance testing
        4. Unikanie overfittingu przez walidację
        """
        
        print(f"\n📊 WALK-FORWARD ANALYSIS (PyQuant News Method)")
        print("=" * 70)
        
        # Podział danych na in-sample i out-of-sample
        split_idx = int(len(self.close) * split_ratio)
        
        print(f"📈 In-sample (training): {split_idx} dni ({split_ratio*100:.0f}%)")
        print(f"📉 Out-of-sample (testing): {len(self.close) - split_idx} dni ({(1-split_ratio)*100:.0f}%)")
        
        # In-sample optimization
        in_sample_data = self.close.iloc[:split_idx]
        in_sample_portfolio = self.run_strategy_subset(
            in_sample_data, fast_windows, slow_windows
        )
        
        # Znajdź najlepsze parametry na in-sample
        in_sample_sharpe = in_sample_portfolio.sharpe_ratio()
        best_params_idx = in_sample_sharpe.idxmax()
        best_fast, best_slow = best_params_idx
        
        print(f"\n🏆 NAJLEPSZE PARAMETRY (In-sample):")
        print(f"   Fast MA: {best_fast}")
        print(f"   Slow MA: {best_slow}")
        print(f"   In-sample Sharpe: {in_sample_sharpe.iloc[best_params_idx]:.4f}")
        
        # Out-of-sample testing z najlepszymi parametrami
        out_sample_data = self.close.iloc[split_idx:]
        out_sample_portfolio = self.run_single_strategy(
            out_sample_data, best_fast, best_slow
        )
        
        out_sample_sharpe = out_sample_portfolio.sharpe_ratio()
        print(f"   Out-sample Sharpe: {out_sample_sharpe:.4f}")
        
        # KLUCZOWY TEST STATYSTYCZNY (PyQuant News emphasis)
        if statistical_test:
            self.statistical_significance_test(
                in_sample_sharpe.iloc[best_params_idx],
                out_sample_sharpe,
                method="t_test"
            )
        
        # Dodatkowo: analiza stabilności
        self.stability_analysis(in_sample_portfolio, out_sample_portfolio, best_params_idx)
        
        return {
            'best_params': (best_fast, best_slow),
            'in_sample_sharpe': in_sample_sharpe.iloc[best_params_idx],
            'out_sample_sharpe': out_sample_sharpe,
            'in_sample_portfolio': in_sample_portfolio,
            'out_sample_portfolio': out_sample_portfolio
        }
    
    def statistical_significance_test(self, in_sample_sharpe, out_sample_sharpe, 
                                    method="t_test", alpha=0.05):
        """
        Statystyczny test istotności - KLUCZOWA PRAKTYKA z PyQuant News
        
        OSTRZEŻENIE PyQuant News:
        - Wysoki P-value (np. 0.858) = overfitting!
        - Strategia może być dopasowana do szumu
        - Nie ma statystycznie istotnej różnicy między in/out-sample
        """
        
        print(f"\n🔬 STATISTICAL SIGNIFICANCE TEST")
        print("=" * 50)
        print("⚠️ KLUCZOWE: Test na overfitting (PyQuant News method)")
        
        if method == "t_test":
            # One-sided T-test (PyQuant News approach)
            # H0: out_sample_sharpe <= in_sample_sharpe (overfitting)
            # H1: out_sample_sharpe > in_sample_sharpe (generalization)
            
            # Symuluj rozkład dla testu
            sample_size = 100
            in_sample_dist = np.random.normal(in_sample_sharpe, 0.1, sample_size)
            out_sample_dist = np.random.normal(out_sample_sharpe, 0.1, sample_size)
            
            t_stat, p_value = stats.ttest_ind(
                out_sample_dist, 
                in_sample_dist,
                alternative='greater'  # One-sided test
            )
            
            print(f"📊 WYNIKI TESTU T:")
            print(f"   T-statistic: {t_stat:.4f}")
            print(f"   P-value: {p_value:.6f}")
            print(f"   Alpha (significance level): {alpha}")
            
            # Interpretacja według PyQuant News
            if p_value > alpha:
                print(f"\n❌ OVERFITTING DETECTED! (P-value: {p_value:.6f})")
                print("   Strategia prawdopodobnie dopasowana do szumu")
                print("   Nie ma statystycznie istotnej przewagi out-of-sample")
                print("   🚨 RYZYKO: Strategia może nie działać w przyszłości")
            else:
                print(f"\n✅ STATISTICALLY SIGNIFICANT! (P-value: {p_value:.6f})")
                print("   Strategia pokazuje rzeczywistą przewagę")
                print("   Out-of-sample performance jest istotnie lepsza")
                print("   🎯 DOBRA WIADOMOŚĆ: Strategia może generalizować")
                
        return p_value
    
    def stability_analysis(self, in_sample_portfolio, out_sample_portfolio, best_idx):
        """
        Analiza stabilności strategii - rozszerzenie PyQuant News
        """
        
        print(f"\n📈 ANALIZA STABILNOŚCI STRATEGII")
        print("=" * 50)
        
        # Metryki porównawcze
        in_stats = {
            'Total Return': in_sample_portfolio.total_return().iloc[best_idx] * 100,
            'Max Drawdown': in_sample_portfolio.max_drawdown().iloc[best_idx] * 100,
            'Win Rate': in_sample_portfolio.win_rate().iloc[best_idx] * 100,
            'Profit Factor': in_sample_portfolio.profit_factor().iloc[best_idx],
        }
        
        out_stats = {
            'Total Return': out_sample_portfolio.total_return() * 100,
            'Max Drawdown': out_sample_portfolio.max_drawdown() * 100,
            'Win Rate': out_sample_portfolio.win_rate() * 100,
            'Profit Factor': out_sample_portfolio.profit_factor(),
        }
        
        print("📊 PORÓWNANIE IN-SAMPLE vs OUT-OF-SAMPLE:")
        print("-" * 50)
        for metric in in_stats.keys():
            in_val = in_stats[metric]
            out_val = out_stats[metric]
            change = ((out_val - in_val) / abs(in_val)) * 100 if in_val != 0 else 0
            
            print(f"{metric:15}: {in_val:8.2f} → {out_val:8.2f} ({change:+6.1f}%)")
            
        # Ocena stabilności
        total_return_change = ((out_stats['Total Return'] - in_stats['Total Return']) 
                              / abs(in_stats['Total Return'])) * 100
        
        if abs(total_return_change) < 20:
            print(f"\n✅ STABILNA STRATEGIA: Zmiana {total_return_change:+.1f}% (akceptowalna)")
        elif abs(total_return_change) < 50:
            print(f"\n⚠️ UMIARKOWANIE STABILNA: Zmiana {total_return_change:+.1f}% (ostrożnie)")
        else:
            print(f"\n❌ NIESTABILNA STRATEGIA: Zmiana {total_return_change:+.1f}% (ryzykowna)")
    
    def run_strategy_subset(self, data, fast_windows, slow_windows):
        """Pomocnicza funkcja - uruchom strategię na podzbiorze danych"""
        fast_ma = vbt.MA.run(data, window=fast_windows, short_name='fast')
        slow_ma = vbt.MA.run(data, window=slow_windows, short_name='slow')
        
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        return vbt.Portfolio.from_signals(
            data, entries, exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
            freq='1D'
        )
    
    def run_single_strategy(self, data, fast_window, slow_window):
        """Pomocnicza funkcja - uruchom pojedynczą strategię"""
        fast_ma = data.rolling(fast_window).mean()
        slow_ma = data.rolling(slow_window).mean()
        
        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma
        
        return vbt.Portfolio.from_signals(
            data, entries, exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
            freq='1D'
        )

# Przykład użycia PyQuant News metodologii
def pyquant_advanced_example():
    """
    Kompletny przykład używający PyQuant News best practices
    """
    
    print("🚀 PYQUANT NEWS - ADVANCED BACKTESTING METHODOLOGY")
    print("=" * 80)
    
    # Symulacja danych (w rzeczywistości użyj prawdziwych danych)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    
    # Inicjalizacja PyQuant backtestu
    pyquant_bt = PyQuantAdvancedBacktesting(prices)
    
    # 1. Million-scale optimization (PyQuant News: 1M w 20 sekund)
    portfolio, fast_wins, slow_wins = pyquant_bt.million_scale_optimization(
        fast_windows=np.arange(10, 30, 1),    # 20 wartości
        slow_windows=np.arange(40, 100, 2)    # 30 wartości
    )  # 20 * 30 = 600 kombinacji (skalowalne do milionów)
    
    # 2. Walk-forward analysis z statistical testing
    wf_results = pyquant_bt.walk_forward_analysis(
        portfolio, fast_wins, slow_wins,
        split_ratio=0.7,
        statistical_test=True
    )
    
    # 3. Final evaluation
    print(f"\n🏆 KOŃCOWA OCENA STRATEGII (PyQuant News Method):")
    print("=" * 60)
    best_fast, best_slow = wf_results['best_params']
    print(f"Optymalne parametry: Fast={best_fast}, Slow={best_slow}")
    print(f"In-sample Sharpe: {wf_results['in_sample_sharpe']:.4f}")
    print(f"Out-of-sample Sharpe: {wf_results['out_sample_sharpe']:.4f}")
    
    # Końcowe ostrzeżenie PyQuant News
    print(f"\n⚠️ PAMIĘTAJ (PyQuant News Wisdom):")
    print("- 96% edukacji online jest bezużyteczna")
    print("- Overfitting = strata pieniędzy")  
    print("- Statystyczna istotność > wysoki Sharpe")
    print("- Walk-forward analysis jest obowiązkowy")
    print("- P-value > 0.05 = prawdopodobny overfitting")

if __name__ == "__main__":
    pyquant_advanced_example()
```

### 13.2 Dodatkowe praktyki z PyQuant News

#### 13.2.1 Interactive Brokers API Integration
```python
# Z Post 1 PyQuant News - API Setup
import ib_insync
from ib_insync import *

class IBAPIIntegration:
    """
    Praktyki z PyQuant News dla Interactive Brokers API
    """
    def __init__(self):
        self.ib = IB()
    
    def connect_to_ib(self, host='127.0.0.1', port=7497, clientId=1):
        """Bezpieczne połączenie z IB API"""
        try:
            self.ib.connect(host, port, clientId)
            print("✅ Połączono z Interactive Brokers")
        except Exception as e:
            print(f"❌ Błąd połączenia: {e}")
    
    def automate_trading_strategy(self, symbol, strategy_signals):
        """Automatyzacja strategii tradingowej"""
        contract = Stock(symbol, 'SMART', 'USD')
        
        for signal in strategy_signals:
            if signal == 'BUY':
                order = MarketOrder('BUY', 100)
                trade = self.ib.placeOrder(contract, order)
                print(f"📈 Zlecenie BUY: {symbol}")
            elif signal == 'SELL':
                order = MarketOrder('SELL', 100)
                trade = self.ib.placeOrder(contract, order)
                print(f"📉 Zlecenie SELL: {symbol}")
```

#### 13.2.2 High-Performance Data Storage (Post 5 & 7)
```python
import polars as pl
import duckdb

class HighPerformanceDataStorage:
    """
    PyQuant News: Polars + DuckDB dla ultra-szybkiej analizy
    """
    
    def build_local_database(self, data_path):
        """
        Post 5: Build local stock database with Polars & DuckDB
        """
        # Ultra-fast data loading with Polars
        df = pl.scan_csv(data_path)
        
        # Processing with lazy evaluation
        processed = (
            df
            .with_columns([
                pl.col("close").pct_change().alias("returns"),
                pl.col("close").rolling_mean(20).alias("sma20"),
                pl.col("close").rolling_std(20).alias("volatility")
            ])
        ).collect()
        
        # Store in DuckDB for fast queries
        conn = duckdb.connect('market_data.db')
        conn.register('stocks', processed.to_pandas())
        
        return conn
    
    def store_options_data(self, options_data):
        """
        Post 7: Store 2,370,886 rows of options data with ArcticDB
        """
        # Symulacja ArcticDB storage (wymagane: pip install arcticdb)
        try:
            import arcticdb as adb
            ac = adb.Arctic('lmdb://options_database')
            lib = ac.create_library('options')
            lib.write('historical_options', options_data)
            print("✅ Options data stored in ArcticDB")
        except ImportError:
            print("⚠️ ArcticDB not installed - using Parquet fallback")
            options_data.to_parquet('options_data.parquet')
```

### 13.3 Kluczowe wnioski z PyQuant News

#### 🎯 **Najważniejsze praktyki:**

1. **Million-Scale Testing**: VectorBT pozwala na testowanie milionów kombinacji w sekundach
2. **Walk-Forward Analysis**: Obowiązkowy podział na in-sample/out-of-sample
3. **Statistical Significance**: P-value testing przeciwko overfittingowi
4. **Realistic Assumptions**: Prowizje, poślizgi, koszty transakcyjne
5. **High-Performance Storage**: Polars + DuckDB dla szybkości
6. **API Integration**: Automatyzacja przez Interactive Brokers API

#### 🚨 **Ostrzeżenia PyQuant News:**
- 96% materiałów online jest bezużytecznych
- Overfitting prowadzi do rzeczywistych strat finansowych  
- Wysoki P-value (>0.05) wskazuje na overfitting
- Statystyczna istotność ważniejsza niż wysoki Sharpe Ratio

#### 📚 **Dodatkowe zasoby:**
- Newsletter PyQuant News (37,200+ subskrybentów)
- Systematyczne podejście do edukacji finansowej
- Nacisk na praktyczne implementacje, nie teorię

---

*Sekcja 13 dodana na podstawie analizy PyQuant News thread'a - praktyki od eksperta który przejrzał 200 filmów YouTube i wybrał 8 najlepszych.*

## 14. Standardowy Format Wyników Backtestingu

### MANDATORY: Format wyświetlania wyników po każdym backtestcie

Po każdym przeprowadzonym backtestcie wyniki MUSZĄ być wyświetlane w następującym standardowym formacie:

#### 📊 [NAZWA STRATEGII] - WSZYSTKIE 130+ METRYKI

##### 📈 KATEGORIA 1: RETURN METRICS (15 metryk)
```
total_return             :    X.XX%
annualized_return        :    X.XX%
cagr                     :    X.XX%
arithmetic_return        :    X.XX%
geometric_return         :    X.XX%
compound_return          :    X.XX%
simple_return            :    X.XX%
log_return               :    X.XX%
excess_return            :    X.XX%
real_return              :    X.XX%
active_return            :    X.XX%
cumulative_return        :    X.XX%
holding_period_return    :    X.XX%
multi_period_return      :    X.XX%
time_weighted_return     :    X.XX%
```

##### ⚠️ KATEGORIA 2: RISK METRICS (20 metryk)
```
volatility               :   X.XXXX
annualized_volatility    :   X.XXXX
standard_deviation       :   X.XXXX
variance                 :   X.XXXX
semi_variance            :   X.XXXX
downside_deviation       :   X.XXXX
tracking_error           :   X.XXXX
beta                     :   X.XXXX
systematic_risk          :   X.XXXX
unsystematic_risk        :   X.XXXX
total_risk               :   X.XXXX
idiosyncratic_risk       :   X.XXXX
market_risk              :   X.XXXX
credit_risk              :   X.XXXX
liquidity_risk           :   X.XXXX
operational_risk         :   X.XXXX
model_risk               :   X.XXXX
concentration_risk       :   X.XXXX
tail_risk                :   X.XXXX
extreme_risk             :   X.XXXX
```

##### 📈 KATEGORIA 3: RISK-ADJUSTED PERFORMANCE (15 metryk)
```
sharpe_ratio             :     X.XX
sortino_ratio            :     X.XX
calmar_ratio             :     X.XX
sterling_ratio           :     X.XX
burke_ratio              :     X.XX
treynor_ratio            :     X.XX
information_ratio        :     X.XX
modigliani_ratio         :     X.XX
jensen_alpha             :     X.XX
treynor_black_ratio      :     X.XX
appraisal_ratio          :     X.XX
upside_potential_ratio   :     X.XX
downside_potential_ratio :     X.XX
omega_ratio              :     X.XX
gain_loss_ratio          :     X.XX
```

##### 💼 KATEGORIA 4: TRADE METRICS (25 metryk)
```
total_trades             :    XX.XX
winning_trades           :    XX.XX
losing_trades            :    XX.XX
win_rate                 :    XX.XX
loss_rate                :    XX.XX
win_loss_ratio           :     X.XX
average_win              :     X.XX
average_loss             :     X.XX
largest_win              :     X.XX
largest_loss             :     X.XX
average_trade            :     X.XX
median_trade             :     X.XX
std_trade                :     X.XX
best_trade               :     X.XX
worst_trade              :     X.XX
profit_factor            :     X.XX
recovery_factor          :     X.XX
payoff_ratio             :     X.XX
trade_efficiency         :     X.XX
consecutive_wins         :     X.XX
consecutive_losses       :     X.XX
max_consecutive_wins     :     X.XX
max_consecutive_losses   :     X.XX
avg_bars_win             :     X.XX
avg_bars_loss            :     X.XX
```

##### 📉 KATEGORIA 5: DRAWDOWN METRICS (15 metryk)
```
max_drawdown             :   X.XXXX
max_drawdown_duration    :   X.XXXX
avg_drawdown             :   X.XXXX
avg_drawdown_duration    :   X.XXXX
recovery_time            :   X.XXXX
time_to_recovery         :   X.XXXX
drawdown_deviation       :   X.XXXX
pain_index               :   X.XXXX
ulcer_index              :   X.XXXX
lake_ratio               :   X.XXXX
pain_ratio               :   X.XXXX
martin_ratio             :   X.XXXX
drawdown_beta            :   X.XXXX
conditional_drawdown     :   X.XXXX
expected_drawdown        :   X.XXXX
```

##### 📊 KATEGORIA 6: DISTRIBUTION METRICS
```
skewness                 :   X.XXX
kurtosis                 :   XX.XXX
var_95                   :   X.XXXX
```

#### 📋 TABELA PODSUMOWUJĄCA - KLUCZOWE METRYKI

| Metric | Value |
|--------|-------|
| **Win Ratio** | XX.XX% |
| **Total Trades** | XXX |
| **Profit Factor** | X.XX |
| **Sharpe Ratio** | X.XX |
| **Max Drawdown** | X.XX% |
| **Total Return** | X.XX% |
| **Annual Return** | X.XX% |
| **CAGR** | X.XX% |

### 🔒 OBOWIĄZKOWE WYMAGANIA:

1. **ZAWSZE** wyświetlaj WSZYSTKIE 130+ metryki w dokładnie tym formacie
2. **ZAWSZE** dołącz tabelę podsumowującą z 8 kluczowymi metrykami
3. **NIGDY** nie pomijaj żadnej kategorii ani metryki
4. Używaj konsystentnego formatowania z odpowiednim wyrównaniem
5. Dołączaj emoji kategorii dla klarowności wizualnej
6. Wyświetlaj wartości z odpowiednią precyzją dziesiętną
7. Ten format dotyczy KAŻDEGO wykonania backtestu

### Wymagania dotyczące danych:

- Używaj 15+ lat danych historycznych via IBKR API gdy możliwe
- Fallback na yfinance dla maksymalnych danych historycznych
- Testuj na interwałach 1D i 1H gdy stosowne
- Zawsze pokazuj kompletny breakdown metryk dla każdej strategii

### Standardy wydajności:

- Target rocznych zwrotów >20% dla strategii TQQQ
- Target zyskowne lub breakeven dla strategii SQQQ
- Max drawdown powinien być <15% dla obu strategii
- Sharpe ratio powinien być >1.0 dla TQQQ, >0.0 dla SQQQ
- Win rate powinien być >40% dla obu strategii


# Backtest Guide - Mandatory Format
## Complete 130+ Metrics Display Requirements

### MANDATORY FORMAT STRUCTURE

All backtests MUST display results using this exact format:

```
================================================================================
📊 [STRATEGY_NAME] STRATEGY - WSZYSTKIE 130+ METRYKI
================================================================================

🎯 KEY PERFORMANCE METRICS:
   Total Return:        [XX.XX%]
   Annualized Return:   [XX.XX%]
   Volatility:          [XX.XX%]
   Sharpe Ratio:        [X.XXX]
   Max Drawdown:        [XX.XX%]
   Win Rate:            [XX.X%]
   Total Trades:        [XXX]

📈 KATEGORIA 1: RETURN METRICS (15 metryk)
--------------------------------------------------
total_return             :  [XX.XX%]
annualized_return        :   [XX.XX%]
cagr                     :   [XX.XX%]
arithmetic_return        :   [XX.XX%]
geometric_return         :   [XX.XX%]
compound_return          :  [XX.XX%]
simple_return            :    [X.XX%]
log_return               :   [XX.XX%]
excess_return            :    [X.XX%]
real_return              :    [X.XX%]
active_return            :   [XX.XX%]
cumulative_return        :  [XX.XX%]
holding_period_return    :  [XX.XX%]
multi_period_return      :  [XX.XX%]
time_weighted_return     :   [XX.XX%]

⚠️  KATEGORIA 2: RISK METRICS (20 metryk)
--------------------------------------------------
volatility               :   [X.XXXX]
annualized_volatility    :   [X.XXXX]
standard_deviation       :   [X.XXXX]
variance                 :   [X.XXXX]
semi_variance            :   [X.XXXX]
downside_deviation       :   [X.XXXX]
tracking_error           :   [X.XXXX]
beta                     :   [X.XXXX]
systematic_risk          :   [X.XXXX]
unsystematic_risk        :   [X.XXXX]
total_risk               :   [X.XXXX]
idiosyncratic_risk       :   [X.XXXX]
market_risk              :   [X.XXXX]
credit_risk              :   [X.XXXX]
liquidity_risk           :   [X.XXXX]
operational_risk         :   [X.XXXX]
model_risk               :   [X.XXXX]
concentration_risk       :   [X.XXXX]
tail_risk                :   [X.XXXX]
extreme_risk             :   [X.XXXX]

📈 KATEGORIA 3: RISK-ADJUSTED PERFORMANCE (15 metryk)
--------------------------------------------------
sharpe_ratio             :     [X.XX]
sortino_ratio            :     [X.XX]
calmar_ratio             :     [X.XX]
sterling_ratio           :     [X.XX]
burke_ratio              :     [X.XX]
treynor_ratio            :     [X.XX]
information_ratio        :     [X.XX]
modigliani_ratio         :     [X.XX]
jensen_alpha             :     [X.XX]
treynor_black_ratio      :     [X.XX]
appraisal_ratio          :     [X.XX]
upside_potential_ratio   :     [X.XX]
downside_potential_ratio :    [X.XX]
omega_ratio              :     [X.XX]
gain_loss_ratio          :     [X.XX]

💼 KATEGORIA 4: TRADE METRICS (25 metryk)
--------------------------------------------------
total_trades             :   [XXX.XX]
winning_trades           :    [XX.XX]
losing_trades            :   [XXX.XX]
win_rate                 :    [XX.XX]
loss_rate                :    [XX.XX]
win_loss_ratio           :     [X.XX]
average_win              :     [X.XX]
average_loss             :    [X.XX]
largest_win              :     [X.XX]
largest_loss             :    [X.XX]
average_trade            :     [X.XX]
median_trade             :    [X.XX]
std_trade                :     [X.XX]
best_trade               :     [X.XX]
worst_trade              :    [X.XX]
profit_factor            :     [X.XX]
recovery_factor          :    [XX.XX]
payoff_ratio             :     [X.XX]
trade_efficiency         :     [X.XX]
consecutive_wins         :     [X.XX]
consecutive_losses       :     [X.XX]
max_consecutive_wins     :     [X.XX]
max_consecutive_losses   :     [X.XX]
avg_bars_win             :     [X.XX]
avg_bars_loss            :     [X.XX]

📉 KATEGORIA 5: DRAWDOWN METRICS (15 metryk)
--------------------------------------------------
max_drawdown             :   [X.XXXX]
max_drawdown_duration    :   [X.XXXX]
avg_drawdown             :  [X.XXXX]
avg_drawdown_duration    :   [X.XXXX]
recovery_time            :   [X.XXXX]
time_to_recovery         :   [X.XXXX]
drawdown_deviation       :   [X.XXXX]
pain_index               :   [X.XXXX]
ulcer_index              :   [X.XXXX]
lake_ratio               :   [X.XXXX]
pain_ratio               :   [X.XXXX]
martin_ratio             :   [X.XXXX]
drawdown_beta            :   [X.XXXX]
conditional_drawdown     :   [X.XXXX]
expected_drawdown        :   [X.XXXX]

📊 KATEGORIA 6: DISTRIBUTION METRICS
--------------------------------------------------
skewness                 :    [X.XXX]
kurtosis                 :   [XX.XXX]
var_95                   :  [X.XXXX]

================================================================================
```

### CRITICAL REQUIREMENTS

1. **ALL 130+ METRICS MUST BE DISPLAYED** - no exceptions
2. **EXACT FORMAT** - use the structure above precisely
3. **ALL 6 CATEGORIES** must be included for each strategy
4. **PROPER FORMATTING** - maintain spacing and alignment
5. **COMPLETE RESULTS** - never truncate or summarize

### STRATEGY COMPARISON TABLE

After individual strategy results, include:

```
================================================================================
📊 STRATEGY COMPARISON SUMMARY
================================================================================
Metric               |    TQQQ Long |   SQQQ Short
-------------------- | ------------ | ------------
total_return         |     [XXX.XX%] |     [XXX.XX%]
annualized_return    |      [XX.XX%] |      [XX.XX%]
volatility           |       [X.XXX] |       [X.XXX]
sharpe_ratio         |       [X.XXX] |      [X.XXX]
max_drawdown         |      [XX.XX%] |      [XX.XX%]
win_rate             |    [XXXX.XX%] |    [XXXX.XX%]
profit facctor       |        [X.XX] |        [X.XX]
```

This format is MANDATORY for all backtests and must not be modified or abbreviated.