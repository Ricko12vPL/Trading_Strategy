# Backtest Guide Phase 2 - Advanced Validation & Implementation
## Comprehensive Risk Assessment and Deployment Checklist

### CRITICAL VALIDATION REQUIREMENTS

This phase focuses on identifying and eliminating sources of false positives and ensuring robust strategy deployment. Even with low p-values and positive Monte Carlo results, multiple validation layers are mandatory before live deployment.

---

## 📊 SECTION 3: SOURCES OF FALSE POSITIVE RESULTS

### 3.1 DATA-SNOOPING AND OVERFITTING DETECTION

**MANDATORY CHECKS:**
```
Data Snooping Assessment Checklist:
☐ Record total number of parameter combinations tested
☐ Record total number of strategy variants evaluated
☐ Calculate actual Bonferroni correction factor
☐ Apply Benjamini-Hochberg false discovery rate correction
☐ Document selection process and reasoning
☐ Verify strategy performance on completely unseen data
☐ Test strategy on different asset classes/markets
☐ Validate strategy logic makes fundamental sense
```

**OVERFITTING INDICATORS TO MONITOR:**
- Strategy performs perfectly on training data but poorly OOS
- Large number of parameters relative to data points
- Complex rules that seem to "memorize" historical patterns
- Performance degrades significantly with minor parameter changes
- Strategy has unrealistic risk-adjusted returns (Sharpe > 3.0 consistently)

### 3.2 SURVIVORSHIP BIAS ELIMINATION

**SURVIVORSHIP BIAS CHECKLIST:**
```
☐ Include delisted/bankrupt securities in historical universe
☐ Account for index composition changes over time
☐ Include failed ETFs, closed funds, merged entities
☐ Adjust for corporate actions: splits, mergers, spin-offs
☐ Use point-in-time fundamental data (no forward-looking info)
☐ Account for IPO dates and delisting dates
☐ Verify benchmark indices use same survivorship-free data
☐ Test strategy on "survivor-free" universe vs current universe
```

### 3.3 LOOK-AHEAD BIAS AND FORWARD LEAK DETECTION

**LOOK-AHEAD BIAS PREVENTION:**
```
Signal Generation Validation:
☐ Verify all indicators use only historical data available at signal time
☐ Check earnings announcements, splits, dividends use announcement dates
☐ Confirm fundamental data uses reporting dates, not "as-of" dates
☐ Validate news/sentiment data timestamps are correct
☐ Ensure corporate actions are processed with proper delays
☐ Check option chains use historical implied volatilities
☐ Verify futures data uses proper continuation methods

Data Pipeline Validation:
☐ Implement strict timestamp controls in data processing
☐ Use point-in-time data reconstruction
☐ Test with simulated "real-time" data feed
☐ Validate data availability assumptions match reality
☐ Check for weekend/holiday data processing errors
```

### 3.4 EXECUTION ASSUMPTIONS REALITY CHECK

**UNREALISTIC EXECUTION ASSUMPTIONS:**
```
Execution Reality Checklist:
☐ Market Impact Assessment:
   - Small trades (<$10k): 0.05-0.15% impact
   - Medium trades ($10k-$100k): 0.15-0.50% impact  
   - Large trades (>$100k): 0.50-2.00% impact
   
☐ Slippage Modeling:
   - Liquid ETFs: 0.01-0.05% slippage
   - Individual stocks: 0.05-0.20% slippage
   - Options: 0.25-1.00% slippage
   - During volatility spikes: 2x-5x normal slippage
   
☐ Bid-Ask Spread Costs:
   - SPY/QQQ: $0.01 spread (~0.002%)
   - Individual large-cap: $0.01-0.05 spread
   - Small-cap stocks: 0.10-0.50% spread
   - Options: 0.05-0.50% spread depending on moneyness
```

### 3.5 TRANSACTION COST MODELING

**COMPREHENSIVE COST STRUCTURE:**
```
Transaction Costs Checklist:
☐ Commission Structure:
   - Stock trades: $0-$0.65 per trade
   - Options: $0.50-$1.00 per contract
   - Futures: $1-$5 per contract
   - International: 0.10-0.50% of trade value
   
☐ Financing Costs:
   - Margin interest: 5-12% annually on borrowed amounts
   - Hard-to-borrow fees: 0.25-20% annually for shorts
   - Overnight funding: SOFR + spread for leveraged positions
   
☐ Operational Costs:
   - Data feeds: $500-$5000/month
   - Platform fees: $100-$1000/month
   - Risk management systems: $1000-$10000/month
   - Infrastructure: $200-$2000/month
```

### 3.6 TEMPORAL DEPENDENCIES AND AUTOCORRELATION

**TIME SERIES STRUCTURE PRESERVATION:**
```
Temporal Validation Methods:
☐ Block Bootstrap Testing:
   - Use blocks of 20-60 consecutive observations
   - Preserve intraday, weekly, monthly patterns
   - Maintain volatility clustering
   - Keep correlation structure intact
   
☐ Model-Based Monte Carlo:
   - Fit ARIMA-GARCH models to returns
   - Generate synthetic data preserving statistical properties
   - Test strategy on model-generated scenarios
   - Validate against multiple volatility regimes
   
☐ Regime-Dependent Analysis:
   - Bull market performance (VIX < 20)
   - Bear market performance (market down >20%)
   - High volatility periods (VIX > 30)
   - Crisis periods (2008, 2020, specific events)
   - Different interest rate environments
```

### 3.7 REGIME DEPENDENCY ASSESSMENT

**MARKET REGIME ROBUSTNESS TESTING:**
```
Regime Analysis Framework:
☐ Bull Market Regimes (2009-2021):
   - Strategy performance during sustained uptrends
   - Correlation with market momentum
   - Performance during low volatility periods
   
☐ Bear Market Regimes (2000-2002, 2007-2009, 2020, 2022):
   - Drawdown protection effectiveness
   - Strategy behavior during market crashes
   - Recovery time after major losses
   
☐ Sideways Markets (2011-2016):
   - Performance in range-bound conditions
   - Whipsaw resistance
   - Transaction cost impact during chop
   
☐ Volatility Regimes:
   - Low VIX (<15): Strategy performance
   - Medium VIX (15-25): Normal operations
   - High VIX (25-40): Stress conditions
   - Extreme VIX (>40): Crisis management
   
☐ Interest Rate Environments:
   - Near-zero rates (2009-2015, 2020-2022)
   - Rising rates (2022-2024)
   - High rates (potential future scenarios)
```

---

## ✅ SECTION 4: PRE-DEPLOYMENT IMPLEMENTATION CHECKLIST

### 4.1 STATISTICAL CORRECTIONS AND MULTIPLE TESTING

**MANDATORY STATISTICAL ADJUSTMENTS:**
```
Multiple Testing Corrections:
☐ Bonferroni Correction Implementation:
   - α_corrected = α_original / number_of_tests
   - Document all tests performed (parameter sweeps, lookback periods, etc.)
   - Record which strategies survive Bonferroni correction
   - Example: If tested 100 parameter combinations, α = 0.05/100 = 0.0005
   
☐ Benjamini-Hochberg False Discovery Rate:
   - Rank all p-values from smallest to largest
   - Find largest k where P(k) ≤ (k/m) × α
   - Control false discovery rate at desired level (typically 0.05 or 0.10)
   - More powerful than Bonferroni for multiple strategies
   
☐ Documentation Requirements:
   - Log all tested parameters and their performance
   - Record selection criteria and thresholds used
   - Document why specific parameters were chosen
   - Maintain audit trail of all decisions
```

### 4.2 OUT-OF-SAMPLE AND WALK-FORWARD VALIDATION

**COMPREHENSIVE OOS TESTING FRAMEWORK:**
```
Out-of-Sample Validation:
☐ Temporal Split Validation:
   - Training: 70% of earliest data
   - Validation: 15% of middle data
   - Test: 15% of most recent data
   - NO overlap between periods
   - Minimum 2-3 years of OOS data
   
☐ Walk-Forward Analysis:
   - Optimization window: 2-3 years
   - Test window: 3-6 months
   - Step size: 1 month (monthly re-optimization)
   - Record performance degradation over time
   - Calculate stability metrics across walks
   
☐ Rolling Window Validation:
   - Fixed window size (e.g., 5 years)
   - Roll window forward monthly/quarterly
   - Test parameter stability over time
   - Identify regime-dependent performance
   
☐ Cross-Validation for Time Series:
   - Time series cross-validation (no random splits)
   - Expanding window validation
   - Blocked cross-validation preserving temporal order
```

### 4.3 ROBUSTNESS AND SENSITIVITY ANALYSIS

**PARAMETER SENSITIVITY TESTING:**
```
Robustness Testing Protocol:
☐ Parameter Sensitivity Analysis:
   - Test each parameter ±10%, ±20%, ±50%
   - Create parameter heat maps showing performance
   - Identify cliff effects and parameter instability
   - Document acceptable parameter ranges
   
☐ Multi-Dimensional Parameter Space:
   - Test parameter combinations, not just individual parameters
   - Use grid search or random search methods
   - Identify parameter interaction effects
   - Map stable regions in parameter space
   
☐ Alternative Signal Definitions:
   - Test different moving average types (EMA, SMA, WMA, HMA)
   - Try alternative momentum indicators
   - Test different lookback periods
   - Validate signal timing (entry delays, confirmation requirements)
   
☐ Market Condition Robustness:
   - Performance during different volatility regimes
   - Sector rotation impact
   - Correlation breakdown scenarios
   - Black swan event resilience
```

### 4.4 ADVANCED BOOTSTRAP AND MONTE CARLO METHODS

**SOPHISTICATED RESAMPLING TECHNIQUES:**

**Block size & frequency guidance (practical note):**
```
- For intraday data: consider block sizes equal to 1 trading day (all intraday bars within day) or 20-60 bars depending on bar resolution.
- For daily data: block size 20-60 days is reasonable to preserve monthly patterns; shorter blocks (5-20) may be used to preserve weekly structure.
- Choose block size based on:
  * Data resolution (tick/minute/daily),
  * Strategy holding period (intraday, multi-day, swing),
  * Observed autocorrelation length (use ACF/PACF to estimate).
- Always run sensitivity: test multiple block sizes (e.g., 20, 40, 60) and report stability of metrics.
```
**Why:** Block size critically affects bootstrap fidelity; give concrete guidance so users adapt by freq.
```
Advanced Bootstrap Methods:
☐ Stationary Block Bootstrap:
   - Block size: 20-60 observations (preserve monthly patterns)
   - Number of bootstrap samples: 10,000+
   - Calculate bootstrap confidence intervals
   - Test significance of performance metrics
   
☐ Circular Block Bootstrap:
   - Handles edge effects better than standard block bootstrap
   - Preserves seasonal patterns and trends
   - More accurate for long-term dependencies
   
☐ Model-Based Bootstrap:
   - Fit ARIMA-GARCH to residuals
   - Generate synthetic return series
   - Preserve volatility clustering and fat tails
   - Test strategy on model-generated scenarios
   
☐ Wild Bootstrap:
   - Robust to heteroskedasticity
   - Maintains original data points
   - Good for non-stationary time series
   - Less assumption-dependent
```

### 4.5 REALISTIC COST AND CAPACITY MODELING

**COMPREHENSIVE COST ANALYSIS:**
```
Real-World Cost Modeling:
☐ Conservative Cost Scenario (Recommended for Initial Sizing):
   - Slippage: 0.10-0.20% per trade
   - Market impact: 0.05-0.15% for moderate size
   - Bid-ask spread: Full spread cost
   - Commission: $1-5 per trade plus SEC fees
   - Financing: Prime + 2-4% on margin
   
☐ Realistic Cost Scenario (Normal Operations):
   - Slippage: 0.05-0.10% per trade
   - Market impact: 0.02-0.08% for appropriate sizing
   - Bid-ask spread: 50-75% of spread
   - Commission: Negotiated rates
   - Financing: Prime + 1-2% on margin
   
☐ Aggressive Cost Scenario (Best Case):
   - Slippage: 0.02-0.05% per trade
   - Market impact: 0.01-0.03% for small size
   - Bid-ask spread: 25-50% of spread
   - Commission: Volume discounts
   - Financing: Prime + 0.5-1% on margin
   
☐ Capacity Analysis:
   - Maximum position size without market impact
   - Daily volume constraints (max 10% of ADV)
   - Liquidity during stress periods
   - Scaling limitations as AUM grows
```

### 4.6 LIQUIDITY AND CAPACITY CONSTRAINTS

**POSITION SIZING AND LIQUIDITY ANALYSIS:**
```
Liquidity Assessment:
☐ Average Daily Volume Analysis:
   - TQQQ: ~$2-5B daily volume → Max position ~$50-200M
   - SQQQ: ~$500M-2B daily volume → Max position ~$10-50M
   - Individual stocks: Max 5-10% of 20-day ADV
   - Options: Much more restrictive, especially for illiquid strikes
   
☐ Market Impact Modeling:
   - Square root law: Impact ∝ √(trade_size/volume)
   - Temporary impact: 0.1 to 1.0% depending on urgency
   - Permanent impact: 0.01 to 0.10% for large trades
   - Non-linear scaling for very large positions
   
☐ Stress Liquidity Testing:
   - Liquidity during 2008, 2020 crisis periods
   - After-hours and pre-market liquidity
   - Earnings announcement periods
   - FOMC and major news event liquidity
   - Market closure risk (circuit breakers)
```

### 4.7 PAPER TRADING AND LIVE SIMULATION

**LIVE MARKET VALIDATION:**
```
Paper Trading Protocol:
☐ Live Market Simulation:
   - Use real-time market data feeds
   - Implement actual order routing
   - Test during different market hours
   - Include pre-market and after-hours sessions
   - Simulate partial fills and order rejections
   
☐ Order Management System Testing:
   - Test limit orders, market orders, stop orders
   - Validate order modification and cancellation
   - Test during high volatility periods
   - Verify risk management controls
   - Test position sizing calculations
   
☐ Data Feed Reliability:
   - Test with primary and backup data sources
   - Handle data feed outages and reconnections
   - Validate data quality and timeliness
   - Test with different data vendors
   - Monitor for data anomalies and bad ticks
   
☐ Performance Tracking:
   - Compare paper trading results to backtest
   - Track slippage and execution quality
   - Monitor latency and fill rates
   - Document any performance degradation
   - Run for minimum 3-6 months before live deployment

**Paper Trading Minimum Requirements (frequency-aware)**:
```
Paper Trading Minimums (choose whichever is met first):
☐ Intraday / High-frequency strategies (>=5 trades/day): min 1-2 months OR min 1000 filled trades
☐ Daily / Mid-frequency strategies (>=1 trade/day to <5/day): min 3 months OR min 300 filled trades
☐ Low-frequency strategies (<1 trade/day): min 3-6 months AND min 100-200 filled trades
☐ General rule: require BOTH a minimum time window and a minimum number of real filled trades to ensure statistical significance.
☐ If using options/futures with low trade rate, require longer observation (6+ months) or higher trade counts (200+ fills).
```
**Why:** Time-only thresholds can be misleading for low-frequency strategies — using both time and trade-count ensures results are meaningful.
```

### 4.8 REGIME AND STRESS TESTING

**COMPREHENSIVE STRESS SCENARIOS:**
```
Stress Testing Framework:
☐ Historical Stress Events:
   - Black Monday (1987): -22% single day
   - Tech Crash (2000-2002): -78% NASDAQ decline
   - Financial Crisis (2007-2009): -57% S&P 500 decline
   - Flash Crash (2010): Intraday volatility spike
   - COVID Crash (2020): -34% decline in 23 days
   - Rate Spike (2022): Bond/growth stock correlation
   
☐ Synthetic Stress Scenarios:
   - Gradual bear market: -2% monthly for 18 months
   - Sharp correction: -20% over 2-3 weeks with recovery
   - Volatility spike: VIX from 15 to 50+ overnight
   - Correlation breakdown: Traditional diversification fails
   - Liquidity crisis: Bid-ask spreads widen 5-10x
   - Interest rate shock: Fed funds +300-500bps quickly
   
☐ Strategy-Specific Stress Tests:
   - Leveraged ETF decay during sideways markets
   - Inverse correlation breakdown between TQQQ/SQQQ
   - Technology sector specific crashes
   - Momentum factor reversals
   - Mean reversion regime changes
```

### 4.9 RISK MANAGEMENT FRAMEWORK

**COMPREHENSIVE RISK CONTROLS:**

### Model revalidation and retraining policy
```
- Scheduled revalidation: run full validation (subset of tests: robustness, rollback, MC) quarterly (every 3 months).
- Triggered revalidation: trigger immediate revalidation if:
  - Realized P&L drops > 20% vs expected over rolling 60 trading days, OR
  - Realized Sharpe falls > 30% vs expected, OR
  - Drawdown exceeds historical worst-case by > X% (configurable), OR
  - Significant change in market structure (e.g., volatility regime shift).
- Record results and decision (retrain / parameter adjust / pause / retire).
```
**Why:** Ensures strategy remains valid over time and adapts to regime shifts.
```
Risk Management Implementation:
☐ Position-Level Risk Controls:
   - Maximum position size: 5-15% of portfolio per trade
   - Single-name concentration: Max 20% in any one security
   - Sector concentration: Max 40% in any one sector
   - Maximum leverage: 2:1 total portfolio leverage
   - Stop loss: Individual position stops at -10% to -20%
   
☐ Portfolio-Level Risk Controls:
   - Maximum drawdown limit: -15% to -25% (kill switch)
   - Daily loss limit: -3% to -5% of portfolio
   - Monthly loss limit: -8% to -12% of portfolio
   - Volatility targeting: Adjust position sizes for consistent risk
   - Correlation monitoring: Reduce positions if correlations spike
   
☐ Dynamic Risk Adjustments:
   - VIX-based position sizing (reduce size when VIX > 30)
   - Market regime detection and adjustment
   - Trailing stop losses for profitable positions
   - Profit taking rules: Take partial profits at +20%, +50%
   - Position scaling: Increase winners, reduce losers
   
☐ Emergency Protocols:
   - Market halt procedures
   - System failure backup plans  
   - Manual override capabilities
   - Communication protocols during stress
   - Liquidity preservation during crisis
```

### 4.10 DOCUMENTATION AND REPRODUCIBILITY

**COMPREHENSIVE RECORD KEEPING:**

### Reproducibility: containerization and seed management (required)
```
- Containerized execution: provide a Docker image or equivalent container defining execution environment.
- Requirements file: include exact package versions (requirements.txt or environment.yml).
- Random seed management: save all RNG seeds used in simulations and Monte Carlo runs to the experiment log to allow bit-for-bit reproducibility.
- Execution scripts: include scripts to reproduce each reported backtest (one-command run).
- Storage policy: define where snapshots (raw data, cleaned data, backtest outputs, logs) are stored (e.g., S3 bucket or GCS) and retention rules.
```
**Why:** Ensures auditors and future developers can reproduce results exactly.
```
Documentation Requirements:
☐ Strategy Documentation:
   - Mathematical formulation of all signals
   - Parameter values and their rationale
   - Risk management rules and thresholds
   - Expected performance characteristics
   - Known limitations and failure modes
   
☐ Version Control and Audit Trail:
   - Git repository with all code versions
   - Tagged releases for different strategy versions
   - Change log documenting all modifications
   - Backtest results for each version
   - Performance attribution for changes
   
☐ Data Lineage and Quality:
   - Data source documentation
   - Data cleaning and preprocessing steps
   - Quality control checks and anomaly detection
   - Data vendor changes and their impact
   - Backup data sources and validation
   
☐ Decision Log:
   - Record of all parameter choices and why
   - Performance review meetings and decisions
   - Risk management adjustments and triggers
   - Market regime changes and strategy responses
   - Lessons learned and future improvements
   
☐ Reproducibility Requirements:
   - Containerized execution environment
   - Requirements.txt with exact package versions
   - Random seed management for reproducible results
   - Automated testing and validation pipelines
   - Performance monitoring and alerting systems
```

---

## 🚨 MANDATORY SIGN-OFF CHECKLIST

**BEFORE LIVE DEPLOYMENT:**
```

Final Validation Checklist:
☐ All statistical corrections applied and documented
☐ Minimum 6 months successful paper trading completed (or frequency-aware minimums)
☐ Out-of-sample performance meets minimum thresholds
☐ Robustness testing shows stable performance across parameter ranges
☐ Stress testing demonstrates acceptable worst-case scenarios
☐ Risk management systems tested and functional
☐ All documentation complete and reviewed
☐ Emergency procedures defined and tested
☐ Capital allocation approved by risk committee
☐ Monitoring and reporting systems operational

Strategy Approved for Live Trading: ☐ YES ☐ NO

Date: ___________
Risk Manager Signature: ___________________
Strategy Developer Signature: ___________________
Operations Lead Signature: ___________________
Compliance Officer Signature: ___________________

**Roles & Owners:**  
- Risk Manager (owner of risk sign-off)  
- Strategy Developer (owner of model logic and backtests)  
- Operations Lead (owner of execution, infra, reconciliation)  
- Compliance Officer (owner of regulatory & instrument checks)  
- Note: All sign-offs must be recorded with date and "last regulatory check date" (YYYY-MM-DD).
```

---

---

## 🚨 CRITICAL REGULATORY COMPLIANCE WARNING

### INSTRUMENT ACCESSIBILITY AND REGULATORY RESTRICTIONS

**IMPORTANT:** Many financial instruments may have accessibility restrictions based on investor domicile and regulatory requirements:

```
🔴 COMMON REGULATORY RESTRICTIONS BY REGION:

EU USERS - MiFID II / ESMA Restrictions:
☐ US-domiciled leveraged ETFs (TQQQ, SQQQ, UPRO, SPXU, etc.) - NO KID documents
☐ Complex derivatives without UCITS classification
☐ Non-EU domiciled structured products
☐ Certain CFDs and spread betting instruments
☐ High-risk cryptocurrency derivatives

UK USERS - FCA Restrictions:
☐ US leveraged ETFs post-Brexit (temporary permissions expired)
☐ Certain cryptocurrency derivatives for retail clients  
☐ Complex structured products without appropriate warnings
☐ Non-UK/EU domiciled funds without regulatory recognition

US USERS - SEC/FINRA Restrictions:
☐ EU UCITS ETFs (reverse restriction)
☐ Certain foreign-domiciled derivatives
☐ Unregistered offshore investment funds
☐ Some cryptocurrency products for retail investors

ASIA-PACIFIC - Local Restrictions:
☐ Jurisdiction-specific product availability
☐ Currency-hedged vs unhedged versions
☐ Local regulatory approval requirements
☐ Tax-efficient vs standard product variants
```

**UNIVERSAL INSTRUMENT VALIDATION PROTOCOL (values illustrative — adapt per market):**

```
PRE-BACKTEST INSTRUMENT VERIFICATION:
☐ Verify instrument availability in target market
☐ Confirm regulatory documentation (KID/KIID/Prospectus)
☐ Check investor qualification requirements
☐ Validate broker/platform access
☐ Confirm trading hours and settlement procedures
☐ Verify currency exposure and hedging options
☐ Check expense ratios and fee structures
☐ Validate liquidity and bid-ask spreads
☐ Confirm tax treatment in investor domicile
☐ Review counterparty and issuer credit risk
```

**INSTRUMENT-AGNOSTIC ALTERNATIVE FRAMEWORK:**

```
SYSTEMATIC ALTERNATIVE SELECTION:
☐ Map restricted instruments to compliant alternatives
☐ Maintain similar risk/return characteristics
☐ Preserve correlation structures where possible
☐ Account for different expense ratios and tracking error
☐ Adjust for different underlying indices or baskets
☐ Consider synthetic vs physical replication methods
☐ Evaluate currency hedging implications
☐ Assess liquidity differences and market impact
☐ Review different market makers and authorized participants
☐ Account for different trading venues and hours
```

**MANDATORY COMPLIANCE CHECKLIST FOR ANY INSTRUMENT:**

_Last regulatory check date: YYYY-MM-DD — update this field before any deployment._

```
UNIVERSAL PRE-DEPLOYMENT VERIFICATION:
☐ Legal accessibility verification for target investor base
☐ Regulatory documentation completeness check
☐ Appropriateness and suitability assessment
☐ Risk disclosure and warning requirements
☐ Tax efficiency analysis for target domiciles
☐ Liquidity and execution quality assessment
☐ Counterparty risk evaluation
☐ Operational risk assessment (settlement, custody)
☐ Regulatory capital treatment verification
☐ Ongoing compliance monitoring requirements
```

**STRATEGY ADAPTATION FRAMEWORK:**

```
TECHNICAL MODIFICATIONS FOR ALTERNATIVE INSTRUMENTS:
☐ Recalibrate parameters for different volatility profiles
☐ Adjust correlation analysis for new instrument relationships
☐ Account for different expense ratios and tracking errors
☐ Modify position sizing for different leverage characteristics
☐ Adapt to different trading hours and market sessions
☐ Update liquidity assumptions and market impact models
☐ Revise currency hedging and exposure calculations
☐ Adjust for different dividend/distribution treatments
☐ Account for synthetic vs physical replication differences
☐ Modify risk management parameters for new instruments
☐ Update benchmark comparisons and performance attribution
☐ Recalibrate entry/exit signals for different price dynamics
```

---

**This checklist is MANDATORY and must be completed in full before any live capital deployment. No exceptions.**