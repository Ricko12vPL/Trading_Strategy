# Przewodnik Algorytmicznego Handlu Opcjami
## Kompletny Podręcznik Strategii i Praktyk Top Firm Kwantowych

### Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Analiza Top Firm Kwantowych](#analiza-top-firm-kwantowych)
3. [Fundamentalne Strategie Opcyjne](#fundamentalne-strategie-opcyjne)
4. [Zaawansowane Strategie Algorytmiczne](#zaawansowane-strategie-algorytmiczne)
5. [Modele Matematyczne i Wycena](#modele-matematyczne-i-wycena)
6. [Machine Learning w Opcjach](#machine-learning-w-opcjach)
7. [Infrastruktura i Technologia](#infrastruktura-i-technologia)
8. [Risk Management Framework](#risk-management-framework)
9. [Execution i Market Microstructure](#execution-i-market-microstructure)
10. [Backtesting i Walidacja](#backtesting-i-walidacja)
11. [Regulatory i Compliance](#regulatory-i-compliance)
12. [Case Studies](#case-studies)
13. [Praktyczna Implementacja](#praktyczna-implementacja)
14. [Portfolio Management](#portfolio-management)
15. [Podsumowanie i Roadmapa](#podsumowanie-i-roadmapa)

---

## Wprowadzenie

### Dlaczego Options Trading?

Opcje oferują unikalną asymetrię risk-reward niedostępną w innych instrumentach. Top firmy kwantowe generują konsystentne zyski poprzez:

- **Leverage**: Kontrola dużych pozycji małym kapitałem
- **Defined Risk**: Znany maksymalny loss przy kupnie opcji
- **Volatility Trading**: Możliwość tradingu samej zmienności
- **Non-linear Payoffs**: Zyski nieproporcjonalne do ruchu bazowego
- **Time Decay**: Zarabianie na upływie czasu (theta)

### Liczby, które mówią same za siebie

| Firma | Roczny Zwrot | AUM | Specjalizacja |
|-------|--------------|-----|---------------|
| Renaissance (Medallion) | 66% (przed opłatami) | $130B+ | Statistical Arbitrage |
| Jane Street | Nieujawniony (est. 30-50%) | $17B+ | Options Market Making |
| Citadel | 24% | $63B | Multi-Strategy |
| Optiver | Nieujawniony | $17B | Derivatives MM |
| SIG | 20-30% | Nieujawniony | Options Trading |
| D.E. Shaw | 10-20% | $60B+ | Quantitative |
| Two Sigma | 15-20% | $60B+ | AI/ML Trading |
| Millennium | 14% | $75B+ | Multi-Strategy |

---

## Analiza Top Firm Kwantowych

### Renaissance Technologies - Szczegółowa Analiza

#### Historia i Filozofia
Renaissance, założone przez Jima Simonsa (byłego kryptografa NSA), rewolucjonizowało trading poprzez zastosowanie czystej matematyki. Kluczowe innowacje:

**Medallion Fund - Sekrety Sukcesu:**
1. **Petabajtowe bazy danych**
   - Dane pogodowe, satelitarne, social media
   - Shipping data, parking lots occupancy
   - Credit card transactions patterns
   - Alternative data przed erą "big data"

2. **4 Million Alphas**
   ```python
   # Przykład struktury alpha w stylu RenTech
   class Alpha:
       def __init__(self, signal_id, universe, lookback):
           self.signal_id = signal_id
           self.universe = universe
           self.lookback = lookback
           self.weights = {}
           
       def calculate_signal(self, data):
           # Proprietary signal calculation
           signal = self.feature_engineering(data)
           signal = self.apply_filters(signal)
           signal = self.normalize(signal)
           return signal
           
       def feature_engineering(self, data):
           features = []
           # Price-based features
           features.append(data['returns'].rolling(self.lookback).mean())
           features.append(data['volume'].rolling(self.lookback).std())
           # Microstructure features
           features.append(data['bid_ask_spread'].ewm(span=self.lookback).mean())
           # Cross-sectional features
           features.append(data['returns'].rank(pct=True))
           return pd.concat(features, axis=1)
   ```

3. **Hidden Markov Models dla Opcji**
   ```python
   from hmmlearn import hmm
   import numpy as np
   
   class OptionHMM:
       def __init__(self, n_states=3):
           self.model = hmm.GaussianHMM(n_components=n_states, 
                                        covariance_type="full")
           self.states = ['Low_Vol', 'Normal', 'High_Vol']
           
       def train(self, option_data):
           # Features: IV, skew, term structure, volume
           features = self.extract_features(option_data)
           self.model.fit(features)
           
       def predict_regime(self, current_data):
           features = self.extract_features(current_data)
           state = self.model.predict(features)
           return self.states[state[-1]]
           
       def extract_features(self, data):
           return np.column_stack([
               data['implied_volatility'],
               data['put_call_skew'],
               data['term_structure_slope'],
               np.log(data['volume'])
           ])
   ```

#### RenTech Options Strategies

**1. Statistical Arbitrage in Options**
```python
class StatArbOptions:
    def __init__(self, lookback=20, z_threshold=2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        
    def identify_pairs(self, options_universe):
        """Identify cointegrated option pairs"""
        from statsmodels.tsa.stattools import coint
        
        pairs = []
        for opt1 in options_universe:
            for opt2 in options_universe:
                if opt1 != opt2:
                    _, p_value, _ = coint(opt1['price'], opt2['price'])
                    if p_value < 0.01:  # 99% confidence
                        pairs.append((opt1, opt2))
        return pairs
        
    def calculate_spread(self, opt1, opt2):
        """Calculate normalized spread"""
        # Hedge ratio via OLS
        X = opt1['price'].values.reshape(-1, 1)
        y = opt2['price'].values
        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
        
        spread = opt2['price'] - hedge_ratio * opt1['price']
        z_score = (spread - spread.rolling(self.lookback).mean()) / \
                  spread.rolling(self.lookback).std()
        return z_score
        
    def generate_signals(self, z_score):
        """Generate trading signals"""
        signals = pd.Series(index=z_score.index, data=0)
        signals[z_score > self.z_threshold] = -1  # Short spread
        signals[z_score < -self.z_threshold] = 1  # Long spread
        signals[abs(z_score) < 0.5] = 0  # Close position
        return signals
```

### Jane Street - Deep Dive

#### Core Competencies
Jane Street dominuje w market making poprzez:

**1. Ultra-Fast Execution w OCaml**
```ocaml
(* Jane Street style OCaml options pricer *)
module OptionPricer = struct
  type option_type = Call | Put
  
  type option_params = {
    strike: float;
    expiry: float;
    opt_type: option_type;
    underlying: float;
    rate: float;
    volatility: float;
  }
  
  let normal_cdf x =
    let a1 =  0.254829592 in
    let a2 = -0.284496736 in
    let a3 =  1.421413741 in
    let a4 = -1.453152027 in
    let a5 =  1.061405429 in
    let p  =  0.3275911 in
    
    let sign = if x < 0. then -1. else 1. in
    let x = abs_float x /. sqrt 2.0 in
    
    let t = 1.0 /. (1.0 +. p *. x) in
    let y = 1.0 -. (((((a5 *. t +. a4) *. t) +. a3) *. t +. a2) *. t +. a1) *. t *. exp(-. x *. x) in
    
    0.5 *. (1.0 +. sign *. y)
  
  let black_scholes params =
    let d1 = (log(params.underlying /. params.strike) +. 
              (params.rate +. params.volatility ** 2. /. 2.) *. params.expiry) /.
              (params.volatility *. sqrt params.expiry) in
    let d2 = d1 -. params.volatility *. sqrt params.expiry in
    
    match params.opt_type with
    | Call -> 
        params.underlying *. normal_cdf d1 -. 
        params.strike *. exp(-. params.rate *. params.expiry) *. normal_cdf d2
    | Put ->
        params.strike *. exp(-. params.rate *. params.expiry) *. normal_cdf (-. d2) -.
        params.underlying *. normal_cdf (-. d1)
        
  let calculate_greeks params =
    let epsilon = 0.01 in
    let price = black_scholes params in
    
    (* Delta *)
    let params_up = {params with underlying = params.underlying +. epsilon} in
    let price_up = black_scholes params_up in
    let delta = (price_up -. price) /. epsilon in
    
    (* Gamma *)
    let params_down = {params with underlying = params.underlying -. epsilon} in
    let price_down = black_scholes params_down in
    let gamma = (price_up -. 2. *. price +. price_down) /. (epsilon ** 2.) in
    
    (* Vega *)
    let params_vol_up = {params with volatility = params.volatility +. epsilon} in
    let price_vol_up = black_scholes params_vol_up in
    let vega = (price_vol_up -. price) /. epsilon in
    
    (* Theta *)
    let params_time = {params with expiry = params.expiry -. (1. /. 365.)} in
    let price_time = black_scholes params_time in
    let theta = price_time -. price in
    
    {price; delta; gamma; vega; theta}
end
```

**2. Market Making Algorithm**
```python
class JaneStreetMarketMaker:
    def __init__(self, symbol, min_spread=0.01, inventory_limit=10000):
        self.symbol = symbol
        self.min_spread = min_spread
        self.inventory_limit = inventory_limit
        self.inventory = 0
        self.pnl = 0
        
    def calculate_fair_value(self, orderbook, trades, options_chain):
        """Multi-factor fair value calculation"""
        
        # 1. Orderbook imbalance
        bid_volume = sum([level[1] for level in orderbook['bids'][:5]])
        ask_volume = sum([level[1] for level in orderbook['asks'][:5]])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # 2. Recent trades momentum
        recent_trades = trades[-100:]
        vwap = sum([t['price'] * t['size'] for t in recent_trades]) / \
               sum([t['size'] for t in recent_trades])
        
        # 3. Options flow
        call_volume = sum([opt['volume'] for opt in options_chain 
                          if opt['type'] == 'call'])
        put_volume = sum([opt['volume'] for opt in options_chain 
                         if opt['type'] == 'put'])
        options_sentiment = (call_volume - put_volume) / (call_volume + put_volume)
        
        # 4. Weighted fair value
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        fair_value = mid_price * 0.4 + vwap * 0.3 + \
                     mid_price * (1 + imbalance * 0.001) * 0.2 + \
                     mid_price * (1 + options_sentiment * 0.0005) * 0.1
                     
        return fair_value
        
    def calculate_spread(self, fair_value, volatility, inventory):
        """Dynamic spread based on inventory and volatility"""
        
        # Base spread
        base_spread = max(self.min_spread, volatility * 0.01)
        
        # Inventory skew
        inventory_ratio = inventory / self.inventory_limit
        inventory_skew = abs(inventory_ratio) * base_spread * 0.5
        
        if inventory > 0:  # Long inventory, skew prices down
            bid_spread = base_spread + inventory_skew
            ask_spread = base_spread - inventory_skew * 0.5
        else:  # Short inventory, skew prices up
            bid_spread = base_spread - inventory_skew * 0.5
            ask_spread = base_spread + inventory_skew
            
        return bid_spread, ask_spread
        
    def generate_quotes(self, fair_value, volatility):
        """Generate bid/ask quotes"""
        bid_spread, ask_spread = self.calculate_spread(
            fair_value, volatility, self.inventory
        )
        
        bid_price = fair_value - bid_spread / 2
        ask_price = fair_value + ask_spread / 2
        
        # Size based on inventory
        base_size = 100
        inventory_factor = 1 - abs(self.inventory / self.inventory_limit)
        
        bid_size = int(base_size * (1 + inventory_factor) if self.inventory <= 0 else 
                      base_size * inventory_factor)
        ask_size = int(base_size * (1 + inventory_factor) if self.inventory >= 0 else 
                      base_size * inventory_factor)
        
        return {
            'bid': {'price': bid_price, 'size': bid_size},
            'ask': {'price': ask_price, 'size': ask_size}
        }
```

### Citadel - Global Quantitative Strategies

#### Multi-Strategy Approach

**1. Cross-Asset Options Arbitrage**
```python
class CitadelCrossAssetArbitrage:
    def __init__(self):
        self.assets = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT']
        self.correlation_window = 60
        self.signal_threshold = 2.0
        
    def calculate_correlation_matrix(self, returns_df):
        """Calculate rolling correlation matrix"""
        corr_matrix = returns_df.rolling(self.correlation_window).corr()
        return corr_matrix
        
    def identify_dislocation(self, option_ivs, historical_corr):
        """Identify correlation dislocations in option markets"""
        
        signals = {}
        
        for asset1 in self.assets:
            for asset2 in self.assets:
                if asset1 != asset2:
                    # Historical correlation
                    hist_corr = historical_corr[asset1][asset2]
                    
                    # Implied correlation from options
                    impl_corr = self.calculate_implied_correlation(
                        option_ivs[asset1], 
                        option_ivs[asset2]
                    )
                    
                    # Z-score of difference
                    diff = impl_corr - hist_corr
                    z_score = diff / historical_corr[asset1][asset2].std()
                    
                    if abs(z_score) > self.signal_threshold:
                        signals[f"{asset1}_{asset2}"] = {
                            'z_score': z_score,
                            'hist_corr': hist_corr,
                            'impl_corr': impl_corr,
                            'trade': self.generate_trade(asset1, asset2, z_score)
                        }
                        
        return signals
        
    def calculate_implied_correlation(self, iv1, iv2):
        """Extract implied correlation from option IVs"""
        # Simplified version - real implementation would use
        # quanto options or dispersion trading formulas
        
        # ATM implied vols
        atm_iv1 = iv1[iv1['moneyness'].between(0.95, 1.05)]['iv'].mean()
        atm_iv2 = iv2[iv2['moneyness'].between(0.95, 1.05)]['iv'].mean()
        
        # Use index options to back out correlation
        # This is simplified - actual method more complex
        basket_variance = (atm_iv1**2 + atm_iv2**2) / 2
        implied_corr = 1 - (basket_variance / (atm_iv1 * atm_iv2))
        
        return np.clip(implied_corr, -1, 1)
        
    def generate_trade(self, asset1, asset2, z_score):
        """Generate dispersion trade"""
        
        trade = {
            'type': 'dispersion',
            'legs': []
        }
        
        if z_score > self.signal_threshold:
            # Implied correlation too high - sell correlation
            trade['legs'].append({
                'asset': 'SPX',  # Sell index vol
                'action': 'sell',
                'instrument': 'straddle',
                'size': 10
            })
            trade['legs'].append({
                'asset': asset1,  # Buy component vol
                'action': 'buy',
                'instrument': 'straddle',
                'size': 5
            })
            trade['legs'].append({
                'asset': asset2,
                'action': 'buy',
                'instrument': 'straddle',
                'size': 5
            })
        else:
            # Implied correlation too low - buy correlation
            trade['legs'] = [{'reverse': True}]  # Opposite of above
            
        return trade
```

**2. Event-Driven Options Strategy**
```python
class EventDrivenOptions:
    def __init__(self):
        self.events = ['earnings', 'fomc', 'economic_data', 'expiration']
        
    def pre_earnings_strategy(self, symbol, earnings_date):
        """Pre-earnings volatility trading"""
        
        options_chain = self.get_options_chain(symbol, earnings_date)
        
        # Calculate term structure
        term_structure = []
        for expiry in options_chain['expirations']:
            days_to_expiry = (expiry - datetime.now()).days
            atm_iv = self.calculate_atm_iv(options_chain, expiry)
            term_structure.append({
                'days': days_to_expiry,
                'iv': atm_iv,
                'crosses_event': expiry > earnings_date
            })
            
        # Identify vol kink
        pre_event_iv = np.mean([ts['iv'] for ts in term_structure 
                                if not ts['crosses_event']])
        post_event_iv = np.mean([ts['iv'] for ts in term_structure 
                                 if ts['crosses_event']])
        
        event_vol = np.sqrt((post_event_iv**2 * 30 - pre_event_iv**2 * 27) / 3)
        
        # Historical earnings moves
        historical_moves = self.get_historical_earnings_moves(symbol)
        expected_move = np.mean(np.abs(historical_moves))
        
        # Trade decision
        if event_vol > expected_move * 1.5:
            return self.sell_earnings_vol(symbol, earnings_date)
        elif event_vol < expected_move * 0.7:
            return self.buy_earnings_vol(symbol, earnings_date)
        else:
            return None
            
    def sell_earnings_vol(self, symbol, earnings_date):
        """Sell overpriced earnings volatility"""
        
        # Iron condor around expected move
        current_price = self.get_current_price(symbol)
        expected_move_pct = self.get_expected_move(symbol, earnings_date)
        
        trade = {
            'strategy': 'iron_condor',
            'expiration': self.get_expiry_after_earnings(earnings_date),
            'legs': [
                {'type': 'put', 'action': 'sell', 
                 'strike': current_price * (1 - expected_move_pct * 1.5)},
                {'type': 'put', 'action': 'buy', 
                 'strike': current_price * (1 - expected_move_pct * 2)},
                {'type': 'call', 'action': 'sell', 
                 'strike': current_price * (1 + expected_move_pct * 1.5)},
                {'type': 'call', 'action': 'buy', 
                 'strike': current_price * (1 + expected_move_pct * 2)}
            ]
        }
        
        return trade
```

### Optiver - Advanced Market Making

#### Asymmetric Trading Strategies

**1. Volatility Surface Arbitrage**
```python
class OptiverVolSurfaceArbitrage:
    def __init__(self):
        self.surface_model = 'SABR'  # or SVI, Heston
        
    def fit_volatility_surface(self, options_data):
        """Fit SABR model to implied volatility surface"""
        
        from scipy.optimize import minimize
        
        def sabr_vol(K, F, T, alpha, beta, rho, nu):
            """SABR implied volatility formula"""
            if K == F:
                return alpha * (1 + (2-3*rho**2)*nu**2*T/24) / F**(1-beta)
                
            z = nu/alpha * (F*K)**((1-beta)/2) * np.log(F/K)
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            
            A = alpha / ((F*K)**((1-beta)/2) * 
                        (1 + (1-beta)**2/24 * np.log(F/K)**2 + 
                         (1-beta)**4/1920 * np.log(F/K)**4))
            
            B = 1 + ((1-beta)**2/24 * alpha**2/(F*K)**(1-beta) + 
                    1/4 * rho*beta*nu*alpha/(F*K)**((1-beta)/2) + 
                    (2-3*rho**2)/24 * nu**2) * T
            
            return A * z/x_z * B
            
        def objective(params):
            """Minimize squared errors"""
            alpha, beta, rho, nu = params
            errors = []
            
            for _, row in options_data.iterrows():
                model_iv = sabr_vol(row['strike'], row['forward'], 
                                   row['time_to_expiry'], 
                                   alpha, beta, rho, nu)
                errors.append((model_iv - row['implied_vol'])**2 * row['vega'])
                
            return np.sum(errors)
            
        # Initial guess and bounds
        x0 = [0.2, 0.5, 0.0, 0.3]
        bounds = [(0.001, 1), (0, 1), (-0.999, 0.999), (0.001, 1)]
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return result.x
        
    def identify_arbitrage_opportunities(self, fitted_params, market_data):
        """Find mispriced options on the surface"""
        
        alpha, beta, rho, nu = fitted_params
        opportunities = []
        
        for _, opt in market_data.iterrows():
            model_price = self.sabr_vol(opt['strike'], opt['forward'],
                                        opt['time_to_expiry'],
                                        alpha, beta, rho, nu)
            
            market_price = opt['implied_vol']
            
            # Price in terms of vega-adjusted dollars
            mispricing = (model_price - market_price) * opt['vega']
            
            if abs(mispricing) > 0.50:  # $0.50 edge after transaction costs
                opportunities.append({
                    'option': opt,
                    'model_iv': model_price,
                    'market_iv': market_price,
                    'edge': mispricing,
                    'trade': 'buy' if mispricing > 0 else 'sell'
                })
                
        return sorted(opportunities, key=lambda x: abs(x['edge']), reverse=True)
```

**2. Smart Execution Algorithm**
```python
class OptiverSmartExecution:
    def __init__(self):
        self.venues = ['CBOE', 'ISE', 'PHLX', 'BOX', 'NASDAQ']
        self.latency = {'CBOE': 0.1, 'ISE': 0.15, 'PHLX': 0.12, 
                       'BOX': 0.18, 'NASDAQ': 0.11}  # milliseconds
        
    def smart_order_routing(self, order, market_data):
        """Optimal order routing across venues"""
        
        routing_plan = []
        remaining_size = order['size']
        
        while remaining_size > 0:
            # Get current quotes from all venues
            quotes = self.get_venue_quotes(order['symbol'], market_data)
            
            # Calculate effective price including fees
            for venue in quotes:
                if order['side'] == 'buy':
                    quotes[venue]['effective_price'] = \
                        quotes[venue]['ask'] + self.get_venue_fees(venue, 'take')
                else:
                    quotes[venue]['effective_price'] = \
                        quotes[venue]['bid'] - self.get_venue_fees(venue, 'take')
                        
            # Sort by best effective price
            sorted_venues = sorted(quotes.items(), 
                                 key=lambda x: x[1]['effective_price'],
                                 reverse=(order['side']=='sell'))
            
            # Route to best venue
            best_venue = sorted_venues[0][0]
            available_size = sorted_venues[0][1]['size']
            
            route_size = min(remaining_size, available_size)
            
            routing_plan.append({
                'venue': best_venue,
                'size': route_size,
                'price': sorted_venues[0][1]['effective_price'],
                'latency': self.latency[best_venue]
            })
            
            remaining_size -= route_size
            
            # Update market data (remove filled liquidity)
            market_data[best_venue]['size'] -= route_size
            
        return self.optimize_routing_sequence(routing_plan)
        
    def optimize_routing_sequence(self, routing_plan):
        """Optimize order of execution based on latency and price"""
        
        # Group by similar prices to execute in parallel
        price_groups = {}
        for route in routing_plan:
            price_key = round(route['price'], 2)
            if price_key not in price_groups:
                price_groups[price_key] = []
            price_groups[price_key].append(route)
            
        # Execute best price group first, then parallel within group
        optimized_plan = []
        for price in sorted(price_groups.keys()):
            group = price_groups[price]
            # Sort by latency within price group
            group.sort(key=lambda x: x['latency'])
            optimized_plan.extend(group)
            
        return optimized_plan
```

### Susquehanna (SIG) - Game Theory Approach

#### Poker-Inspired Trading

**1. Game Theory Options Pricing**
```python
class SIGGameTheoryPricing:
    def __init__(self):
        self.nash_equilibrium_cache = {}
        
    def calculate_nash_equilibrium(self, market_makers, option):
        """Find Nash equilibrium for option pricing among MMs"""
        
        import nashpy as nash
        
        # Define payoff matrices
        n_strategies = 5  # Different spread levels
        spreads = np.linspace(0.01, 0.10, n_strategies)
        
        # Payoff depends on spread and probability of execution
        def calculate_payoff(my_spread, opponent_spread, volatility):
            if my_spread < opponent_spread:
                # I win the trade
                execution_prob = 0.8
                profit = my_spread - volatility * 0.01  # Adjusted for risk
            elif my_spread == opponent_spread:
                # Split the flow
                execution_prob = 0.4
                profit = my_spread - volatility * 0.01
            else:
                # Lose the trade
                execution_prob = 0.1
                profit = 0
                
            return execution_prob * profit
            
        # Build payoff matrices
        payoff_matrix_1 = np.zeros((n_strategies, n_strategies))
        payoff_matrix_2 = np.zeros((n_strategies, n_strategies))
        
        current_volatility = option['implied_volatility']
        
        for i, spread1 in enumerate(spreads):
            for j, spread2 in enumerate(spreads):
                payoff_matrix_1[i, j] = calculate_payoff(spread1, spread2, 
                                                         current_volatility)
                payoff_matrix_2[i, j] = calculate_payoff(spread2, spread1, 
                                                         current_volatility)
                
        # Find Nash equilibrium
        game = nash.Game(payoff_matrix_1, payoff_matrix_2)
        equilibria = list(game.support_enumeration())
        
        if equilibria:
            # Use first Nash equilibrium
            eq = equilibria[0]
            optimal_spread_idx = np.argmax(eq[0])
            return spreads[optimal_spread_idx]
        else:
            return spreads[n_strategies // 2]  # Default to middle spread
            
    def multi_player_game(self, players, option_chain):
        """Multi-player game for complex option strategies"""
        
        class Player:
            def __init__(self, name, capital, risk_tolerance):
                self.name = name
                self.capital = capital
                self.risk_tolerance = risk_tolerance
                self.strategy = None
                
            def choose_strategy(self, market_state, other_players):
                # Each player chooses based on their utility function
                strategies = ['vol_sell', 'vol_buy', 'delta_hedge', 'spread']
                
                expected_utilities = {}
                for strat in strategies:
                    utility = self.calculate_utility(strat, market_state, 
                                                    other_players)
                    expected_utilities[strat] = utility
                    
                self.strategy = max(expected_utilities, 
                                  key=expected_utilities.get)
                return self.strategy
                
            def calculate_utility(self, strategy, market_state, others):
                base_return = self.expected_return(strategy, market_state)
                risk = self.expected_risk(strategy, market_state)
                competition_factor = self.competition_adjustment(strategy, others)
                
                utility = base_return - self.risk_tolerance * risk
                utility *= competition_factor
                
                return utility
                
        # Simulate multi-round game
        rounds = 10
        for round in range(rounds):
            strategies = {}
            for player in players:
                other_players = [p for p in players if p != player]
                strategy = player.choose_strategy(option_chain, other_players)
                strategies[player.name] = strategy
                
            # Update market based on collective actions
            option_chain = self.update_market(option_chain, strategies)
            
        return strategies, option_chain
```

**2. Bluff Detection in Options Flow**
```python
class OptionsFlowBluffDetection:
    def __init__(self):
        self.historical_patterns = {}
        self.bluff_indicators = []
        
    def detect_spoofing(self, order_book_history, time_window=1000):
        """Detect spoofing patterns in options order book"""
        
        potential_spoofs = []
        
        for i in range(len(order_book_history) - time_window):
            window = order_book_history[i:i+time_window]
            
            # Pattern 1: Large orders that disappear quickly
            large_orders = self.find_large_orders(window)
            
            for order in large_orders:
                lifetime = self.calculate_order_lifetime(order, window)
                
                if lifetime < 100 and order['size'] > self.get_avg_size() * 10:
                    # Large order with very short lifetime
                    if not self.was_executed(order, window):
                        potential_spoofs.append({
                            'timestamp': order['timestamp'],
                            'order': order,
                            'pattern': 'quick_cancel',
                            'confidence': 0.8
                        })
                        
            # Pattern 2: Layering detection
            layering = self.detect_layering(window)
            if layering:
                potential_spoofs.append(layering)
                
            # Pattern 3: Momentum ignition
            ignition = self.detect_momentum_ignition(window)
            if ignition:
                potential_spoofs.append(ignition)
                
        return potential_spoofs
        
    def detect_layering(self, window):
        """Detect layering pattern"""
        
        # Look for multiple orders at increasing price levels
        # that get cancelled when price approaches
        
        orders_by_participant = {}
        
        for event in window:
            if event['type'] == 'new_order':
                participant = event.get('participant_id', 'unknown')
                if participant not in orders_by_participant:
                    orders_by_participant[participant] = []
                orders_by_participant[participant].append(event)
                
        for participant, orders in orders_by_participant.items():
            if len(orders) >= 5:  # Multiple orders
                # Check if orders are at increasing price levels
                prices = [o['price'] for o in orders]
                if all(prices[i] <= prices[i+1] for i in range(len(prices)-1)):
                    # Check cancellation pattern
                    cancel_rate = self.get_cancellation_rate(participant, window)
                    if cancel_rate > 0.9:
                        return {
                            'pattern': 'layering',
                            'participant': participant,
                            'confidence': 0.7,
                            'orders': orders
                        }
                        
        return None
```

### D.E. Shaw & Two Sigma - Technology-First Approach

#### Machine Learning Infrastructure

**1. D.E. Shaw's Multi-Strategy Framework**
```python
class DEShawMultiStrategy:
    def __init__(self):
        self.strategies = {
            'stat_arb': StatisticalArbitrage(),
            'vol_arb': VolatilityArbitrage(),
            'event': EventDriven(),
            'macro': MacroOptions(),
            'ml': MachineLearningStrategy()
        }
        self.capital_allocation = {}
        
    def optimize_capital_allocation(self, market_regime):
        """Dynamic capital allocation using Markowitz optimization"""
        
        from scipy.optimize import minimize
        
        # Get expected returns and covariance for each strategy
        returns = []
        strategies_list = []
        
        for name, strategy in self.strategies.items():
            hist_returns = strategy.get_historical_returns()
            returns.append(hist_returns)
            strategies_list.append(name)
            
        returns_df = pd.DataFrame(returns).T
        returns_df.columns = strategies_list
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Add regime-specific adjustments
        regime_multipliers = self.get_regime_multipliers(market_regime)
        for strat in strategies_list:
            expected_returns[strat] *= regime_multipliers.get(strat, 1.0)
            
        # Optimization objective
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_vol
            return -sharpe  # Minimize negative Sharpe
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        
        # Bounds (0 to 40% per strategy)
        bounds = tuple((0, 0.4) for _ in range(len(strategies_list)))
        
        # Initial guess (equal weight)
        x0 = np.array([1/len(strategies_list)] * len(strategies_list))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        # Store allocation
        self.capital_allocation = dict(zip(strategies_list, result.x))
        
        return self.capital_allocation
        
    def execute_portfolio(self, capital, market_data):
        """Execute portfolio based on optimal allocation"""
        
        portfolio_trades = []
        
        for strategy_name, allocation in self.capital_allocation.items():
            strategy_capital = capital * allocation
            
            if strategy_capital > 0:
                strategy = self.strategies[strategy_name]
                trades = strategy.generate_trades(strategy_capital, market_data)
                
                # Add strategy tag to trades
                for trade in trades:
                    trade['strategy'] = strategy_name
                    trade['allocation'] = allocation
                    
                portfolio_trades.extend(trades)
                
        return self.risk_manage_portfolio(portfolio_trades)
```

**2. Two Sigma's AI-Driven Approach**
```python
class TwoSigmaAITrading:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def build_ensemble_model(self, training_data):
        """Build ensemble of different ML models"""
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb
        
        # Prepare features
        X = self.engineer_features(training_data)
        y = training_data['target_return']
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5),
            'xgb': XGBRegressor(n_estimators=100, max_depth=6),
            'lgb': lgb.LGBMRegressor(n_estimators=100, max_depth=7),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50, 25))
        }
        
        # Train and validate each model
        model_scores = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            model_scores[name] = score
            self.models[name] = model
            
            # Feature importance (where applicable)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
                
        # Create ensemble weights based on validation scores
        total_score = sum(model_scores.values())
        self.ensemble_weights = {name: score/total_score 
                                for name, score in model_scores.items()}
        
        return self.models, self.ensemble_weights
        
    def engineer_features(self, data):
        """Comprehensive feature engineering for options"""
        
        features = pd.DataFrame()
        
        # Price-based features
        features['return_1d'] = data['close'].pct_change(1)
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)
        
        # Volatility features
        features['realized_vol_5d'] = features['return_1d'].rolling(5).std()
        features['realized_vol_20d'] = features['return_1d'].rolling(20).std()
        features['vol_ratio'] = features['realized_vol_5d'] / features['realized_vol_20d']
        
        # Options-specific features
        features['iv_rank'] = data['implied_vol'].rolling(252).rank(pct=True)
        features['put_call_ratio'] = data['put_volume'] / data['call_volume']
        features['iv_skew'] = data['25_delta_put_iv'] - data['25_delta_call_iv']
        
        # Term structure
        features['term_structure_slope'] = data['60_day_iv'] - data['30_day_iv']
        features['term_structure_curve'] = (data['90_day_iv'] - data['60_day_iv']) - \
                                          (data['60_day_iv'] - data['30_day_iv'])
        
        # Greeks features
        features['gamma_exposure'] = data['total_gamma'] * data['spot_price']**2 * 0.01
        features['vanna'] = data['delta'] * data['vega'] / data['spot_price']
        features['charm'] = data['delta'] * data['theta'] / data['spot_price']
        
        # Microstructure features
        features['bid_ask_spread'] = data['ask'] - data['bid']
        features['bid_ask_imbalance'] = (data['bid_size'] - data['ask_size']) / \
                                        (data['bid_size'] + data['ask_size'])
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(data['close'])
        features['macd_signal'] = self.calculate_macd(data['close'])
        
        # Market regime features
        features['vix_level'] = data['vix']
        features['vix_change'] = data['vix'].pct_change()
        features['correlation_spy_bonds'] = self.calculate_rolling_correlation(
            data['spy_returns'], data['tlt_returns'], 60
        )
        
        return features.fillna(method='ffill').fillna(0)
        
    def generate_predictions(self, current_data):
        """Generate ensemble predictions"""
        
        X = self.engineer_features(current_data)
        
        predictions = {}
        weighted_prediction = 0
        
        for name, model in self.models.items():
            pred = model.predict(X.iloc[-1:])
            predictions[name] = pred[0]
            weighted_prediction += pred[0] * self.ensemble_weights[name]
            
        return {
            'ensemble_prediction': weighted_prediction,
            'individual_predictions': predictions,
            'confidence': self.calculate_prediction_confidence(predictions)
        }
```

---

## Fundamentalne Strategie Opcyjne

### Advanced Greeks Trading

#### 1. Vega Trading Strategies

```python
class VegaTradingStrategy:
    def __init__(self):
        self.vega_limit = 10000  # Maximum vega exposure
        self.iv_percentile_threshold = 80  # Trade when IV rank > 80%
        
    def volatility_term_structure_trade(self, option_chain):
        """Trade volatility term structure anomalies"""
        
        # Calculate term structure
        term_structure = []
        expirations = sorted(option_chain['expiration'].unique())
        
        for exp in expirations:
            exp_data = option_chain[option_chain['expiration'] == exp]
            atm_options = exp_data[abs(exp_data['delta']) < 0.55]
            avg_iv = atm_options['implied_vol'].mean()
            days_to_exp = (exp - datetime.now()).days
            
            term_structure.append({
                'expiration': exp,
                'days': days_to_exp,
                'iv': avg_iv
            })
            
        ts_df = pd.DataFrame(term_structure)
        
        # Fit term structure model
        from scipy.optimize import curve_fit
        
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c
            
        params, _ = curve_fit(power_law, ts_df['days'], ts_df['iv'])
        
        # Find anomalies
        ts_df['model_iv'] = power_law(ts_df['days'], *params)
        ts_df['residual'] = ts_df['iv'] - ts_df['model_iv']
        ts_df['z_score'] = (ts_df['residual'] - ts_df['residual'].mean()) / \
                          ts_df['residual'].std()
        
        trades = []
        
        for _, row in ts_df.iterrows():
            if abs(row['z_score']) > 2:
                # Significant deviation from model
                if row['z_score'] > 2:
                    # IV too high for this expiration
                    trade = self.create_vega_short(row['expiration'], option_chain)
                else:
                    # IV too low
                    trade = self.create_vega_long(row['expiration'], option_chain)
                    
                trades.append(trade)
                
        return trades
        
    def create_vega_short(self, expiration, option_chain):
        """Create vega-short position (sell volatility)"""
        
        exp_options = option_chain[option_chain['expiration'] == expiration]
        current_price = exp_options.iloc[0]['underlying_price']
        
        # Iron butterfly for pure vega play
        trade = {
            'strategy': 'iron_butterfly',
            'expiration': expiration,
            'legs': [
                {'action': 'sell', 'type': 'call', 'strike': current_price},
                {'action': 'sell', 'type': 'put', 'strike': current_price},
                {'action': 'buy', 'type': 'call', 
                 'strike': current_price * 1.05},
                {'action': 'buy', 'type': 'put', 
                 'strike': current_price * 0.95}
            ],
            'target_vega': -500,
            'max_loss': self.calculate_max_loss_butterfly(current_price)
        }
        
        return trade
        
    def volatility_arbitrage_scanner(self, markets):
        """Scan for volatility arbitrage across related markets"""
        
        opportunities = []
        
        # Example: SPY vs SPX vs ES options
        spy_iv = self.get_atm_iv('SPY')
        spx_iv = self.get_atm_iv('SPX')
        es_iv = self.get_atm_iv('ES')
        
        # Adjust for dividends and multipliers
        spy_iv_adjusted = spy_iv * 1.01  # Adjust for dividends
        es_iv_adjusted = es_iv * 0.99    # Futures typically lower IV
        
        # Look for divergences
        if spx_iv > spy_iv_adjusted + 0.02:
            opportunities.append({
                'type': 'iv_spread',
                'sell': 'SPX',
                'buy': 'SPY',
                'edge': spx_iv - spy_iv_adjusted
            })
            
        return opportunities
```

#### 2. Gamma Scalping Automation

```python
class AutomatedGammaScalping:
    def __init__(self, symbol):
        self.symbol = symbol
        self.position = {'options': {}, 'stock': 0}
        self.pnl = 0
        self.rehedge_threshold = 50  # Re-hedge when delta > 50
        
    def initialize_position(self, current_price, volatility):
        """Initialize gamma scalping position"""
        
        # Buy ATM straddle for maximum gamma
        atm_strike = round(current_price / 5) * 5  # Round to nearest 5
        
        self.position['options'] = {
            'call': {
                'strike': atm_strike,
                'quantity': 10,
                'gamma': 0.05,
                'delta': 0.5,
                'theta': -0.10,
                'vega': 0.20
            },
            'put': {
                'strike': atm_strike,
                'quantity': 10,
                'gamma': 0.05,
                'delta': -0.5,
                'theta': -0.10,
                'vega': 0.20
            }
        }
        
        # Initial hedge (should be delta-neutral)
        self.calculate_hedge()
        
    def calculate_hedge(self):
        """Calculate required stock hedge"""
        
        total_delta = 0
        
        for option_type, details in self.position['options'].items():
            total_delta += details['delta'] * details['quantity'] * 100
            
        # Hedge to make delta-neutral
        self.position['stock'] = -total_delta
        
    def update_greeks(self, current_price, time_passed):
        """Update Greeks based on price movement"""
        
        price_change = current_price - self.last_price
        
        for option_type, details in self.position['options'].items():
            # Update delta based on gamma
            details['delta'] += details['gamma'] * price_change
            
            # Decay theta
            details['theta'] *= (1 - time_passed / 365)
            
            # Update gamma (simplified - would use model in practice)
            details['gamma'] *= (1 - abs(price_change) / current_price * 0.1)
            
        self.last_price = current_price
        
    def should_rehedge(self):
        """Determine if rehedging is needed"""
        
        total_delta = sum([opt['delta'] * opt['quantity'] * 100 
                          for opt in self.position['options'].values()])
        total_delta += self.position['stock']
        
        return abs(total_delta) > self.rehedge_threshold
        
    def execute_rehedge(self, current_price):
        """Execute rehedging trades"""
        
        old_stock_position = self.position['stock']
        self.calculate_hedge()
        
        shares_to_trade = self.position['stock'] - old_stock_position
        
        if shares_to_trade != 0:
            # Execute trade
            trade_price = current_price
            
            if shares_to_trade > 0:
                # Buy shares
                self.pnl -= shares_to_trade * trade_price
                print(f"BUY {shares_to_trade} shares at {trade_price}")
            else:
                # Sell shares
                self.pnl += abs(shares_to_trade) * trade_price
                print(f"SELL {abs(shares_to_trade)} shares at {trade_price}")
                
            return {
                'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                'quantity': abs(shares_to_trade),
                'price': trade_price,
                'total_delta_after': 0  # Should be delta-neutral
            }
            
        return None
        
    def calculate_pnl(self, current_price, initial_price):
        """Calculate total P&L including theta decay"""
        
        # Option P&L (simplified - would use proper model)
        option_pnl = 0
        
        for option_type, details in self.position['options'].items():
            if option_type == 'call':
                intrinsic = max(0, current_price - details['strike'])
            else:
                intrinsic = max(0, details['strike'] - current_price)
                
            # Add gamma P&L
            gamma_pnl = 0.5 * details['gamma'] * (current_price - initial_price)**2
            gamma_pnl *= details['quantity'] * 100
            
            # Subtract theta decay
            theta_pnl = details['theta'] * details['quantity'] * 100
            
            option_pnl += gamma_pnl + theta_pnl
            
        # Stock P&L
        stock_pnl = self.position['stock'] * (current_price - initial_price)
        
        total_pnl = option_pnl + stock_pnl + self.pnl
        
        return {
            'option_pnl': option_pnl,
            'stock_pnl': stock_pnl,
            'trading_pnl': self.pnl,
            'total_pnl': total_pnl
        }
```

#### 3. Advanced Delta Strategies

```python
class AdvancedDeltaStrategies:
    def __init__(self):
        self.delta_bands = [-100, -50, 0, 50, 100]
        self.portfolio_delta_limit = 1000
        
    def delta_targeting_strategy(self, market_outlook, option_chain):
        """Maintain specific delta exposure based on market view"""
        
        target_delta = self.calculate_target_delta(market_outlook)
        current_delta = self.calculate_portfolio_delta()
        
        delta_difference = target_delta - current_delta
        
        if abs(delta_difference) > 50:
            # Need to adjust
            trades = self.generate_delta_adjustment_trades(
                delta_difference, option_chain
            )
            return trades
            
        return []
        
    def calculate_target_delta(self, market_outlook):
        """Convert market outlook to target delta"""
        
        # Market outlook from -1 (very bearish) to +1 (very bullish)
        # Convert to delta exposure
        
        if market_outlook > 0.7:
            return 500  # Bullish
        elif market_outlook > 0.3:
            return 200  # Moderately bullish
        elif market_outlook > -0.3:
            return 0    # Neutral
        elif market_outlook > -0.7:
            return -200 # Moderately bearish
        else:
            return -500 # Bearish
            
    def smart_delta_hedging(self, portfolio, hedging_instruments):
        """Optimize delta hedging across multiple instruments"""
        
        from scipy.optimize import linprog
        
        current_delta = sum([pos['delta'] * pos['quantity'] 
                           for pos in portfolio])
        
        # Set up linear programming problem
        # Minimize: cost of hedging
        # Subject to: delta = 0
        
        n_instruments = len(hedging_instruments)
        
        # Objective: minimize cost
        costs = [inst['ask'] - inst['bid'] for inst in hedging_instruments]
        
        # Constraint: sum of deltas = -current_delta
        A_eq = [[inst['delta'] for inst in hedging_instruments]]
        b_eq = [-current_delta]
        
        # Bounds: position limits
        bounds = [(-inst['max_position'], inst['max_position']) 
                 for inst in hedging_instruments]
        
        # Solve
        result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, 
                        method='highs')
        
        if result.success:
            hedging_trades = []
            for i, quantity in enumerate(result.x):
                if abs(quantity) > 0.01:
                    hedging_trades.append({
                        'instrument': hedging_instruments[i]['symbol'],
                        'quantity': int(quantity),
                        'delta': hedging_instruments[i]['delta']
                    })
                    
            return hedging_trades
        else:
            return self.fallback_delta_hedge(current_delta)
```

---

## Zaawansowane Strategie Algorytmiczne

### Complex Multi-Leg Strategies

#### 1. Dispersion Trading

```python
class DispersionTrading:
    def __init__(self):
        self.index = 'SPX'
        self.components = self.get_sp500_components()
        self.correlation_window = 60
        
    def calculate_implied_correlation(self, date):
        """Calculate implied correlation from option prices"""
        
        # Index implied volatility
        index_iv = self.get_atm_iv(self.index, date)
        
        # Component implied volatilities and weights
        component_ivs = []
        weights = []
        
        for symbol in self.components[:50]:  # Top 50 for efficiency
            iv = self.get_atm_iv(symbol, date)
            weight = self.get_index_weight(symbol)
            component_ivs.append(iv)
            weights.append(weight)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Implied correlation from variance decomposition
        # Var(Index) = Sum(wi^2 * σi^2) + Sum(wi*wj*ρij*σi*σj)
        
        weighted_var_sum = np.sum(weights**2 * np.array(component_ivs)**2)
        
        # Solve for average correlation
        implied_correlation = (index_iv**2 - weighted_var_sum) / \
                            (np.sum(weights * np.array(component_ivs))**2 - weighted_var_sum)
        
        return np.clip(implied_correlation, 0, 1)
        
    def calculate_realized_correlation(self, lookback_days=60):
        """Calculate realized correlation from returns"""
        
        returns = pd.DataFrame()
        
        for symbol in self.components[:50]:
            prices = self.get_price_history(symbol, lookback_days)
            returns[symbol] = prices.pct_change()
            
        correlation_matrix = returns.corr()
        
        # Average pairwise correlation
        n = len(correlation_matrix)
        upper_triangle = np.triu(correlation_matrix, k=1)
        avg_correlation = upper_triangle.sum() / ((n * (n-1)) / 2)
        
        return avg_correlation
        
    def generate_dispersion_trade(self):
        """Generate dispersion trade signals"""
        
        implied_corr = self.calculate_implied_correlation(datetime.now())
        realized_corr = self.calculate_realized_correlation()
        
        correlation_spread = implied_corr - realized_corr
        
        trade = {'legs': []}
        
        if correlation_spread > 0.1:  # Implied correlation too high
            # Sell index volatility, buy component volatility
            
            # Sell index straddle
            index_straddle = self.create_straddle(self.index, 'sell', 10)
            trade['legs'].append(index_straddle)
            
            # Buy component straddles
            for symbol in self.components[:20]:  # Top 20 components
                weight = self.get_index_weight(symbol)
                size = int(10 * weight * 100)  # Scale by weight
                
                if size > 0:
                    component_straddle = self.create_straddle(symbol, 'buy', size)
                    trade['legs'].append(component_straddle)
                    
            trade['type'] = 'short_correlation'
            trade['expected_edge'] = correlation_spread * 1000  # Dollar edge
            
        elif correlation_spread < -0.1:  # Implied correlation too low
            # Buy index volatility, sell component volatility
            # (Opposite of above)
            trade['type'] = 'long_correlation'
            
        return trade if trade['legs'] else None
        
    def monitor_dispersion_position(self, position):
        """Real-time monitoring and adjustment"""
        
        metrics = {
            'correlation_pnl': 0,
            'vega_pnl': 0,
            'theta_collected': 0,
            'current_correlation': self.calculate_realized_correlation(5),
            'position_delta': 0
        }
        
        for leg in position['legs']:
            leg_metrics = self.calculate_leg_metrics(leg)
            metrics['vega_pnl'] += leg_metrics['vega_pnl']
            metrics['theta_collected'] += leg_metrics['theta']
            metrics['position_delta'] += leg_metrics['delta']
            
        # Correlation P&L (simplified)
        initial_corr = position['entry_correlation']
        current_corr = metrics['current_correlation']
        metrics['correlation_pnl'] = (initial_corr - current_corr) * \
                                     position['correlation_sensitivity']
        
        return metrics
```

#### 2. Volatility Surface Trading

```python
class VolatilitySurfaceTrading:
    def __init__(self):
        self.surface_models = ['SABR', 'SVI', 'Heston']
        self.calibration_weights = 'vega'  # Weight by vega in calibration
        
    def fit_svi_surface(self, option_data, expiry):
        """Fit SVI (Stochastic Volatility Inspired) model"""
        
        from scipy.optimize import differential_evolution
        
        def svi_volatility(k, a, b, rho, m, sigma):
            """SVI parameterization"""
            return np.sqrt(a + b * (rho * (k - m) + 
                          np.sqrt((k - m)**2 + sigma**2)))
            
        def objective(params):
            """Weighted least squares objective"""
            a, b, rho, m, sigma = params
            
            errors = []
            for _, opt in option_data[option_data['expiry'] == expiry].iterrows():
                k = np.log(opt['strike'] / opt['forward'])
                model_vol = svi_volatility(k, a, b, rho, m, sigma)
                market_vol = opt['implied_vol']
                weight = opt['vega'] if self.calibration_weights == 'vega' else 1
                
                errors.append(weight * (model_vol - market_vol)**2)
                
            return np.sum(errors)
            
        # Bounds for SVI parameters
        bounds = [
            (0, 1),      # a
            (0, 1),      # b  
            (-0.99, 0.99), # rho
            (-0.5, 0.5),   # m
            (0.01, 1)      # sigma
        ]
        
        # Differential evolution for global optimization
        result = differential_evolution(objective, bounds, maxiter=1000)
        
        return result.x
        
    def detect_surface_arbitrage(self, surface_params):
        """Detect arbitrage opportunities in volatility surface"""
        
        arbitrages = []
        
        # Butterfly arbitrage
        butterflies = self.check_butterfly_arbitrage(surface_params)
        arbitrages.extend(butterflies)
        
        # Calendar arbitrage
        calendars = self.check_calendar_arbitrage(surface_params)
        arbitrages.extend(calendars)
        
        # No-arbitrage conditions
        static_arb = self.check_static_arbitrage(surface_params)
        arbitrages.extend(static_arb)
        
        return arbitrages
        
    def check_butterfly_arbitrage(self, surface_params):
        """Check for butterfly arbitrage opportunities"""
        
        opportunities = []
        
        for expiry in surface_params['expiries']:
            strikes = surface_params['strikes'][expiry]
            
            for i in range(1, len(strikes) - 1):
                k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
                
                # Get implied vols
                iv1 = self.get_surface_iv(surface_params, k1, expiry)
                iv2 = self.get_surface_iv(surface_params, k2, expiry)
                iv3 = self.get_surface_iv(surface_params, k3, expiry)
                
                # Butterfly spread price
                c1 = self.black_scholes(k1, expiry, iv1)
                c2 = self.black_scholes(k2, expiry, iv2)
                c3 = self.black_scholes(k3, expiry, iv3)
                
                butterfly_price = c1 - 2*c2 + c3
                
                # Check convexity condition
                if butterfly_price < 0:
                    opportunities.append({
                        'type': 'butterfly',
                        'expiry': expiry,
                        'strikes': [k1, k2, k3],
                        'mispricing': abs(butterfly_price)
                    })
                    
        return opportunities
        
    def volatility_surface_pca(self, historical_surfaces):
        """PCA on volatility surfaces for trading signals"""
        
        from sklearn.decomposition import PCA
        
        # Flatten surfaces into vectors
        surface_vectors = []
        
        for date, surface in historical_surfaces.items():
            vector = []
            for expiry in surface['expiries']:
                for strike in surface['strikes']:
                    iv = surface['ivs'].get((expiry, strike), np.nan)
                    vector.append(iv)
                    
            surface_vectors.append(vector)
            
        # PCA
        pca = PCA(n_components=5)
        pca.fit(surface_vectors)
        
        # Current surface score
        current_vector = self.flatten_current_surface()
        current_scores = pca.transform([current_vector])[0]
        
        # Historical distribution of scores
        historical_scores = pca.transform(surface_vectors)
        
        # Z-scores for each component
        z_scores = []
        for i in range(5):
            mean = historical_scores[:, i].mean()
            std = historical_scores[:, i].std()
            z = (current_scores[i] - mean) / std
            z_scores.append(z)
            
        # Trading signals based on extreme scores
        signals = []
        
        if abs(z_scores[0]) > 2:  # First PC often level
            signals.append({
                'type': 'level',
                'direction': 'sell' if z_scores[0] > 2 else 'buy',
                'magnitude': abs(z_scores[0])
            })
            
        if abs(z_scores[1]) > 2:  # Second PC often slope
            signals.append({
                'type': 'slope',
                'direction': 'steepen' if z_scores[1] > 2 else 'flatten',
                'magnitude': abs(z_scores[1])
            })
            
        return signals, z_scores
```

### High-Frequency Options Strategies

#### 1. Latency Arbitrage

```python
class LatencyArbitrage:
    def __init__(self):
        self.venues = {
            'CBOE': {'latency': 0.1, 'maker_fee': -0.20, 'taker_fee': 0.30},
            'ISE': {'latency': 0.15, 'maker_fee': -0.18, 'taker_fee': 0.32},
            'PHLX': {'latency': 0.12, 'maker_fee': -0.22, 'taker_fee': 0.28},
            'NASDAQ': {'latency': 0.11, 'maker_fee': -0.19, 'taker_fee': 0.31}
        }
        self.positions = {}
        
    def detect_latency_opportunity(self, symbol, news_timestamp):
        """Detect latency arbitrage after news event"""
        
        opportunities = []
        
        # Get quotes from all venues
        quotes = {}
        for venue, info in self.venues.items():
            # Simulate quote arrival time based on latency
            arrival_time = news_timestamp + timedelta(
                milliseconds=info['latency']
            )
            quotes[venue] = self.get_quote_at_time(symbol, venue, arrival_time)
            
        # Find the venue that updates first
        first_venue = min(self.venues.keys(), 
                         key=lambda v: self.venues[v]['latency'])
        
        # New fair value based on news
        new_fair_value = quotes[first_venue]['mid']
        
        # Check other venues for stale quotes
        for venue, quote in quotes.items():
            if venue != first_venue:
                old_mid = quote['mid']
                
                if abs(new_fair_value - old_mid) > 0.05:  # 5 cent opportunity
                    opp = {
                        'venue': venue,
                        'old_price': old_mid,
                        'new_price': new_fair_value,
                        'edge': abs(new_fair_value - old_mid),
                        'action': 'buy' if new_fair_value > old_mid else 'sell',
                        'time_window': self.venues[venue]['latency'] - 
                                      self.venues[first_venue]['latency']
                    }
                    opportunities.append(opp)
                    
        return opportunities
        
    def execute_latency_trade(self, opportunity):
        """Execute latency arbitrage trade"""
        
        venue = opportunity['venue']
        
        # Hit the stale quote
        if opportunity['action'] == 'buy':
            # Buy at stale ask before it updates
            order = {
                'venue': venue,
                'side': 'buy',
                'price': opportunity['old_price'] + 0.01,  # Stale ask
                'size': 100,
                'type': 'IOC',  # Immediate or cancel
                'timestamp': datetime.now()
            }
        else:
            # Sell at stale bid
            order = {
                'venue': venue,
                'side': 'sell',
                'price': opportunity['old_price'] - 0.01,  # Stale bid
                'size': 100,
                'type': 'IOC',
                'timestamp': datetime.now()
            }
            
        # Execute with minimal latency
        execution = self.send_order_fast_path(order)
        
        if execution['filled']:
            # Immediately hedge or close on another venue
            hedge = self.create_hedge_order(execution, opportunity['new_price'])
            hedge_execution = self.send_order_fast_path(hedge)
            
            # Calculate P&L
            pnl = self.calculate_arbitrage_pnl(execution, hedge_execution)
            
            return {
                'success': True,
                'pnl': pnl,
                'execution_time': execution['latency'],
                'total_time': hedge_execution['timestamp'] - order['timestamp']
            }
            
        return {'success': False}
```

#### 2. Order Anticipation

```python
class OrderAnticipation:
    def __init__(self):
        self.order_flow_model = self.train_order_flow_model()
        self.min_confidence = 0.75
        
    def train_order_flow_model(self):
        """Train model to predict large orders"""
        
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Load historical order flow data
        # Features: order book imbalance, recent trades, patterns
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
        # Training code here...
        
        return model
        
    def detect_institutional_flow(self, order_book_snapshot):
        """Detect potential institutional order flow"""
        
        features = self.extract_flow_features(order_book_snapshot)
        
        # Predict probability of large order coming
        prob = self.order_flow_model.predict_proba([features])[0][1]
        
        if prob > self.min_confidence:
            # Analyze order book for direction
            direction = self.analyze_order_direction(order_book_snapshot)
            
            return {
                'probability': prob,
                'direction': direction,
                'expected_size': self.estimate_order_size(features),
                'expected_impact': self.estimate_price_impact(features)
            }
            
        return None
        
    def extract_flow_features(self, snapshot):
        """Extract features indicating institutional flow"""
        
        features = []
        
        # Order book imbalance
        bid_volume = sum([level[1] for level in snapshot['bids'][:10]])
        ask_volume = sum([level[1] for level in snapshot['asks'][:10]])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        features.append(imbalance)
        
        # Depth asymmetry
        bid_depth = len([l for l in snapshot['bids'] if l[1] > 100])
        ask_depth = len([l for l in snapshot['asks'] if l[1] > 100])
        depth_ratio = bid_depth / (ask_depth + 1)
        features.append(depth_ratio)
        
        # Recent large trades
        large_trades = [t for t in snapshot['recent_trades'] 
                       if t['size'] > 500]
        buy_volume = sum([t['size'] for t in large_trades 
                         if t['side'] == 'buy'])
        sell_volume = sum([t['size'] for t in large_trades 
                          if t['side'] == 'sell'])
        trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1)
        features.append(trade_imbalance)
        
        # Quote stuffing detection
        quote_updates = snapshot.get('quote_update_rate', 0)
        features.append(min(quote_updates / 100, 1))
        
        # Hidden liquidity indicators
        exec_prob = snapshot.get('hidden_execution_probability', 0.5)
        features.append(exec_prob)
        
        return features
        
    def front_run_strategy(self, detection):
        """Create position ahead of anticipated order"""
        
        # Note: This is for educational purposes
        # Front-running client orders is illegal
        # This represents prediction of public flow patterns
        
        if detection['probability'] > 0.8:
            position_size = int(detection['expected_size'] * 0.1)  # Take 10%
            
            if detection['direction'] == 'buy':
                # Buy before the large buy order
                entry_order = {
                    'side': 'buy',
                    'size': position_size,
                    'type': 'limit',
                    'price': 'best_ask',
                    'time_in_force': 'IOC'
                }
                
                # Plan exit after expected impact
                exit_price = 'current_price * (1 + expected_impact)'
                
            else:
                # Sell before large sell order
                entry_order = {
                    'side': 'sell',
                    'size': position_size,
                    'type': 'limit',
                    'price': 'best_bid',
                    'time_in_force': 'IOC'
                }
                
                exit_price = 'current_price * (1 - expected_impact)'
                
            return {
                'entry': entry_order,
                'exit_target': exit_price,
                'stop_loss': 'entry_price * 0.995',
                'confidence': detection['probability']
            }
            
        return None
```

---

## Modele Matematyczne i Wycena

### Advanced Pricing Models

#### 1. Stochastic Volatility Models

```python
class StochasticVolatilityModels:
    def __init__(self):
        self.model_type = 'Heston'
        
    def heston_characteristic_function(self, phi, S0, v0, kappa, theta, sigma, 
                                      rho, lambd, tau, r):
        """
        Heston model characteristic function
        Used for option pricing via Fourier methods
        """
        
        # Characteristic function parameters
        a = kappa * theta
        u = -0.5
        b = kappa + lambd
        
        d = np.sqrt((rho * sigma * phi * 1j - b)**2 - 
                   sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - rho * sigma * phi * 1j + d) / \
            (b - rho * sigma * phi * 1j - d)
        
        C = r * phi * 1j * tau + (a / sigma**2) * \
            ((b - rho * sigma * phi * 1j + d) * tau - 
             2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
        
        D = ((b - rho * sigma * phi * 1j + d) / sigma**2) * \
            ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
        
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))
        
    def heston_price_fft(self, option_type, S0, K, T, r, v0, kappa, theta, 
                        sigma, rho, lambd):
        """Price options using FFT with Heston model"""
        
        N = 4096  # FFT points
        alpha = 1.5
        eta = 0.25
        b = 200
        
        # Log-strike grid
        k = np.linspace(-b, b, N)
        delta_k = k[1] - k[0]
        
        # Simpson's rule weights
        simpson_weights = np.ones(N)
        simpson_weights[0] = simpson_weights[-1] = 0.5
        simpson_weights[1::2] = 4/3
        simpson_weights[2::2] = 2/3
        
        # Characteristic function evaluation
        u = np.arange(N) * eta
        
        # Modified characteristic function for call option
        psi = np.zeros(N, dtype=complex)
        
        for i in range(N):
            integrand = self.heston_characteristic_function(
                u[i] - (alpha + 1) * 1j, S0, v0, kappa, theta, 
                sigma, rho, lambd, T, r
            )
            
            psi[i] = np.exp(-r * T) * integrand / \
                     (alpha**2 + alpha - u[i]**2 + 1j * (2 * alpha + 1) * u[i])
                     
        # FFT
        x = np.exp(1j * b * u) * psi * simpson_weights
        fft_result = np.fft.fft(x)
        
        # Extract prices
        log_strikes = k
        call_prices = np.real(np.exp(-alpha * log_strikes) * fft_result / np.pi)
        
        # Interpolate to get price at strike K
        log_K = np.log(K / S0)
        price = np.interp(log_K, log_strikes, call_prices) * S0
        
        if option_type == 'put':
            # Put-call parity
            price = price - S0 + K * np.exp(-r * T)
            
        return price
        
    def calibrate_heston(self, market_prices, strikes, expiries, S0, r):
        """Calibrate Heston model to market prices"""
        
        from scipy.optimize import differential_evolution
        
        def objective(params):
            v0, kappa, theta, sigma, rho = params
            lambd = 0  # Risk-neutral measure
            
            errors = []
            
            for i, (K, T, market_price) in enumerate(
                zip(strikes, expiries, market_prices)
            ):
                model_price = self.heston_price_fft(
                    'call', S0, K, T, r, v0, kappa, theta, 
                    sigma, rho, lambd
                )
                
                # Weighted by vega (approximated)
                vega_proxy = np.sqrt(T) * S0 * 0.4  # Simplified
                error = ((model_price - market_price) / vega_proxy)**2
                errors.append(error)
                
            return np.sum(errors)
            
        # Parameter bounds
        bounds = [
            (0.01, 0.5),   # v0
            (0.1, 10),     # kappa  
            (0.01, 0.5),   # theta
            (0.1, 2),      # sigma
            (-0.95, 0.95)  # rho
        ]
        
        # Optimize
        result = differential_evolution(objective, bounds, maxiter=100, 
                                      workers=-1)
        
        return {
            'v0': result.x[0],
            'kappa': result.x[1],
            'theta': result.x[2],
            'sigma': result.x[3],
            'rho': result.x[4],
            'error': result.fun
        }
```

#### 2. Local Volatility Models

```python
class LocalVolatilityModel:
    def __init__(self):
        self.vol_surface = None
        self.local_vol_surface = None
        
    def dupire_local_volatility(self, S, K, T, r, q):
        """
        Calculate local volatility using Dupire formula
        σ_local^2 = (∂C/∂T + qC + (r-q)K∂C/∂K) / (0.5 * K^2 * ∂²C/∂K²)
        """
        
        # Get implied volatility at (K, T)
        iv = self.get_implied_vol(K, T)
        
        # Calculate call price and Greeks using Black-Scholes
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r - q + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        d2 = d1 - iv*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # Numerical derivatives
        dK = K * 0.001
        dT = T * 0.001
        
        # ∂C/∂T (Theta)
        call_T_plus = self.black_scholes_price(S, K, T+dT, r, q, 
                                              self.get_implied_vol(K, T+dT))
        dC_dT = (call_T_plus - call_price) / dT
        
        # ∂C/∂K (Delta with respect to strike)
        call_K_plus = self.black_scholes_price(S, K+dK, T, r, q,
                                              self.get_implied_vol(K+dK, T))
        call_K_minus = self.black_scholes_price(S, K-dK, T, r, q,
                                               self.get_implied_vol(K-dK, T))
        dC_dK = (call_K_plus - call_K_minus) / (2*dK)
        
        # ∂²C/∂K² (Gamma with respect to strike)
        d2C_dK2 = (call_K_plus - 2*call_price + call_K_minus) / (dK**2)
        
        # Dupire formula
        numerator = dC_dT + q*call_price + (r-q)*K*dC_dK
        denominator = 0.5 * K**2 * d2C_dK2
        
        if denominator > 0:
            local_var = numerator / denominator
            local_vol = np.sqrt(max(local_var, 0.0001))
        else:
            local_vol = iv  # Fallback to implied vol
            
        return local_vol
        
    def build_local_vol_surface(self, spot, strikes, expiries, risk_free_rate):
        """Build entire local volatility surface"""
        
        local_vol_surface = {}
        
        for T in expiries:
            for K in strikes:
                local_vol = self.dupire_local_volatility(
                    spot, K, T, risk_free_rate, 0
                )
                local_vol_surface[(K, T)] = local_vol
                
        self.local_vol_surface = local_vol_surface
        return local_vol_surface
        
    def monte_carlo_local_vol(self, S0, K, T, r, n_paths=10000, n_steps=100):
        """Price option using Monte Carlo with local volatility"""
        
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_steps):
            t = i * dt
            
            for j in range(n_paths):
                S = paths[j, i]
                
                # Get local volatility at (S, t)
                local_vol = self.interpolate_local_vol(S, t)
                
                # Generate random shock
                dW = np.random.normal(0, np.sqrt(dt))
                
                # Euler step
                paths[j, i+1] = S * np.exp((r - 0.5*local_vol**2)*dt + 
                                          local_vol*dW)
                
        # Calculate payoff
        payoffs = np.maximum(paths[:, -1] - K, 0)  # Call option
        
        # Discount
        price = np.exp(-r*T) * np.mean(payoffs)
        
        # Standard error
        std_error = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, std_error
```

---

## Machine Learning w Opcjach

### Deep Learning Models

#### 1. LSTM for Volatility Prediction

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class LSTMVolatilityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.lookback = 60
        
    def create_model(self, input_shape):
        """Create LSTM model for volatility prediction"""
        
        model = models.Sequential([
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Custom loss function for volatility
        def volatility_loss(y_true, y_pred):
            # Asymmetric loss - penalize under-prediction more
            error = y_true - y_pred
            return tf.where(error > 0, 
                          error * 1.5,  # Under-prediction penalty
                          tf.abs(error))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=volatility_loss,
            metrics=['mae', 'mse']
        )
        
        return model
        
    def prepare_data(self, price_data, option_data):
        """Prepare data for LSTM training"""
        
        features = []
        
        # Price-based features
        returns = np.log(price_data['close'] / price_data['close'].shift(1))
        features.append(returns)
        
        # Realized volatility at different scales
        for window in [5, 10, 20, 60]:
            rv = returns.rolling(window).std() * np.sqrt(252)
            features.append(rv)
            
        # GARCH volatility
        garch_vol = self.calculate_garch_volatility(returns)
        features.append(garch_vol)
        
        # Options-based features
        features.append(option_data['atm_iv'])
        features.append(option_data['skew'])
        features.append(option_data['term_structure'])
        features.append(option_data['put_call_ratio'])
        
        # Volume features
        features.append(np.log(price_data['volume']))
        features.append(np.log(option_data['option_volume']))
        
        # Technical indicators
        rsi = self.calculate_rsi(price_data['close'])
        features.append(rsi)
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.lookback, len(feature_matrix) - 1):
            X.append(feature_matrix[i-self.lookback:i])
            # Predict next day's realized volatility
            y.append(returns[i+1:i+21].std() * np.sqrt(252))  # 20-day forward vol
            
        return np.array(X), np.array(y)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train LSTM model"""
        
        # Create model
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
        
    def predict_volatility_path(self, current_data, horizon=30):
        """Predict volatility path for next N days"""
        
        predictions = []
        current_sequence = current_data[-self.lookback:]
        
        for _ in range(horizon):
            # Predict next volatility
            next_vol = self.model.predict(
                current_sequence.reshape(1, self.lookback, -1),
                verbose=0
            )[0, 0]
            
            predictions.append(next_vol)
            
            # Update sequence (simplified - would need to generate other features)
            new_features = current_sequence[-1].copy()
            new_features[0] = next_vol  # Update vol feature
            
            # Shift and append
            current_sequence = np.vstack([current_sequence[1:], new_features])
            
        return np.array(predictions)
        
    def calculate_garch_volatility(self, returns):
        """Calculate GARCH(1,1) volatility forecast"""
        from arch import arch_model
        
        # Fit GARCH model
        model = arch_model(returns.dropna(), vol='Garch', p=1, q=1)
        result = model.fit(disp='off')
        
        # Get conditional volatility
        return result.conditional_volatility

#### 2. Database Architecture

```python
class OptionsDataArchitecture:
    def __init__(self):
        self.tick_db = TickDatabase()
        self.analytics_db = AnalyticsDatabase()
        self.reference_db = ReferenceDatabase()
        
class TickDatabase:
    """High-performance tick data storage"""
    
    def __init__(self):
        self.storage_engine = 'Arctic'  # Or KDB+
        
    def setup_arctic_store(self):
        """Setup Arctic for tick data storage"""
        
        from arctic import Arctic
        import pandas as pd
        
        # Connect to MongoDB
        store = Arctic('localhost')
        
        # Create libraries for different data types
        if 'options_ticks' not in store.list_libraries():
            store.initialize_library('options_ticks', 
                                   lib_type='TickStore')
            
        if 'options_quotes' not in store.list_libraries():
            store.initialize_library('options_quotes', 
                                   lib_type='VersionStore')
            
        return store
        
    def store_tick_data(self, symbol, data):
        """Store tick data efficiently"""
        
        store = self.setup_arctic_store()
        tick_lib = store['options_ticks']
        
        # Convert to Arctic format
        tick_data = []
        for tick in data:
            tick_data.append({
                'index': tick['timestamp'],
                'price': tick['price'],
                'size': tick['size'],
                'bid': tick['bid'],
                'ask': tick['ask'],
                'bid_size': tick['bid_size'],
                'ask_size': tick['ask_size']
            })
            
        # Write to Arctic
        tick_lib.write(symbol, tick_data)
        
    def query_tick_data(self, symbol, start_date, end_date):
        """Query tick data with date range"""
        
        store = self.setup_arctic_store()
        tick_lib = store['options_ticks']
        
        # Read data
        data = tick_lib.read(symbol, 
                           date_range=(start_date, end_date))
        
        return pd.DataFrame(data)
        
class AnalyticsDatabase:
    """Database for analytics and aggregated data"""
    
    def __init__(self):
        self.engine = self.create_postgres_engine()
        
    def create_postgres_engine(self):
        """Create PostgreSQL connection"""
        
        from sqlalchemy import create_engine
        
        engine = create_engine(
            'postgresql://user:password@localhost:5432/options_analytics',
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True
        )
        
        return engine
        
    def create_tables(self):
        """Create analytics tables"""
        
        sql = """
        -- Volatility surface snapshots
        CREATE TABLE IF NOT EXISTS volatility_surfaces (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10),
            snapshot_time TIMESTAMP,
            expiration DATE,
            strike DECIMAL(10,2),
            implied_vol DECIMAL(6,4),
            bid_vol DECIMAL(6,4),
            ask_vol DECIMAL(6,4),
            volume INTEGER,
            open_interest INTEGER,
            INDEX idx_symbol_time (symbol, snapshot_time),
            INDEX idx_expiration (expiration)
        );
        
        -- Greeks time series
        CREATE TABLE IF NOT EXISTS greeks_history (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10),
            option_type VARCHAR(4),
            strike DECIMAL(10,2),
            expiration DATE,
            timestamp TIMESTAMP,
            delta DECIMAL(6,4),
            gamma DECIMAL(8,6),
            theta DECIMAL(8,4),
            vega DECIMAL(8,4),
            rho DECIMAL(8,4),
            INDEX idx_option (symbol, strike, expiration)
        );
        
        -- Trading signals
        CREATE TABLE IF NOT EXISTS trading_signals (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(50),
            symbol VARCHAR(10),
            signal_time TIMESTAMP,
            signal_type VARCHAR(20),
            strength DECIMAL(4,2),
            metadata JSONB,
            INDEX idx_strategy_time (strategy_name, signal_time)
        );
        
        -- Position history
        CREATE TABLE IF NOT EXISTS position_history (
            id SERIAL PRIMARY KEY,
            position_id VARCHAR(50),
            symbol VARCHAR(10),
            option_type VARCHAR(4),
            strike DECIMAL(10,2),
            expiration DATE,
            quantity INTEGER,
            entry_time TIMESTAMP,
            entry_price DECIMAL(10,4),
            exit_time TIMESTAMP,
            exit_price DECIMAL(10,4),
            pnl DECIMAL(12,2),
            INDEX idx_position (position_id),
            INDEX idx_symbol_time (symbol, entry_time)
        );
        """
        
        with self.engine.connect() as conn:
            conn.execute(sql)

#### 3. Real-time Processing Pipeline

```python
class RealtimeProcessingPipeline:
    def __init__(self):
        self.kafka_consumer = self.setup_kafka()
        self.redis_cache = self.setup_redis()
        self.processing_threads = []
        
    def setup_kafka(self):
        """Setup Kafka for streaming data"""
        
        from kafka import KafkaConsumer, KafkaProducer
        
        # Consumer for market data
        consumer = KafkaConsumer(
            'options_market_data',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='options_trading_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Producer for signals
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        return consumer
        
    def setup_redis(self):
        """Setup Redis for caching"""
        
        import redis
        
        r = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        return r
        
    def process_stream(self):
        """Main processing loop"""
        
        for message in self.kafka_consumer:
            data = message.value
            
            # Update cache
            self.update_cache(data)
            
            # Process through strategies
            signals = self.run_strategies(data)
            
            # Send signals to execution
            if signals:
                self.producer.send('trading_signals', signals)
                
    def update_cache(self, data):
        """Update Redis cache with latest data"""
        
        key = f"{data['symbol']}:{data['expiration']}:{data['strike']}"
        
        # Store latest quote
        self.redis_cache.hset(key, mapping={
            'bid': data['bid'],
            'ask': data['ask'],
            'last': data['last'],
            'iv': data['implied_vol'],
            'volume': data['volume'],
            'timestamp': data['timestamp']
        })
        
        # Set expiration
        self.redis_cache.expire(key, 3600)  # 1 hour
        
    def run_strategies(self, data):
        """Run trading strategies on new data"""
        
        signals = []
        
        # Run each strategy in parallel
        import concurrent.futures
        
        strategies = [
            self.volatility_arbitrage_check,
            self.gamma_scalping_check,
            self.dispersion_check,
            self.event_driven_check
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(strategy, data) 
                      for strategy in strategies]
            
            for future in concurrent.futures.as_completed(futures):
                signal = future.result()
                if signal:
                    signals.append(signal)
                    
        return signals

---

## Risk Management Framework

### Comprehensive Risk System

#### 1. Portfolio Risk Management

```python
class PortfolioRiskManager:
    def __init__(self):
        self.risk_limits = {
            'max_var_95': 50000,  # Max VaR at 95% confidence
            'max_var_99': 100000,  # Max VaR at 99% confidence
            'max_delta': 10000,    # Max delta exposure
            'max_gamma': 1000,     # Max gamma exposure
            'max_vega': 50000,     # Max vega exposure
            'max_theta': -10000,   # Max theta (daily decay)
            'max_concentration': 0.2,  # Max 20% in single position
            'max_leverage': 3.0    # Max 3x leverage
        }
        
    def calculate_var(self, portfolio, confidence_level=0.95, 
                     horizon_days=1, method='historical'):
        """Calculate Value at Risk"""
        
        if method == 'historical':
            return self.historical_var(portfolio, confidence_level, horizon_days)
        elif method == 'parametric':
            return self.parametric_var(portfolio, confidence_level, horizon_days)
        elif method == 'monte_carlo':
            return self.monte_carlo_var(portfolio, confidence_level, horizon_days)
            
    def historical_var(self, portfolio, confidence_level, horizon_days):
        """Historical simulation VaR"""
        
        # Get historical returns for portfolio components
        returns = []
        
        for position in portfolio:
            hist_returns = self.get_historical_returns(
                position['symbol'], 
                lookback_days=252
            )
            
            # Weight by position size
            weighted_returns = hist_returns * position['delta_dollars']
            returns.append(weighted_returns)
            
        # Aggregate portfolio returns
        portfolio_returns = pd.DataFrame(returns).sum(axis=0)
        
        # Scale to horizon
        scaled_returns = portfolio_returns * np.sqrt(horizon_days)
        
        # Calculate VaR
        var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        
        return abs(var)
        
    def monte_carlo_var(self, portfolio, confidence_level, horizon_days, 
                       n_simulations=10000):
        """Monte Carlo VaR"""
        
        # Get portfolio parameters
        portfolio_value = sum([p['market_value'] for p in portfolio])
        
        # Estimate portfolio volatility
        returns = []
        weights = []
        
        for position in portfolio:
            ret = self.get_historical_returns(position['symbol'])
            returns.append(ret)
            weights.append(position['market_value'] / portfolio_value)
            
        returns_df = pd.DataFrame(returns).T
        cov_matrix = returns_df.cov()
        
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Run simulations
        simulated_returns = np.random.normal(
            0, 
            portfolio_vol * np.sqrt(horizon_days),
            n_simulations
        )
        
        # Calculate VaR
        var = np.percentile(simulated_returns * portfolio_value,
                          (1 - confidence_level) * 100)
        
        return abs(var)
        
    def calculate_stress_tests(self, portfolio):
        """Run stress test scenarios"""
        
        scenarios = {
            'market_crash': {
                'spot_change': -0.20,
                'vol_change': 0.50,
                'correlation': 0.9
            },
            'vol_spike': {
                'spot_change': -0.05,
                'vol_change': 1.0,
                'correlation': 0.7
            },
            'black_swan': {
                'spot_change': -0.30,
                'vol_change': 2.0,
                'correlation': 1.0
            },
            'squeeze': {
                'spot_change': 0.15,
                'vol_change': 0.30,
                'correlation': 0.5
            }
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            scenario_pnl = 0
            
            for position in portfolio:
                # Calculate position P&L under scenario
                position_pnl = self.calculate_scenario_pnl(
                    position, params
                )
                scenario_pnl += position_pnl
                
            results[scenario_name] = {
                'pnl': scenario_pnl,
                'pnl_pct': scenario_pnl / self.get_portfolio_value(portfolio)
            }
            
        return results
        
    def calculate_scenario_pnl(self, position, scenario):
        """Calculate position P&L under stress scenario"""
        
        # Current values
        spot = position['underlying_price']
        iv = position['implied_vol']
        
        # Shocked values
        new_spot = spot * (1 + scenario['spot_change'])
        new_iv = iv * (1 + scenario['vol_change'])
        
        # Recalculate option price
        if position['type'] == 'option':
            current_price = position['market_value']
            
            # Use Black-Scholes with shocked inputs
            new_price = self.black_scholes(
                new_spot,
                position['strike'],
                position['time_to_expiry'],
                position['risk_free_rate'],
                new_iv,
                position['option_type']
            )
            
            pnl = (new_price - current_price) * position['quantity'] * 100
            
        else:  # Stock position
            pnl = scenario['spot_change'] * position['market_value']
            
        return pnl
    def __init__(self, d_model=512, n_heads=8, n_layers=6):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
    def build_transformer_model(self):
        """Build transformer model for options prediction"""
        
        import tensorflow as tf
        from tensorflow.keras import layers
        
        # Input layers
        price_input = layers.Input(shape=(None, 10), name='price_features')
        options_input = layers.Input(shape=(None, 20), name='options_features')
        
        # Combine inputs
        combined = layers.Concatenate()([price_input, options_input])
        
        # Linear projection
        x = layers.Dense(self.d_model)(combined)
        
        # Positional encoding
        positions = tf.range(start=0, limit=tf.shape(x)[1], dtype=tf.float32)
        position_encoding = self.positional_encoding(positions, self.d_model)
        x = x + position_encoding
        
        # Transformer blocks
        for _ in range(self.n_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads
            )(x, x)
            
            # Add & Norm
            x = layers.LayerNormalization()(x + attn_output)
            
            # Feed forward
            ff_output = layers.Dense(self.d_model * 4, activation='relu')(x)
            ff_output = layers.Dense(self.d_model)(ff_output)
            
            # Add & Norm
            x = layers.LayerNormalization()(x + ff_output)
            
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Multiple outputs
        price_output = layers.Dense(1, name='price_prediction')(x)
        vol_output = layers.Dense(1, activation='softplus', 
                                 name='volatility_prediction')(x)
        direction_output = layers.Dense(3, activation='softmax', 
                                       name='direction_prediction')(x)
        
        model = tf.keras.Model(
            inputs=[price_input, options_input],
            outputs=[price_output, vol_output, direction_output]
        )
        
        return model
        
    def positional_encoding(self, position, d_model):
        """Sinusoidal positional encoding"""
        
        import tensorflow as tf
        
        angle_rads = self.get_angles(
            position,
            tf.range(d_model, dtype=tf.float32),
            d_model
        )
        
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return pos[:, tf.newaxis] * angle_rates[tf.newaxis, :]

---

## Infrastruktura i Technologia

### System Architecture

#### 1. Low-Latency Trading Infrastructure

```python
class TradingInfrastructure:
    def __init__(self):
        self.components = {
            'market_data': MarketDataHandler(),
            'order_management': OrderManagementSystem(),
            'risk_engine': RiskEngine(),
            'execution': ExecutionEngine(),
            'monitoring': SystemMonitoring()
        }
        
class MarketDataHandler:
    def __init__(self):
        self.feed_handlers = {}
        self.data_queue = deque(maxlen=100000)
        self.latency_target = 0.1  # milliseconds
        
    def setup_multicast_feeds(self):
        """Setup low-latency multicast feeds"""
        
        import socket
        import struct
        
        # OPRA feed configuration
        MULTICAST_GROUP = '224.0.0.251'
        SERVER_ADDRESS = ('', 10000)
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind to server address
        sock.bind(SERVER_ADDRESS)
        
        # Tell OS to add socket to multicast group
        group = socket.inet_aton(MULTICAST_GROUP)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Set socket to non-blocking
        sock.setblocking(0)
        
        return sock
        
    def parse_opra_message(self, data):
        """Parse OPRA protocol message"""
        
        # OPRA message structure (simplified)
        message_type = data[0]
        
        if message_type == ord('q'):  # Quote message
            return self.parse_quote(data)
        elif message_type == ord('t'):  # Trade message
            return self.parse_trade(data)
        elif message_type == ord('i'):  # NBBO update
            return self.parse_nbbo(data)
            
    def process_market_data(self, message):
        """Process incoming market data with minimal latency"""
        
        import time
        
        start_time = time.perf_counter_ns()
        
        # Update order book
        if message['type'] == 'quote':
            self.update_order_book(message)
            
        # Check for signals
        signals = self.check_signals(message)
        
        # Send to execution if signal triggered
        if signals:
            self.send_to_execution(signals)
            
        # Measure latency
        latency = (time.perf_counter_ns() - start_time) / 1_000_000  # ms
        
        if latency > self.latency_target:
            self.log_latency_breach(latency, message)
            
        return latency

class OrderManagementSystem:
    def __init__(self):
        self.orders = {}
        self.executions = {}
        self.positions = {}
        
    def create_order(self, symbol, side, quantity, order_type, **kwargs):
        """Create new order with validation"""
        
        order = {
            'id': self.generate_order_id(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'type': order_type,
            'status': 'pending',
            'timestamp': datetime.now(),
            **kwargs
        }
        
        # Risk checks
        if not self.pass_risk_checks(order):
            order['status'] = 'rejected'
            order['reject_reason'] = 'Failed risk checks'
            return order
            
        # Add to order management
        self.orders[order['id']] = order
        
        # Route order
        self.route_order(order)
        
        return order
        
    def route_order(self, order):
        """Smart order routing"""
        
        # Get best venue for execution
        venues = self.get_venue_quotes(order['symbol'])
        
        # Sort by price and fees
        if order['side'] == 'buy':
            venues.sort(key=lambda v: v['ask'] + v['take_fee'])
        else:
            venues.sort(key=lambda v: -(v['bid'] - v['take_fee']))
            
        # Send to best venue
        best_venue = venues[0]
        self.send_order_to_venue(order, best_venue['venue'])
        
class ExecutionEngine:
    def __init__(self):
        self.execution_algos = {
            'aggressive': self.aggressive_execution,
            'passive': self.passive_execution,
            'iceberg': self.iceberg_execution,
            'sniper': self.sniper_execution
        }
        
    def aggressive_execution(self, order):
        """Aggressive execution - cross the spread"""
        
        # Take liquidity immediately
        if order['side'] == 'buy':
            # Hit the ask
            execution_price = self.get_best_ask(order['symbol'])
        else:
            # Hit the bid
            execution_price = self.get_best_bid(order['symbol'])
            
        return self.execute_at_price(order, execution_price, 'IOC')
        
    def passive_execution(self, order):
        """Passive execution - provide liquidity"""
        
        # Place limit order at favorable price
        if order['side'] == 'buy':
            # Place at bid
            limit_price = self.get_best_bid(order['symbol'])
        else:
            # Place at ask
            limit_price = self.get_best_ask(order['symbol'])
            
        return self.place_limit_order(order, limit_price)
        
    def iceberg_execution(self, order, show_size=100):
        """Iceberg order - hide large size"""
        
        total_size = order['quantity']
        executed = 0
        
        while executed < total_size:
            # Show only small size
            visible_size = min(show_size, total_size - executed)
            
            # Place visible order
            partial_order = order.copy()
            partial_order['quantity'] = visible_size
            
            result = self.passive_execution(partial_order)
            executed += result['filled_quantity']
            
            # Random delay to avoid detection
            time.sleep(random.uniform(0.1, 0.5))
            
        return executed
        
    def sniper_execution(self, order, trigger_condition):
        """Sniper execution - wait for specific condition"""
        
        while not trigger_condition():
            time.sleep(0.001)  # 1ms polling
            
        # Execute immediately when condition met
        return self.aggressive_execution(order)
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        # Neural network
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()
        
    def build_network(self):
        """Build deep Q-network"""
        
        class DQN(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQN, self).__init__()
                
                self.fc1 = nn.Linear(input_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, output_size)
                
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = torch.relu(self.fc4(x))
                x = self.fc5(x)
                return x
                
        return DQN(self.state_size, self.action_size)
        
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
            
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class OptionsRLEnvironment:
    def __init__(self, option_chain, initial_capital=100000):
        self.option_chain = option_chain
        self.initial_capital = initial_capital
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.current_step = 0
        self.done = False
        
        return self.get_state()
        
    def get_state(self):
        """Get current state representation"""
        
        state = []
        
        # Market state
        current_data = self.option_chain.iloc[self.current_step]
        
        # Price and volume
        state.append(current_data['underlying_price'])
        state.append(current_data['volume'])
        
        # Volatility features
        state.append(current_data['implied_volatility'])
        state.append(current_data['historical_volatility'])
        state.append(current_data['iv_rank'])
        
        # Greeks
        state.append(current_data['delta'])
        state.append(current_data['gamma'])
        state.append(current_data['vega'])
        state.append(current_data['theta'])
        
        # Position features
        state.append(self.capital / self.initial_capital)
        state.append(len(self.positions))
        state.append(self.calculate_portfolio_greeks()['delta'])
        state.append(self.calculate_portfolio_greeks()['gamma'])
        
        return np.array(state)
        
    def step(self, action):
        """Execute action and return new state, reward, done"""
        
        # Define action space
        actions = {
            0: 'hold',
            1: 'buy_call',
            2: 'sell_call',
            3: 'buy_put',
            4: 'sell_put',
            5: 'buy_straddle',
            6: 'sell_straddle',
            7: 'close_all'
        }
        
        action_type = actions[action]
        
        # Execute action
        if action_type == 'buy_call':
            self.execute_trade('call', 'buy', 1)
        elif action_type == 'sell_call':
            self.execute_trade('call', 'sell', 1)
        elif action_type == 'buy_put':
            self.execute_trade('put', 'buy', 1)
        elif action_type == 'sell_put':
            self.execute_trade('put', 'sell', 1)
        elif action_type == 'buy_straddle':
            self.execute_trade('call', 'buy', 1)
            self.execute_trade('put', 'buy', 1)
        elif action_type == 'sell_straddle':
            self.execute_trade('call', 'sell', 1)
            self.execute_trade('put', 'sell', 1)
        elif action_type == 'close_all':
            self.close_all_positions()
            
        # Calculate reward
        old_value = self.capital
        self.update_positions()
        new_value = self.calculate_portfolio_value()
        
        reward = (new_value - old_value) / self.initial_capital
        
        # Apply risk penalty
        portfolio_greeks = self.calculate_portfolio_greeks()
        risk_penalty = abs(portfolio_greeks['delta']) * 0.001 + \
                      abs(portfolio_greeks['gamma']) * 0.01 + \
                      abs(portfolio_greeks['vega']) * 0.001
        
        reward -= risk_penalty
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.option_chain) - 1:
            self.done = True
            
        # Check for bankruptcy
        if self.capital < self.initial_capital * 0.2:
            self.done = True
            reward -= 1.0  # Large penalty for bankruptcy
            
        return self.get_state(), reward, self.done
        
    def execute_trade(self, option_type, action, quantity):
        """Execute option trade"""
        
        current_option = self.get_current_option(option_type)
        
        if action == 'buy':
            cost = current_option['ask'] * quantity * 100
            if cost <= self.capital:
                self.capital -= cost
                self.add_position(option_type, quantity, current_option)
        else:  # sell
            # For simplicity, assume we can always sell
            premium = current_option['bid'] * quantity * 100
            self.capital += premium
            self.add_position(option_type, -quantity, current_option)
            
    def train_agent(self, episodes=1000):
        """Train RL agent"""
        
        agent = OptionsRLAgent(state_size=13, action_size=8)
        
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            
            while not self.done:
                action = agent.get_action(state)
                next_state, reward, done = self.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if len(agent.memory) > 32:
                    agent.replay(32)
                    
            # Update target network
            if episode % 10 == 0:
                agent.update_target_network()
                
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
                  
        return agent
            