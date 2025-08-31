#!/usr/bin/env python3
"""
INSTITUTIONAL COMBINED STRATEGY - PAPER TRADING DEPLOYMENT
===========================================================
Deploy the institutional-grade combined strategy to IBKR paper trading
Ready for live execution with full risk management and monitoring

PORTFOLIO ALLOCATION:
- TIER 2 (65%): XAR, EWG, XLF, TSLA, FXI (13% each)
- TIER 3 (15%): AVGO, NVDA (7.5% each)
- CASH RESERVE (20%): Risk management buffer
"""

import sys
import os
from datetime import datetime
import signal
import time
from pathlib import Path

# Import our strategy
from INSTITUTIONAL_COMBINED_STRATEGY import InstitutionalCombinedStrategy, StrategyConfig, logger

class PaperTradingDeployment:
    """
    Deployment wrapper for paper trading with monitoring and safety controls
    """
    
    def __init__(self):
        self.strategy = None
        self.running = False
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
        if self.strategy:
            logger.info("💾 Saving final performance report...")
            self.strategy.save_performance_report()
            
            logger.info("🔌 Disconnecting from IBKR...")
            self.strategy.disconnect_from_ibkr()
        
        logger.info("✅ Graceful shutdown completed")
        sys.exit(0)
    
    def deploy_strategy(self, duration_hours: int = 8):
        """Deploy strategy for paper trading"""
        
        print("🏛️ INSTITUTIONAL COMBINED STRATEGY - PAPER TRADING DEPLOYMENT")
        print("="*80)
        print("🎯 READY FOR LIVE IBKR API PAPER TRADING")
        print("📊 Following backtest_guide.md and backtest_guide_phase2.md standards")
        print("⚠️ ENSURE IBKR TRADER WORKSTATION IS RUNNING WITH PAPER TRADING ACCOUNT")
        print("="*80)
        
        # Initialize strategy with institutional parameters
        config = StrategyConfig(
            initial_capital=100000,
            tier2_allocation=0.65,     # 65% in high-confidence instruments
            tier3_allocation=0.15,     # 15% in speculative instruments  
            cash_reserve=0.20,         # 20% cash buffer
            
            # Risk management (institutional grade)
            max_portfolio_drawdown=0.18,  # 18% kill switch
            daily_loss_limit=0.035,       # 3.5% daily limit
            single_position_limit=0.15,   # 15% max per instrument
            
            # Execution parameters
            slippage_factor=0.0008,       # 8bps realistic slippage
            commission_per_trade=1.0,     # $1 per trade
            rebalance_frequency="weekly", # Weekly rebalancing
            
            # IBKR connection
            ibkr_host="127.0.0.1",
            ibkr_port=7497,
            client_id=10
        )
        
        self.strategy = InstitutionalCombinedStrategy(config)
        
        # Display strategy information
        self._display_strategy_info(config)
        
        # Pre-deployment checks
        if not self._run_predeploy_checks():
            logger.error("❌ Pre-deployment checks failed. Aborting deployment.")
            return False
        
        # Auto-confirm deployment for testing
        print("\n🚀 Auto-deploying to paper trading...")
        print("✅ Deployment confirmed")
        
        # Start deployment
        logger.info("🚀 STARTING PAPER TRADING DEPLOYMENT")
        self.start_time = datetime.now()
        self.running = True
        
        try:
            # Run strategy
            self.strategy.run_strategy(duration_hours=duration_hours)
            
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.strategy:
                self.strategy.save_performance_report()
                self.strategy.disconnect_from_ibkr()
        
        logger.info("✅ Paper trading deployment completed")
        return True
    
    def _display_strategy_info(self, config: StrategyConfig):
        """Display detailed strategy information"""
        
        print(f"\n💰 CAPITAL ALLOCATION:")
        print(f"   Initial Capital: ${config.initial_capital:,}")
        print(f"   TIER 2 (High Confidence): ${config.initial_capital * config.tier2_allocation:,.0f} ({config.tier2_allocation:.0%})")
        print(f"   TIER 3 (Speculative): ${config.initial_capital * config.tier3_allocation:,.0f} ({config.tier3_allocation:.0%})")
        print(f"   Cash Reserve: ${config.initial_capital * config.cash_reserve:,.0f} ({config.cash_reserve:.0%})")
        
        print(f"\n🎯 INSTRUMENT ALLOCATION:")
        tier2_per_instrument = (config.tier2_allocation / 5) * 100  # 5 TIER2 instruments
        tier3_per_instrument = (config.tier3_allocation / 2) * 100  # 2 TIER3 instruments
        
        print("   TIER 2 (High Confidence p < 0.05):")
        tier2_instruments = [
            ("XAR", "Aerospace/Defense ETF", 0.0201),
            ("EWG", "Germany ETF", 0.0187), 
            ("XLF", "Financial Sector ETF", 0.0240),
            ("TSLA", "Tesla Inc", 0.0294),
            ("FXI", "China Large-Cap ETF", 0.0277)
        ]
        
        for symbol, description, p_value in tier2_instruments:
            allocation_usd = config.initial_capital * (config.tier2_allocation / 5)
            print(f"      {symbol}: ${allocation_usd:,.0f} ({tier2_per_instrument:.1f}%) | {description} | p={p_value:.4f}")
        
        print("   TIER 3 (Speculative p < 0.10):")
        tier3_instruments = [
            ("AVGO", "Broadcom Inc", 0.0649),
            ("NVDA", "NVIDIA Corp", 0.0722)
        ]
        
        for symbol, description, p_value in tier3_instruments:
            allocation_usd = config.initial_capital * (config.tier3_allocation / 2)
            print(f"      {symbol}: ${allocation_usd:,.0f} ({tier3_per_instrument:.1f}%) | {description} | p={p_value:.4f}")
        
        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"   Portfolio Drawdown Limit: {config.max_portfolio_drawdown:.0%} (Kill Switch)")
        print(f"   Daily Loss Limit: {config.daily_loss_limit:.1%}")
        print(f"   Single Position Limit: {config.single_position_limit:.0%}")
        print(f"   Rebalancing: {config.rebalance_frequency.title()}")
        
        print(f"\n⚙️ EXECUTION PARAMETERS:")
        print(f"   Expected Slippage: {config.slippage_factor * 10000:.1f} bps")
        print(f"   Commission per Trade: ${config.commission_per_trade:.2f}")
        print(f"   IBKR Connection: {config.ibkr_host}:{config.ibkr_port} (Client ID {config.client_id})")
    
    def _run_predeploy_checks(self) -> bool:
        """Run pre-deployment validation checks"""
        
        print(f"\n🔍 PRE-DEPLOYMENT VALIDATION CHECKS:")
        
        checks_passed = 0
        total_checks = 6
        
        # Check 1: IBKR Connection
        try:
            success = self.strategy.connect_to_ibkr()
            if success:
                print("   ✅ IBKR API Connection: PASSED")
                checks_passed += 1
                self.strategy.disconnect_from_ibkr()  # Disconnect for now
            else:
                print("   ❌ IBKR API Connection: FAILED")
        except Exception as e:
            print(f"   ❌ IBKR API Connection: FAILED ({e})")
        
        # Check 2: Market Data Access
        try:
            test_data = self.strategy.get_market_data('SPY', period='5d')
            if not test_data.empty and len(test_data) > 0:
                print("   ✅ Market Data Access: PASSED")
                checks_passed += 1
            else:
                print("   ❌ Market Data Access: FAILED (No data)")
        except Exception as e:
            print(f"   ❌ Market Data Access: FAILED ({e})")
        
        # Check 3: Signal Generation
        try:
            test_signals = self.strategy.generate_signals('XAR')
            if 'signal' in test_signals and 'confidence' in test_signals:
                print("   ✅ Signal Generation: PASSED")
                checks_passed += 1
            else:
                print("   ❌ Signal Generation: FAILED (Invalid signals)")
        except Exception as e:
            print(f"   ❌ Signal Generation: FAILED ({e})")
        
        # Check 4: Risk Management
        try:
            risk_status = self.strategy.check_risk_limits()
            if isinstance(risk_status, dict) and 'portfolio_drawdown_ok' in risk_status:
                print("   ✅ Risk Management: PASSED")
                checks_passed += 1
            else:
                print("   ❌ Risk Management: FAILED")
        except Exception as e:
            print(f"   ❌ Risk Management: FAILED ({e})")
        
        # Check 5: Portfolio Status
        try:
            status = self.strategy.get_portfolio_status()
            if status['portfolio_value'] == self.strategy.config.initial_capital:
                print("   ✅ Portfolio Status: PASSED")
                checks_passed += 1
            else:
                print("   ❌ Portfolio Status: FAILED")
        except Exception as e:
            print(f"   ❌ Portfolio Status: FAILED ({e})")
        
        # Check 6: Logging and Reporting
        try:
            logger.info("Test log message")
            print("   ✅ Logging System: PASSED")
            checks_passed += 1
        except Exception as e:
            print(f"   ❌ Logging System: FAILED ({e})")
        
        # Summary
        pass_rate = checks_passed / total_checks
        print(f"\n📊 PRE-DEPLOYMENT CHECK SUMMARY:")
        print(f"   Passed: {checks_passed}/{total_checks} ({pass_rate:.0%})")
        
        if pass_rate >= 0.8:  # 80% pass rate required
            print("   🎯 STATUS: READY FOR DEPLOYMENT")
            return True
        else:
            print("   ❌ STATUS: NOT READY - Fix issues before deployment")
            return False
    
    def monitor_deployment(self):
        """Monitor running deployment"""
        if not self.strategy or not self.running:
            print("❌ No active deployment to monitor")
            return
        
        print("\n📊 DEPLOYMENT MONITORING:")
        print("Press Ctrl+C to stop monitoring and shutdown strategy")
        
        try:
            while self.running:
                # Get current status
                status = self.strategy.get_portfolio_status()
                
                # Display key metrics
                elapsed = datetime.now() - self.start_time
                print(f"\n⏱️  Runtime: {elapsed}")
                print(f"💰 Portfolio Value: ${status['portfolio_value']:,.2f}")
                print(f"📈 Total Return: {status['total_return_pct']:+.2f}%")
                print(f"📉 Max Drawdown: {status['max_drawdown']:.2f}%")
                print(f"💵 Cash: ${status['cash']:,.2f}")
                
                # Show positions
                if status['positions']:
                    print("🎯 Current Positions:")
                    for symbol, pos_data in status['positions'].items():
                        print(f"   {symbol}: {pos_data['shares']:,} shares @ ${pos_data['price']:.2f} = ${pos_data['value']:,.0f} ({pos_data['allocation_pct']:.1f}%)")
                
                # Wait 30 seconds
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

def main():
    """Main deployment function"""
    
    # Parse command line arguments (simple version)
    duration_hours = 8  # Default 8 hours
    if len(sys.argv) > 1:
        try:
            duration_hours = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid duration. Using default 8 hours.")
    
    print("🏛️ INSTITUTIONAL COMBINED STRATEGY DEPLOYMENT")
    print("="*60)
    print("⚠️  IMPORTANT PREREQUISITES:")
    print("   1. IBKR Trader Workstation must be running")
    print("   2. Paper Trading account must be active")
    print("   3. API connections must be enabled")
    print("   4. Port 7497 must be available")
    print("="*60)
    
    # Initialize deployment
    deployment = PaperTradingDeployment()
    
    # Deploy strategy
    success = deployment.deploy_strategy(duration_hours=duration_hours)
    
    if success:
        print("✅ Deployment completed successfully")
        return 0
    else:
        print("❌ Deployment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())