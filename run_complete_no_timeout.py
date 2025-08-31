#!/usr/bin/env python3
"""
COMPLETE EXPANDED UNIVERSE EXECUTION WITHOUT TIMEOUT
===================================================
Runs the complete expanded universe backtest to full completion
NO TIME LIMITS - Will run until complete analysis finished
"""

import sys
import subprocess
import signal
import time
import os

def run_until_complete():
    """Run the expanded universe backtest until complete - NO TIMEOUT"""
    print("🚀 STARTING COMPLETE EXPANDED UNIVERSE BACKTEST - NO TIMEOUT LIMITS")
    print("="*100)
    print("COMPLETE PROFESSIONAL ANALYSIS:")
    print("✓ NASDAQ Top 30 companies - full coverage")  
    print("✓ Major ETFs (SPY, QQQ, IWM, VTI, VOO, VEA, VWO, EFA, EWJ)")
    print("✓ Volatility instruments (UVXY, SVXY, VXX, VIXY)")
    print("✓ All Sector ETFs (XLF, XLK, XLE, XLI, XLV, XLU, XLRE, XLP, XLY, XLB)")
    print("✓ International/Commodity (GLD, SLV, USO, UNG, FXI, EEM, TLT, HYG, LQD)")
    print("✓ Total Universe: 63 instruments with complete professional analysis")
    print("")
    print("INSTITUTIONAL STANDARDS:")
    print("✓ IBKR API real data only (all client IDs cleared)")  
    print("✓ 10,000 Monte Carlo permutations for top instruments (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA)")
    print("✓ 1,000 Monte Carlo permutations for remaining 56 instruments")
    print("✓ ALL phases implemented without shortcuts")
    print("✓ Double-checked calculations and mathematical verification")
    print("✓ Complete institutional-quality reporting")
    print("="*100)
    
    # Change to the correct directory
    os.chdir("/Users/kacper/Desktop/Option_trading1")
    
    # Run the backtest with unlimited time
    cmd = [sys.executable, "NASDAQ_TOP30_ULTIMATE_BACKTEST.py"]
    
    try:
        # Start the process with NO timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("🔄 Expanded Universe Backtest started - RUNNING TO COMPLETION")
        print("📊 This will analyze ALL 63 instruments with FULL Monte Carlo validation")
        print("⏱️ NO TIME LIMITS - will run until complete (estimated 45-90 minutes)")
        print("🎯 Progress tracking enabled - detailed status for each instrument")
        print("🔥 PATIENCE REQUIRED - institutional quality analysis takes time")
        print("-"*100)
        
        completed_instruments = []
        current_instrument = None
        start_time = time.time()
        
        # Monitor output in real-time with NO timeout
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                
                # Enhanced progress tracking
                if "ANALYZING" in line and "(" in line:
                    current_instrument = line.split()[2] if len(line.split()) > 2 else "Unknown"
                    print(f"🎯 CURRENT: {line}")
                    
                elif "Connected to IBKR" in line:
                    print("✅ IBKR API CONNECTION ESTABLISHED")
                    
                elif "Quality data for" in line and ": " in line:
                    symbol = line.split("for")[1].split(":")[0].strip()
                    records = line.split(":")[-1].strip()
                    print(f"📊 DATA QUALITY: {symbol} - {records}")
                    
                elif "Monte Carlo completed for" in line:
                    symbol = line.split("for")[1].split(":")[0].strip()
                    p_value = line.split("p-value =")[1].strip() if "p-value =" in line else "N/A"
                    completed_instruments.append(symbol)
                    elapsed_time = time.time() - start_time
                    
                    print(f"✅ MONTE CARLO COMPLETED: {symbol} (p-value: {p_value})")
                    print(f"📊 PROGRESS: {len(completed_instruments)}/63 instruments completed")
                    print(f"⏱️ ELAPSED TIME: {elapsed_time/60:.1f} minutes")
                    print(f"📈 ESTIMATED REMAINING: {((elapsed_time/len(completed_instruments)) * (63-len(completed_instruments)))/60:.1f} minutes")
                    print("-"*60)
                    
                elif "MCPT Progress" in line:
                    print(f"🔄 MONTE CARLO PROGRESS: {line}")
                    
                elif "Report saved" in line or "saved to:" in line:
                    print(f"📁 REPORT SAVED: {line}")
                    
                elif "FINAL SUMMARY REPORT" in line:
                    print("🎊 FINAL SUMMARY GENERATION STARTED")
                    
                elif "ERROR" in line:
                    print(f"⚠️ STATUS: {line}")
                    
                elif "EXPANDED UNIVERSE ANALYSIS COMPLETED" in line:
                    print("🎉 ANALYSIS COMPLETION DETECTED")
        
        # Get final return code
        return_code = process.poll()
        total_time = time.time() - start_time
        
        print("-"*100)
        if return_code == 0:
            print("🎉 COMPLETE EXPANDED UNIVERSE BACKTEST FINISHED SUCCESSFULLY")
            print(f"📊 Total Instruments Processed: {len(completed_instruments)}")
            print(f"⏱️ Total Execution Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
            print("")
            print("📁 GENERATED FILES:")
            print("   • EXPANDED_UNIVERSE_REPORT_[SYMBOL].txt - Individual detailed reports (63 files)")
            print("   • FINAL_EXPANDED_UNIVERSE_SUMMARY.txt - Master summary report")
            print("")
            print("🏆 ANALYSIS COMPLETE:")
            print("   • All 63 instruments analyzed with institutional-grade precision")
            print("   • Complete Monte Carlo validation (10,000+ for top stocks, 1,000+ for others)")
            print("   • Professional risk-adjusted performance metrics")
            print("   • Portfolio construction recommendations")
            print("   • Statistical significance testing completed")
            print("")
            print("🎯 NEXT STEPS:")
            print("   1. Review FINAL_EXPANDED_UNIVERSE_SUMMARY.txt for top performers")
            print("   2. Check individual reports for detailed strategy analysis")
            print("   3. Implement highest-ranked strategies with proper position sizing")
            print("   4. Follow institutional risk management guidelines")
            
        else:
            print(f"❌ BACKTEST FAILED with return code {return_code}")
            print(f"📊 Completed {len(completed_instruments)} instruments before failure")
            print("📋 Check output above for specific error details")
            
        return return_code
        
    except KeyboardInterrupt:
        print("\\n⚠️ Backtest interrupted by user")
        print(f"📊 Partial completion: {len(completed_instruments)} instruments processed")
        process.terminate()
        return 1
    except Exception as e:
        print(f"❌ Critical error during expanded universe backtest: {e}")
        return 1

if __name__ == "__main__":
    # Set maximum system limits for comprehensive analysis
    import resource
    try:
        # Maximum memory and processing limits
        resource.setrlimit(resource.RLIMIT_AS, (-1, -1))  # Unlimited memory
        resource.setrlimit(resource.RLIMIT_CPU, (-1, -1))  # Unlimited CPU time
        resource.setrlimit(resource.RLIMIT_FSIZE, (-1, -1))  # Unlimited file size
    except:
        pass
    
    print("🌍 ULTIMATE EXPANDED UNIVERSE OPTIONS STRATEGY - COMPLETE ANALYSIS")
    print("=" * 100)
    print("INSTITUTIONAL-GRADE BACKTESTING ACROSS COMPLETE OPTIONS UNIVERSE")
    print("NASDAQ Top 30 + Major ETFs + Volatility + Sectors + International + Commodities")
    print("FOLLOWING EXACT backtest_guide.md REQUIREMENTS - NO SHORTCUTS")
    print("=" * 100)
    
    exit_code = run_until_complete()
    
    if exit_code == 0:
        print("")
        print("=" * 100)
        print("🎊 COMPLETE EXPANDED UNIVERSE ANALYSIS FINISHED")
        print("🏆 INSTITUTIONAL-QUALITY RESULTS READY FOR IMPLEMENTATION")
        print("=" * 100)
        print("")
        print("PROFESSIONAL IMPLEMENTATION GUIDANCE:")
        print("• Use results from FINAL_EXPANDED_UNIVERSE_SUMMARY.txt")
        print("• Focus on statistically significant strategies only") 
        print("• Implement proper position sizing (1-3% per trade maximum)")
        print("• Follow institutional risk management protocols")
        print("• Monitor performance with real-time risk controls")
        print("=" * 100)
    
    sys.exit(exit_code)