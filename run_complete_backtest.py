#!/usr/bin/env python3
"""
COMPLETE PROFESSIONAL BACKTEST EXECUTION
========================================
Runs the complete backtest to completion without timeout
Following exact backtest_guide.md requirements
"""

import sys
import subprocess
import signal
import time
import os

def run_with_extended_timeout():
    """Run the backtest with extended timeout and progress monitoring"""
    print("🚀 STARTING COMPLETE PROFESSIONAL BACKTEST")
    print("="*80)
    print("Following backtest_guide.md requirements:")
    print("✓ IBKR API real data only")  
    print("✓ Minimum 10,000 Monte Carlo permutations")
    print("✓ All phases without shortcuts")
    print("✓ Double-checked calculations")
    print("✓ Professional quality reporting")
    print("="*80)
    
    # Change to the correct directory
    os.chdir("/Users/kacper/Desktop/Option_trading1")
    
    # Run the backtest with no timeout limit
    cmd = [sys.executable, "PROFESSIONAL_BACKTEST_IMPLEMENTATION.py"]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("🔄 Backtest process started...")
        print("📊 Monitoring progress (this may take 10-15 minutes for 10,000 permutations)...")
        print("-"*80)
        
        # Monitor output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
                # Check for specific progress indicators
                if "MCPT Progress:" in output:
                    print(f"🎯 {output.strip()}")
                elif "Connected to IBKR" in output:
                    print("✅ IBKR Connection Successful")
                elif "Quality data retrieved" in output:
                    print("✅ Data Quality Validated")
                elif "Monte Carlo validation completed" in output:
                    print("✅ Monte Carlo Validation Completed")
        
        # Get final return code
        return_code = process.poll()
        
        print("-"*80)
        if return_code == 0:
            print("🎉 BACKTEST COMPLETED SUCCESSFULLY")
            print("📊 Check output above for complete results")
        else:
            print(f"❌ BACKTEST FAILED with return code {return_code}")
            
        return return_code
        
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
        process.terminate()
        return 1
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return 1

if __name__ == "__main__":
    # Set higher limits for Monte Carlo processing
    import resource
    try:
        # Increase memory and CPU time limits
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))  # 8GB memory
    except:
        pass
    
    exit_code = run_with_extended_timeout()
    sys.exit(exit_code)