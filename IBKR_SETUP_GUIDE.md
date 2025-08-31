# IBKR API Setup Guide for MEGA-SCALE Backtesting

## 🚀 Quick Setup Steps

### 1. Install IBKR Software
Choose ONE of these options:

#### Option A: Trader Workstation (TWS) - Full Platform
- Download from: https://www.interactivebrokers.com/en/trading/tws.php
- Full desktop trading platform with charts and tools

#### Option B: IB Gateway - API Only (Recommended for backtesting)
- Download from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php  
- Lightweight, API-focused application
- Less memory usage, better for backtesting

### 2. Configure API Settings

#### For Trader Workstation (TWS):
1. Start TWS and login to your account
2. Go to: **File > Global Configuration > API > Settings**
3. Enable these settings:
   - ✅ **Enable ActiveX and Socket Clients** 
   - ✅ **Allow connections from localhost only**
   - ✅ **Read-Only API** (for safety during backtesting)
4. Set **Socket Port**:
   - **7497** for Paper Trading (recommended for backtesting)
   - **7496** for Live Trading (be careful!)
5. **Trusted IPs**: Add `127.0.0.1`
6. **Master API Client ID**: Leave empty or set to `999`
7. Click **OK** and **restart TWS**

#### For IB Gateway:
1. Start IB Gateway and login
2. The configuration dialog appears automatically
3. Set **Socket Port**:
   - **4002** for Paper Trading (recommended)
   - **4001** for Live Trading  
4. ✅ **Read-Only API**
5. Click **OK**

### 3. Account Requirements

#### Paper Trading (Recommended for backtesting):
- Free with any IBKR account
- No real money at risk
- Perfect for backtesting and development
- Access to real market data (delayed 15-20 minutes)

#### Live Trading:
- Requires funded account with minimum balance
- Real money and commissions
- Real-time market data (subscription fees apply)
- **⚠️ NOT recommended for backtesting**

### 4. Test Connection

Run this quick test to verify your setup:

```python
from ib_insync import *
import asyncio

# Test connection
ib = IB()
try:
    # For TWS Paper Trading
    ib.connect('127.0.0.1', 7497, clientId=1)
    print("✅ Connected to TWS Paper Trading")
    
    # OR for IB Gateway Paper Trading  
    # ib.connect('127.0.0.1', 4002, clientId=1)
    # print("✅ Connected to IB Gateway Paper Trading")
    
    # Test basic functionality
    positions = ib.positions()
    print(f"📊 Account positions: {len(positions)}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
finally:
    ib.disconnect()
```

### 5. Run MEGA-SCALE Backtest

Once connected, run:
```bash
python NASDAQ100_SP500_MEGA_BACKTEST.py
```

The program will automatically:
- Try multiple ports (7497, 7496, 4002, 4001)
- Test different client IDs
- Provide detailed troubleshooting if connection fails

## 🔧 Troubleshooting

### Connection Timeout Error
```
TimeoutError(60, "Connect call failed ('127.0.0.1', 7497)")
```

**Solutions:**
1. **Check if TWS/Gateway is running** and logged in
2. **Verify API settings** are enabled in configuration  
3. **Restart TWS/Gateway** after changing settings
4. **Check firewall** is not blocking the connection
5. **Try different client ID** (program does this automatically)

### "Socket port in use" Error
- Another application is using the API connection
- Close other trading applications or change client ID

### Market Data Issues
- Paper trading has delayed data (15-20 minutes)
- For real-time data, paid market data subscriptions required
- Program automatically handles different data types

### Permission Denied
- Check "Enable ActiveX and Socket Clients" is enabled
- Verify "127.0.0.1" is in Trusted IPs
- Try "Read-Only API" mode for backtesting

## 📊 Data Quality Comparison

| Source | Update Frequency | Historical Data | Cost | Best For |
|--------|------------------|-----------------|------|----------|
| **IBKR API** | Real-time/15min delayed | 15+ years | Free* | Professional backtesting |
| **yfinance** | Daily close | 5+ years | Free | Quick analysis |

*Free with IBKR account, paid market data subscriptions for real-time

## 🎯 Recommended Setup for MEGA-SCALE Backtesting

1. **Use IB Gateway** (lighter than TWS)
2. **Paper Trading Account** (port 4002)  
3. **Read-Only API** enabled
4. **15+ years historical data** per symbol
5. **All 254 symbols** in universe processed

This setup provides institutional-quality data for mega-scale backtesting while maintaining safety through paper trading.