---

# Trading Bot — Automated Dip-Buy + Take-Profit Strategy

### Works with **IBKR** (Interactive Brokers) & **Zerodha (India)**

Runs automatically every **x minutes** using Docker.

---

## ⭐ Overview

This project runs the **B1.2 algorithmic trading strategy**:

* Tracks each stock’s **All-Time High (ATH)**
* Buys dips automatically at **10% / 15% / 20%** drops
* Takes profit at **20% / 40% / 60%** above ATH
* Includes **low-marker heavy buys** for deep crashes
* Displays clean tables, events, buy/sell logs

The bot is designed to:

✔ Run automatically every 2 minutes

✔ Connect to **IBKR** OR **Zerodha**

✔ Place real trades **only if enabled**

✔ Print logs + save CSV output

---

# Requirements

* Python 3.10+
* Docker (optional but recommended)
* IBKR account OR Zerodha account
* API credentials from your broker

---

# Setup (Local Installation)

```bash
git clone https://github.com/yourname/b1.2-trading-bot.git
cd b1.2-trading-bot

pip install -r requirements.txt
```

Run manually:

```bash
python trading_bot.py
```

---

# Run Using Docker (Recommended)

### Build image:

```bash
docker build -t b12bot .
```

### Run container (checks every 2 minutes):

```bash
docker run -d \
  --name b12trader \
  -e BROKER=IBKR \
  -e IB_HOST=host.docker.internal \
  -e IB_PORT=7497 \
  -e IB_CLIENT_ID=101 \
  -e ZERODHA_API_KEY= \
  -e ZERODHA_API_SECRET= \
  -e ZERODHA_REQUEST_TOKEN= \
  b12bot
```

---

# Choosing Your Broker

Set in environment variable:

```
BROKER=IBKR
```

or

```
BROKER=ZERODHA
```

The bot automatically uses the correct connector.

---

# IBKR Setup (Interactive Brokers)

### Step 1 — Install IB Gateway or TWS

Download from:
[https://www.interactivebrokers.com/en/index.php?f=16040](https://www.interactivebrokers.com/en/index.php?f=16040)

Use **Paper Trading** for safety.

---

### Step 2 — Enable API Access

In IB Gateway/TWS:

```
Settings → API → Settings
✔ Enable ActiveX and Socket Clients
Trusted IPs: 127.0.0.1
Socket Port: 7497 (paper) / 7496 (live)
```

---

### Step 3 — Environment variables

```
BROKER=IBKR
IB_HOST=host.docker.internal
IB_PORT=7497
IB_CLIENT_ID=101
```

---

### Step 4 — Run

```bash
docker run -d --name b12trader \
  -e BROKER=IBKR \
  -e IB_HOST=host.docker.internal \
  -e IB_PORT=7497 \
  -e IB_CLIENT_ID=101 \
  b12bot
```

---

# Zerodha Setup (India)

You need a Zerodha **Kite Connect API** subscription:
[https://kite.trade](https://kite.trade)

---

### Step 1 — Get API Key + Secret

From your Kite developer dashboard.

---

### Step 2 — Get Request Token

Login using:

```
https://kite.trade/connect/login?api_key=YOUR_API_KEY
```

After login, Zerodha redirects to your redirect URL with:

```
request_token=XYZ
```

---

### Step 3 — Put in environment:

```
BROKER=ZERODHA
ZERODHA_API_KEY=xxxx
ZERODHA_API_SECRET=xxxx
ZERODHA_REQUEST_TOKEN=xxxx
```

The bot automatically generates an **access token** and stores it locally.

---

### Step 4 — Run

```bash
docker run -d --name b12trader \
  -e BROKER=ZERODHA \
  -e ZERODHA_API_KEY=xxx \
  -e ZERODHA_API_SECRET=yyy \
  -e ZERODHA_REQUEST_TOKEN=zzz \
  b12bot
```

---

# Auto-Run Every x Minutes

The Docker container has a built-in loop:

```python
while True:
    run_strategy()
    time.sleep(120)   # 2 minutes
```

It only trades during:

* US market hours for US stocks
* Indian market hours for Zerodha

Outside these hours it only prints logs.

---

# Dry-Run Mode (Safe Mode)

To test without placing real trades:

```
DRY_RUN=1
```

Example:

```bash
docker run -d \
  -e BROKER=IBKR \
  -e DRY_RUN=1 \
  b12bot
```

Bot will print buy/sell signals but **won’t execute** orders.

---

# Logs & Output

The bot saves:

```
strategy_results_b1_2.csv
trade_logs_b1_2.csv
low_marker_counts_b1_2.csv
```

All inside `/logs` directory.

---

# How It Works Internally

The strategy does:

1. Fetch live price from broker
2. Check if new ATH created
3. Check dip % from ATH
4. If dip matches 10/15/20% → BUY
5. If price hits 1.2/1.4/1.6× ATH → SELL
6. If deep crash with high volatility → HEAVY BUY
7. Logs all events

Everything is automated once running.

---

# Safety Notes

✔ Always start with **DRY_RUN=1**
✔ Use **IBKR Paper Trading** before real trading
✔ Zerodha charges small per-order fees
✔ This bot is NOT HFT — checks only every 2 minutes
✔ You can stop at any time:

```bash
docker stop b12trader
```

---

# Support

Coming soon...

* Version with Telegram alerts
* Version with email notifications
* Version that trades options
* Version with machine-learning filters


