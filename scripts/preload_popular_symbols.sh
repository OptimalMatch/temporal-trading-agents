#!/bin/bash
# Pre-load popular crypto and stock symbols into the cache

echo "=================================================================="
echo "PRE-LOADING POPULAR MARKET DATA"
echo "=================================================================="
echo ""
echo "This will download and cache data for commonly analyzed symbols."
echo "This may take 15-30 minutes but only needs to be done once."
echo ""

# Popular cryptocurrencies (2 years)
CRYPTO_SYMBOLS=(
    "BTC-USD"
    "ETH-USD"
    "SOL-USD"
    "XRP-USD"
    "ADA-USD"
    "DOGE-USD"
    "AVAX-USD"
    "DOT-USD"
)

# Popular US stocks (5 years)
STOCK_SYMBOLS=(
    "AAPL"
    "MSFT"
    "GOOGL"
    "AMZN"
    "TSLA"
    "NVDA"
    "META"
    "AMD"
    "NFLX"
    "DIS"
)

# Run the preload script
python3 /app/scripts/preload_market_data.py "${CRYPTO_SYMBOLS[@]}" "${STOCK_SYMBOLS[@]}"

echo ""
echo "=================================================================="
echo "PRE-LOAD COMPLETE"
echo "=================================================================="
echo ""
echo "You can now run analyses on these symbols with instant data access."
