# Data Preloading and Cache Management

This document explains how to pre-download market data to speed up analyses.

## Why Preload Data?

Downloading 5 years of stock data from S3 (1200+ files) can take 10-20 minutes during an analysis. By pre-loading data once, subsequent analyses are instant.

## Cache Location

### Inside Container
- Path: `/tmp/crypto_data_cache/`
- Mounted from persistent Docker volume

### On Host Machine
- Path: `/var/lib/docker/volumes/temporal-trading-agents_market_data_cache/_data`
- Requires sudo to access: `sudo ls -lh /var/lib/docker/volumes/temporal-trading-agents_market_data_cache/_data`

### Cache Format
Data is stored as pickle files with naming pattern:
```
{SYMBOL}_{PERIOD}_{INTERVAL}.pkl
```

Examples:
- `BTC-USD_2y_1d.pkl` - Bitcoin, 2 years, daily
- `AAPL_5y_1d.pkl` - Apple, 5 years, daily
- `TSLA_5y_1d.pkl` - Tesla, 5 years, daily

## Preloading Data

### Method 1: Single Symbol

```bash
docker exec temporal-trading-backend python3 /app/scripts/preload_market_data.py AAPL
```

This will:
- Auto-detect asset type (stock vs crypto)
- Use 5 years for stocks, 2 years for crypto
- Download and cache the data

### Method 2: Multiple Symbols

```bash
docker exec temporal-trading-backend python3 /app/scripts/preload_market_data.py \
  BTC-USD ETH-USD AAPL TSLA MSFT GOOGL
```

### Method 3: Batch Preload Popular Symbols

```bash
docker exec temporal-trading-backend bash /app/scripts/preload_popular_symbols.sh
```

This preloads:
- **Crypto (2 years):** BTC-USD, ETH-USD, SOL-USD, XRP-USD, ADA-USD, DOGE-USD, AVAX-USD, DOT-USD
- **Stocks (5 years):** AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, AMD, NFLX, DIS

### Custom Period

```bash
docker exec temporal-trading-backend python3 /app/scripts/preload_market_data.py AAPL --period 2y
```

## Data Periods by Asset Type

The system automatically uses appropriate periods:

| Asset Type | Default Period | Trading Days | S3 Files |
|------------|---------------|--------------|----------|
| Crypto     | 2 years       | ~730         | ~500     |
| Stocks     | 5 years       | ~1,260       | ~1,250   |

## Cache Persistence

The cache is stored in a Docker volume (`market_data_cache`) which persists across:
- Container restarts
- Container rebuilds
- System reboots

The cache will **NOT** persist if you:
- Delete the Docker volume: `docker volume rm temporal-trading-agents_market_data_cache`
- Run `docker compose down -v` (removes volumes)

## Cache Management

### View Cache Contents

```bash
docker exec temporal-trading-backend ls -lh /tmp/crypto_data_cache/
```

### Clear Entire Cache

```bash
docker exec temporal-trading-backend rm -rf /tmp/crypto_data_cache/*
```

### Clear Specific Symbol

```bash
docker exec temporal-trading-backend rm /tmp/crypto_data_cache/AAPL_5y_1d.pkl
```

### Check Cache Size

```bash
docker exec temporal-trading-backend du -sh /tmp/crypto_data_cache/
```

## Performance Comparison

| Scenario | First Run (No Cache) | Subsequent Runs (Cached) |
|----------|---------------------|--------------------------|
| BTC-USD (2y) | ~3-5 minutes | <1 second |
| AAPL (5y) | ~10-20 minutes | <1 second |

## Recommended Workflow

1. **One-time setup:** Run batch preload for symbols you'll analyze frequently
   ```bash
   docker exec temporal-trading-backend bash /app/scripts/preload_popular_symbols.sh
   ```

2. **Add new symbols as needed:**
   ```bash
   docker exec temporal-trading-backend python3 /app/scripts/preload_market_data.py NVDA
   ```

3. **Monitor cache:**
   ```bash
   docker exec temporal-trading-backend ls -lh /tmp/crypto_data_cache/
   ```

## Troubleshooting

### Cache not persisting after restart

Check if volume is properly mounted:
```bash
docker inspect temporal-trading-backend | grep -A 10 Mounts
```

Should show:
```json
"Destination": "/tmp/crypto_data_cache",
"Source": "/var/lib/docker/volumes/temporal-trading-agents_market_data_cache/_data"
```

### Out of disk space

Check volume size:
```bash
docker system df -v | grep market_data_cache
```

Clear old/unused symbols to free space.

### Corrupted cache

If you get errors about corrupted pickle files:
```bash
docker exec temporal-trading-backend rm -rf /tmp/crypto_data_cache/*
```

Then re-run the preload script.
