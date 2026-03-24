"""
DESCARGA MANUAL DE DATOS
Corre esto primero si run.py falla con rate limit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import time
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","XOM","JNJ","WMT","PG"]
START   = "2018-01-01"
END     = "2023-12-31"

print(f"Descargando {len(TICKERS)} tickers...")
print(f"Periodo: {START} -> {END}\n")

ok = 0
for ticker in TICKERS:
    path = Path(f"data/raw/{ticker}.parquet")
    if path.exists():
        print(f"  [SKIP] {ticker} ya existe")
        ok += 1
        continue
    print(f"  Descargando {ticker}...", end=" ")
    try:
        df = yf.download(ticker, start=START, end=END,
                         auto_adjust=True, progress=False)
        if df.empty:
            print("sin datos")
            time.sleep(5)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["ticker"] = ticker
        df.to_parquet(path, compression="snappy")
        print(f"OK ({len(df)} filas)")
        ok += 1
        time.sleep(3)
    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(10)

print(f"\nResultado: {ok}/{len(TICKERS)} descargados")
print("Ahora corre: python run.py demo")
