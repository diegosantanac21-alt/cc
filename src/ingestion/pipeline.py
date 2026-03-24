"""
INGESTION DE DATOS
Download -> Clean -> Validate -> Store (Parquet)
data/raw/ es INMUTABLE — nunca se sobreescribe.
"""
import os
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

try:
    import polars as pl
    HAS_PL = True
except ImportError:
    HAS_PL = False

from config.settings import DATA_RAW, DATA_PROC, DATA_FEAT, UNIVERSE_CFG, DATA_CFG

logger = logging.getLogger("ingestion")


class FrameEngine:

    def __init__(self, prefer_polars=True):
        self.prefer_polars = prefer_polars and HAS_PL

    def read_parquet(self, path):
        if self.prefer_polars:
            df = pl.read_parquet(path)
            if "Date" in df.columns:
                return df.to_pandas().set_index("Date")
            return df.to_pandas()
        return pd.read_parquet(path)

    def clean_ohlcv(self, df):
        if not self.prefer_polars:
            return df
        reset_df = df.reset_index(names="Date") if df.index.name else df.reset_index()
        if "index" in reset_df.columns and "Date" not in reset_df.columns:
            reset_df = reset_df.rename(columns={"index": "Date"})
        pl_df = pl.from_pandas(reset_df)
        fill_cols = [c for c in ["Open", "High", "Low", "Close"] if c in pl_df.columns]
        if fill_cols:
            pl_df = pl_df.with_columns([pl.col(c).forward_fill(limit=5).alias(c) for c in fill_cols])
        if "Volume" in pl_df.columns:
            pl_df = pl_df.with_columns(pl.col("Volume").fill_null(0))
        if "Close" in pl_df.columns:
            pl_df = pl_df.with_columns([
                (pl.col("Close").log() - pl.col("Close").shift(1).log()).alias("log_ret"),
                pl.col("Close").pct_change().alias("simple_ret"),
            ])
        if "Close" in pl_df.columns and "Volume" in pl_df.columns:
            pl_df = pl_df.with_columns((pl.col("Close") * pl.col("Volume")).alias("dollar_vol"))
        return pl_df.to_pandas().set_index("Date")


class OHLCVDownloader:

    def __init__(self, raw_dir=DATA_RAW, rate_limit=3.0, max_retries=3, provider=None):
        self.raw_dir     = raw_dir
        self.rate_limit  = rate_limit
        self.max_retries = max_retries
        self.provider    = (provider or DATA_CFG.provider).lower()
        self.frame_engine = FrameEngine()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._alpaca_data_client = None
        self._alpaca_trading_client = None

    def _normalize_provider(self, provider):
        provider = (provider or self.provider).lower()
        aliases = {
            "yf": "yfinance",
            "yahoo": "yfinance",
            "alpaca": "alpaca",
        }
        return aliases.get(provider, provider)

    def _get_alpaca_data_client(self):
        if not HAS_ALPACA:
            raise ImportError("alpaca-py no instalado. Corre: pip install alpaca-py")
        if self._alpaca_data_client is None:
            api_key = DATA_CFG.alpaca_api_key or os.getenv("APCA_API_KEY_ID", "")
            api_secret = DATA_CFG.alpaca_api_secret or os.getenv("APCA_API_SECRET_KEY", "")
            if not api_key or not api_secret:
                raise ValueError("Faltan credenciales de Alpaca. Define ALPACA_API_KEY y ALPACA_API_SECRET.")
            self._alpaca_data_client = StockHistoricalDataClient(api_key, api_secret)
        return self._alpaca_data_client

    def _get_alpaca_trading_client(self):
        if not HAS_ALPACA:
            raise ImportError("alpaca-py no instalado. Corre: pip install alpaca-py")
        if self._alpaca_trading_client is None:
            api_key = DATA_CFG.alpaca_api_key or os.getenv("APCA_API_KEY_ID", "")
            api_secret = DATA_CFG.alpaca_api_secret or os.getenv("APCA_API_SECRET_KEY", "")
            if not api_key or not api_secret:
                raise ValueError("Faltan credenciales de Alpaca. Define ALPACA_API_KEY y ALPACA_API_SECRET.")
            self._alpaca_trading_client = TradingClient(api_key, api_secret, paper=True)
        return self._alpaca_trading_client

    def _download_with_yfinance(self, ticker, start, end):
        if not HAS_YF:
            raise ImportError("yfinance no instalado. Corre: pip install yfinance")
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
        if raw.empty:
            raise ValueError(f"Sin datos para {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        raw["ticker"] = ticker
        return raw

    def _download_with_alpaca(self, ticker, start, end):
        client = self._get_alpaca_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime(),
            adjustment=DATA_CFG.alpaca_adjustment,
            feed=DATA_CFG.alpaca_feed,
        )
        raw = client.get_stock_bars(req).df
        if raw.empty:
            raise ValueError(f"Sin datos para {ticker}")
        raw = raw.reset_index()
        if "symbol" in raw.columns:
            raw = raw.rename(columns={"symbol": "ticker"})
        else:
            raw["ticker"] = ticker
        if "timestamp" in raw.columns:
            raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
            raw = raw.set_index("timestamp")
        raw = raw.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "trade_count", "vwap", "ticker"] if c in raw.columns]
        raw = raw[keep_cols].dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in raw.columns])
        raw.index = pd.to_datetime(raw.index)
        return raw

    def resolve_instruments(
        self,
        tickers=None,
        source=None,
        asset_class=None,
        tradable_only=None,
        fractionable_only=None,
        limit=None,
    ):
        if tickers:
            return list(dict.fromkeys(tickers))

        source = (source or DATA_CFG.universe_source).lower()
        if source != "alpaca":
            return list(dict.fromkeys(UNIVERSE_CFG.tickers))

        client = self._get_alpaca_trading_client()
        asset_class = (asset_class or DATA_CFG.alpaca_asset_class).lower()
        tradable_only = DATA_CFG.alpaca_tradable_only if tradable_only is None else tradable_only
        fractionable_only = DATA_CFG.alpaca_fractionable_only if fractionable_only is None else fractionable_only
        limit = DATA_CFG.alpaca_limit if limit is None else limit

        asset_class_enum = {
            "us_equity": AssetClass.US_EQUITY,
            "crypto": AssetClass.CRYPTO,
        }.get(asset_class, AssetClass.US_EQUITY)

        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=asset_class_enum)
        assets = client.get_all_assets(req)
        symbols = []
        for asset in assets:
            if tradable_only and not getattr(asset, "tradable", False):
                continue
            if fractionable_only and not getattr(asset, "fractionable", False):
                continue
            symbol = getattr(asset, "symbol", None)
            if symbol:
                symbols.append(symbol)
            if limit and len(symbols) >= limit:
                break
        if not symbols:
            raise ValueError("Alpaca no devolvio instrumentos validos para el filtro actual")
        logger.info(f"Instrumentos cargados desde Alpaca: {len(symbols)}")
        return symbols

    def download_ticker(self, ticker, start, end):
        provider = self._normalize_provider(self.provider)
        for attempt in range(self.max_retries):
            try:
                if provider == "alpaca":
                    raw = self._download_with_alpaca(ticker, start, end)
                else:
                    raw = self._download_with_yfinance(ticker, start, end)
                time.sleep(self.rate_limit)
                logger.info(f"[OK] {ticker}: {len(raw)} filas ({provider})")
                return raw
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"[RETRY {attempt+1}] {ticker}: {exc} — esperar {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Descarga fallida: {ticker}")

    def save_raw(self, df, ticker):
        path = self.raw_dir / f"{ticker}.parquet"
        if path.exists():
            logger.info(f"[SKIP] {ticker} ya existe")
            return path
        df.to_parquet(path, compression="snappy")
        cs = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        logger.info(f"[SAVED] {ticker} — {len(df)} filas | sha={cs}")
        return path

    def load_raw(self, ticker):
        path = self.raw_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No hay datos raw para {ticker}")
        return self.frame_engine.read_parquet(path)

    def download_universe(self, tickers=None, start=None, end=None, source=None):
        tickers = self.resolve_instruments(tickers=tickers, source=source)
        start   = start   or UNIVERSE_CFG.start_date
        end     = end     or UNIVERSE_CFG.end_date
        results = {}
        logger.info(f"Descargando {len(tickers)} tickers...")
        for i, t in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] {t}")
            try:
                df = self.download_ticker(t, start, end)
                self.save_raw(df, t)
                results[t] = "ok"
            except Exception as e:
                results[t] = f"error: {e}"
                logger.error(f"[FAIL] {t}: {e}")
        ok = sum(1 for v in results.values() if v == "ok")
        logger.info(f"Descarga: {ok}/{len(tickers)} OK")
        return results


class DataCleaner:

    def __init__(self, prefer_polars=True):
        self.frame_engine = FrameEngine(prefer_polars=prefer_polars)

    def detect_outliers(self, df, z_thresh=5.0):
        ret      = df["Close"].pct_change()
        roll_mu  = ret.rolling(252, min_periods=60).mean()
        roll_std = ret.rolling(252, min_periods=60).std()
        z        = (ret - roll_mu) / (roll_std + 1e-10)
        return z.abs() > z_thresh

    def handle_gaps(self, df):
        ohlc = [c for c in ["Open","High","Low","Close"] if c in df.columns]
        df[ohlc] = df[ohlc].ffill(limit=5)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)
        return df

    def clean(self, df, ticker):
        df = self.frame_engine.clean_ohlcv(df.copy())
        outliers = self.detect_outliers(df)
        if outliers.sum() > 0:
            logger.warning(f"[{ticker}] {outliers.sum()} outliers detectados")
        df = self.handle_gaps(df)
        if "log_ret" not in df.columns:
            df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        if "simple_ret" not in df.columns:
            df["simple_ret"] = df["Close"].pct_change()
        if "Volume" in df.columns and "dollar_vol" not in df.columns:
            df["dollar_vol"] = df["Close"] * df["Volume"]
        df["is_outlier"] = outliers
        return df.dropna(subset=["Close"])


class DataValidator:

    def validate(self, df, ticker):
        total   = len(df)
        cols    = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        missing = df[cols].isnull().mean().mean() if cols else 1.0
        cleaner = DataCleaner()
        try:
            outliers = cleaner.detect_outliers(df).sum()
        except Exception:
            outliers = 0
        zero_vol = int((df["Volume"] == 0).sum()) if "Volume" in df.columns else 0
        gaps     = int((df.index.to_series().diff() > pd.Timedelta("7D")).sum()) if total > 1 else 0

        miss_score = 1.0 - float(missing)
        out_score  = 1.0 - min(outliers / max(total, 1) / 0.05, 1.0)
        vol_score  = 1.0 - min(zero_vol / max(total, 1) / 0.10, 1.0)
        score = 0.40 + 0.30*miss_score + 0.20*out_score + 0.10*vol_score

        return {
            "ticker":        ticker,
            "total_rows":    total,
            "missing_pct":   round(float(missing)*100, 2),
            "outlier_count": int(outliers),
            "zero_vol_days": zero_vol,
            "date_gaps":     gaps,
            "quality_score": round(float(score), 4),
            "passed":        score >= 0.80,
        }

    def validate_universe(self, tickers, store):
        reports = []
        for t in tickers:
            try:
                df = store.load("raw", t)
                r  = self.validate(df, t)
                status = "PASS" if r["passed"] else "FAIL"
                logger.info(f"[{status}] {t}: score={r['quality_score']:.3f}")
                reports.append(r)
            except Exception as e:
                reports.append({
                    "ticker": t, "passed": False,
                    "quality_score": 0.0, "error": str(e)
                })
        df_out = pd.DataFrame(reports)
        if "quality_score" in df_out.columns:
            df_out = df_out.sort_values("quality_score", ascending=False)
        return df_out


class DataStore:

    LAYERS = {
        "raw":       DATA_RAW,
        "processed": DATA_PROC,
        "features":  DATA_FEAT,
    }

    def save(self, df, layer, ticker, version="v1"):
        base = self.LAYERS[layer]
        path = base / version / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy")
        return path

    def load(self, layer, ticker, start=None, end=None, version="v1"):
        base = self.LAYERS.get(layer, DATA_RAW)
        engine = FrameEngine()
        candidates = [
            base / version / f"{ticker}.parquet",
            base / f"{ticker}.parquet",
        ]
        for c in candidates:
            if c.exists():
                df = engine.read_parquet(c)
                if start:
                    df = df[pd.to_datetime(df.index) >= pd.Timestamp(start)]
                if end:
                    df = df[pd.to_datetime(df.index) <= pd.Timestamp(end)]
                return df
        raise FileNotFoundError(f"{layer}/{ticker} no encontrado")

    def exists(self, layer, ticker, version="v1"):
        base = self.LAYERS.get(layer, DATA_RAW)
        return (base / version / f"{ticker}.parquet").exists() or \
               (base / f"{ticker}.parquet").exists()

    def load_panel(self, layer, tickers, column="Close", start=None, end=None):
        series = {}
        for t in tickers:
            try:
                series[t] = self.load(layer, t, start, end)[column]
            except Exception:
                pass
        return pd.DataFrame(series)


class IngestionPipeline:

    def __init__(self, provider=None):
        self.downloader = OHLCVDownloader(provider=provider)
        self.cleaner    = DataCleaner()
        self.validator  = DataValidator()
        self.store      = DataStore()

    def run(self, tickers=None, start=None, end=None, force=False, source=None):
        tickers = self.downloader.resolve_instruments(tickers=tickers, source=source)
        start   = start   or UNIVERSE_CFG.start_date
        end     = end     or UNIVERSE_CFG.end_date

        print(f"\n{'='*60}")
        print(f"  INGESTION  |  {len(tickers)} tickers  |  {start} -> {end}  |  {self.downloader.provider}")
        print(f"{'='*60}")

        to_dl = tickers if force else [t for t in tickers
                                        if not self.store.exists("raw", t)]
        if to_dl:
            print(f"\n[1/3] Descargando {len(to_dl)} tickers...")
            self.downloader.download_universe(to_dl, start, end, source=source)
        else:
            print("[1/3] Datos ya descargados — skip")

        print("[2/3] Limpiando y procesando...")
        for t in tickers:
            try:
                raw   = self.downloader.load_raw(t)
                clean = self.cleaner.clean(raw, t)
                self.store.save(clean, "processed", t)
            except Exception as e:
                logger.error(f"[FAIL] {t}: {e}")

        print("[3/3] Validando calidad...")
        quality = self.validator.validate_universe(tickers, self.store)

        passed_count = int(quality["passed"].sum()) if "passed" in quality.columns else 0
        print(f"\n  Tickers aptos: {passed_count}/{len(tickers)}")

        show_cols = [c for c in ["ticker","quality_score","passed",
                                  "total_rows","missing_pct","outlier_count"]
                     if c in quality.columns]
        if show_cols:
            print(quality[show_cols].to_string(index=False))

        print(f"{'='*60}\n")
        return quality
