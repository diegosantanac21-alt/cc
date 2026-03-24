"""
INDICADORES TECNICOS Y SENALES
Momentum, Mean Reversion, Volatilidad, Microestructura
REGLA: Todo rolling usa .shift(1) — sin look-ahead bias.
"""
import logging
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    HAS_SM = True
except ImportError:
    HAS_SM = False

from config.settings import EDGE_CFG

logger = logging.getLogger("indicators")


def _ema(s, w):
    return s.ewm(span=w, adjust=False).mean()

def _sma(s, w):
    return s.rolling(w, min_periods=max(w//2,1)).mean()

def _std(s, w):
    return s.rolling(w, min_periods=max(w//2,1)).std()

def _rsi(s, w=14):
    delta = s.shift(1).diff()
    gain  = delta.clip(lower=0).rolling(w, min_periods=w).mean()
    loss  = (-delta.clip(upper=0)).rolling(w, min_periods=w).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


class MomentumSignals:

    def ma_cross(self, prices, fast, slow):
        result = {}
        for col in prices.columns:
            p = prices[col].dropna()
            f = _ema(p.shift(1), fast)
            s = _ema(p.shift(1), slow)
            sig = pd.Series(np.where(f > s, 1.0, -1.0), index=p.index)
            result[col] = sig
        return pd.DataFrame(result).reindex(prices.index)

    def triple_ma(self, prices, s, m, l):
        result = {}
        for col in prices.columns:
            p  = prices[col].shift(1)
            e1 = _ema(p, s); e2 = _ema(p, m); e3 = _ema(p, l)
            sig = np.where((e1>e2)&(e2>e3), 1.0,
                  np.where((e1<e2)&(e2<e3), -1.0, 0.0))
            result[col] = pd.Series(sig, index=prices.index)
        return pd.DataFrame(result)

    def momentum_n(self, prices, lookback, skip=1):
        shifted = prices.shift(skip + 1)
        ret     = prices.shift(1) / (shifted + 1e-10) - 1
        return ret.clip(-0.5, 0.5)

    def trend_vs_ma(self, prices, window=200):
        lag = prices.shift(1)
        ma  = _sma(lag, window)
        return (lag - ma) / (ma.abs() + 1e-10)

    def roc(self, prices, window=20):
        return prices.shift(1) / (prices.shift(window+1) + 1e-10) - 1


class MeanReversionSignals:

    def zscore(self, prices, window=60):
        lag = prices.shift(1)
        mu  = _sma(lag, window)
        std = _std(lag, window)
        z   = (lag - mu) / (std + 1e-10)
        return -z

    def bollinger_pct_b(self, prices, window=20, k=2.0):
        lag   = prices.shift(1)
        mu    = _sma(lag, window)
        std   = _std(lag, window)
        upper = mu + k * std
        lower = mu - k * std
        pct_b = (lag - lower) / (upper - lower + 1e-10)
        return (0.5 - pct_b)

    def rsi_signal(self, prices, window=14):
        result = {}
        for col in prices.columns:
            result[col] = _rsi(prices[col], window)
        return (50 - pd.DataFrame(result, index=prices.index))

    def half_life_map(self, prices):
        if not HAS_SM:
            return {}
        hl = {}
        for col in prices.columns:
            p = prices[col].dropna()
            try:
                delta = p.diff().dropna()
                lag   = p.shift(1).dropna()
                idx   = delta.index.intersection(lag.index)
                res   = sm.OLS(delta.loc[idx], sm.add_constant(lag.loc[idx])).fit()
                theta = res.params.iloc[1]
                hl[col] = float(-np.log(2)/theta) if theta < 0 else np.inf
            except Exception:
                hl[col] = np.inf
        return hl


class VolatilitySignals:

    def realized_vol(self, log_returns, windows=None):
        windows = windows or [5, 21, 63]
        result  = {}
        for w in windows:
            rv = log_returns.shift(1).rolling(w, min_periods=max(w//2,1)).std() * np.sqrt(252)
            result[f"rvol_{w}d"] = rv
        return result

    def vol_momentum(self, log_returns):
        v5  = log_returns.shift(1).rolling(5,  min_periods=3).std()
        v20 = log_returns.shift(1).rolling(20, min_periods=10).std()
        return v5 / (v20 + 1e-10)

    def low_vol_signal(self, log_returns, window=63):
        rv = log_returns.shift(1).rolling(window, min_periods=max(window//2,1)).std()
        return -rv

    def parkinson_vol(self, high, low, window=21):
        hl  = np.log((high.shift(1)+1e-10) / (low.shift(1)+1e-10))
        pk  = hl.pow(2).rolling(window, min_periods=max(window//2,1)).mean()
        return np.sqrt(pk / (4 * np.log(2))) * np.sqrt(252)


class HybridSignals:

    def trend_rsi_lowvol(self, prices, trend_window=150, momentum_window=63, rsi_window=14, vol_window=21):
        lag = prices.shift(1)
        ma = _sma(lag, trend_window)
        trend = (lag / (ma + 1e-10) - 1).clip(-1.0, 1.0)
        momentum = (lag / (lag.shift(momentum_window) + 1e-10) - 1).clip(-1.0, 1.0)

        log_ret = np.log(prices / prices.shift(1))
        vol = log_ret.shift(1).rolling(vol_window, min_periods=max(vol_window // 2, 1)).std() * np.sqrt(252)

        rsi = {}
        for col in prices.columns:
            rsi[col] = _rsi(prices[col], rsi_window)
        rsi = pd.DataFrame(rsi, index=prices.index)

        # Favorece tendencia y momentum, penaliza volatilidad alta y RSI extremo.
        rsi_confirmation = ((rsi >= 50) & (rsi <= 72)).astype(float)
        overbought_penalty = (rsi > 72).astype(float) * 0.25
        oversold_penalty = (rsi < 40).astype(float) * 0.35

        score = 0.45 * trend + 0.35 * momentum - 0.20 * vol
        score = score * (0.5 + 0.5 * rsi_confirmation) - overbought_penalty - oversold_penalty
        return score


class ProfessionalAlphaSignals:

    def trend_quality(self, prices, fast_mom=63, slow_mom=252, trend_window=200, vol_window=63):
        lag = prices.shift(1)
        mom_fast = lag / (lag.shift(fast_mom) + 1e-10) - 1
        mom_slow = lag / (lag.shift(slow_mom) + 1e-10) - 1
        trend = lag / (_sma(lag, trend_window) + 1e-10) - 1
        log_ret = np.log(prices / prices.shift(1))
        vol = log_ret.shift(1).rolling(vol_window, min_periods=max(vol_window // 2, 1)).std() * np.sqrt(252)

        # Alpha de mediano plazo: momentum persistente con penalizacion por volatilidad.
        return 0.35 * mom_fast + 0.40 * mom_slow + 0.25 * trend - 0.20 * vol

    def pullback_in_trend(self, prices, trend_window=150, pullback_window=10, rsi_window=14):
        lag = prices.shift(1)
        trend = lag / (_sma(lag, trend_window) + 1e-10) - 1
        pullback = -(lag / (lag.shift(pullback_window) + 1e-10) - 1)

        rsi = {}
        for col in prices.columns:
            rsi[col] = _rsi(prices[col], rsi_window)
        rsi = pd.DataFrame(rsi, index=prices.index)

        # Busca comprar retrocesos dentro de una tendencia alcista estable.
        confirmation = ((trend > 0).astype(float) * ((50 - rsi) / 50).clip(-1.0, 1.0))
        score = 0.55 * pullback + 0.45 * confirmation
        return score.where(trend > 0, -trend.abs())

    def compression_breakout(self, prices, breakout_window=55, short_vol_window=10, long_vol_window=63):
        lag = prices.shift(1)
        rolling_high = lag.rolling(breakout_window, min_periods=max(breakout_window // 2, 1)).max()
        breakout = lag / (rolling_high + 1e-10) - 1

        log_ret = np.log(prices / prices.shift(1))
        short_vol = log_ret.shift(1).rolling(short_vol_window, min_periods=max(short_vol_window // 2, 1)).std()
        long_vol = log_ret.shift(1).rolling(long_vol_window, min_periods=max(long_vol_window // 2, 1)).std()
        compression = 1.0 - short_vol / (long_vol + 1e-10)

        medium_mom = lag / (lag.shift(21) + 1e-10) - 1

        # Favorece activos cerca de maximos y con compresion de volatilidad.
        return 0.45 * breakout + 0.35 * compression + 0.20 * medium_mom

    def pro_multi_alpha(self, prices):
        tq = self.trend_quality(prices)
        pb = self.pullback_in_trend(prices)
        cb = self.compression_breakout(prices)
        return 0.40 * tq + 0.30 * pb + 0.30 * cb


class MicrostructureSignals:

    def amihud(self, prices, volume, window=21):
        ret_abs    = prices.pct_change().shift(1).abs()
        dollar_vol = (prices * volume).shift(1)
        illiq      = ret_abs / (dollar_vol + 1e-10) * 1e6
        return illiq.rolling(window, min_periods=max(window//2,1)).mean()

    def log_adv(self, prices, volume, window=21):
        dv = (prices.shift(1) * volume.shift(1)).rolling(
              window, min_periods=max(window//2,1)).mean()
        return np.log(dv + 1)


class CrossSectionNormalizer:

    def zscore(self, df):
        mu  = df.mean(axis=1)
        std = df.std(axis=1)
        return df.subtract(mu, axis=0).divide(std + 1e-10, axis=0)

    def rank(self, df):
        return df.rank(axis=1, pct=True, na_option="keep")

    def winsorize(self, df, pct=0.025):
        lo = df.quantile(pct, axis=1)
        hi = df.quantile(1-pct, axis=1)
        return df.clip(lower=lo, upper=hi, axis=0)

    def full(self, df):
        return self.zscore(self.winsorize(df))


class ICAnalyzer:

    def compute(self, signal, fwd_returns):
        ics = {}
        for date in signal.index:
            try:
                s = signal.loc[date].dropna()
                r = fwd_returns.loc[date].reindex(s.index).dropna()
                common = s.index.intersection(r.index)
                if len(common) < 15:
                    continue
                ic, _ = scipy_stats.spearmanr(s[common], r[common])
                if not np.isnan(ic):
                    ics[date] = ic
            except Exception:
                continue
        return pd.Series(ics, name="IC")

    def summary(self, ic):
        ic  = ic.dropna()
        n   = len(ic)
        if n < 5:
            return {"viable": False, "n_periods": n, "ic_mean": 0.0,
                    "icir": 0.0, "t_stat": 0.0, "pct_positive": 0.0}
        mu   = ic.mean()
        std  = ic.std()
        se   = std / np.sqrt(n)
        t    = mu / se if se > 0 else 0.0
        icir = mu / std if std > 0 else 0.0
        cfg  = EDGE_CFG
        return {
            "n_periods":    n,
            "ic_mean":      round(float(mu),   4),
            "ic_std":       round(float(std),  4),
            "icir":         round(float(icir), 3),
            "t_stat":       round(float(t),    3),
            "pct_positive": round(float((ic>0).mean()), 3),
            "viable": (abs(mu) >= cfg.ic_min_mean and
                       abs(icir) >= cfg.ic_min_icir and
                       abs(t) >= cfg.ic_min_tstat),
        }

    def decay(self, signal, prices, horizons=None):
        horizons = horizons or [1, 5, 10, 21, 42, 63]
        log_ret  = np.log(prices / prices.shift(1))
        rows = []
        for h in horizons:
            fwd  = log_ret.rolling(h).sum().shift(-h)
            ic   = self.compute(signal, fwd)
            row  = self.summary(ic)
            row["horizon_days"] = h
            rows.append(row)
        return pd.DataFrame(rows).set_index("horizon_days")
