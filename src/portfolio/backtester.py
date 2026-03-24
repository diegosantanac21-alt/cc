"""
Ejecucion de cartera para convertir alphas en una estrategia operable.

Incluye:
- seleccion cross-sectional por ranking
- pesos long/short o long-only
- rebalanceo diario/semanal/mensual
- ajuste simple por volatilidad
- backtest de cartera con costos
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import PORTFOLIO_CFG
from src.risk.metrics import RiskMetrics

logger = logging.getLogger("portfolio")

EPS = 1e-12


@dataclass
class PortfolioBacktestResult:
    alpha_name: str
    weights: pd.DataFrame
    returns: pd.Series
    equity: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    turnover: pd.Series
    report: dict
    meta: dict = field(default_factory=dict)

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"  PORTFOLIO STRATEGY: {self.alpha_name}")
        print(f"{'=' * 60}")
        print(f"  Rebanceo:      {self.meta.get('rebalance_frequency', 'n/a')}")
        print(f"  Long/Short:    {self.meta.get('long_short', False)}")
        print(f"  Top/Bottom:    {self.meta.get('top_n', 0)} / {self.meta.get('bottom_n', 0)}")
        print(f"  Universo medio:{self.report.get('avg_active_names', 0):>10.2f}")
        print(f"  Turnover prom.:{self.report.get('avg_turnover', 0):>10.4f}")
        print("")
        print(f"  Retorno total: {self.report.get('total_return', 0) * 100:>10.2f}%")
        print(f"  CAGR:          {self.report.get('cagr', 0) * 100:>10.2f}%")
        print(f"  Sharpe:        {self.report.get('sharpe', 0):>10.3f}")
        print(f"  Sortino:       {self.report.get('sortino', 0):>10.3f}")
        print(f"  Max Drawdown:  {self.report.get('max_drawdown', 0) * 100:>10.2f}%")
        print(f"  Calmar:        {self.report.get('calmar', 0):>10.3f}")
        print(f"  Beta vs bench: {self.report.get('beta', np.nan):>10.3f}")
        print(f"  IR vs bench:   {self.report.get('ir', 0):>10.3f}")
        print(f"{'=' * 60}\n")


class AlphaPortfolioBacktester:

    def __init__(self, cfg=None):
        self.cfg = cfg or PORTFOLIO_CFG
        self.risk = RiskMetrics()

    def _rebalance_mask(self, index):
        idx = pd.DatetimeIndex(index)
        freq = str(self.cfg.rebalance_frequency).lower()
        if freq in {"daily", "day", "d"}:
            return pd.Series(True, index=idx)
        if freq in {"weekly", "week", "w"}:
            labels = idx.to_period("W-FRI")
        elif freq in {"monthly", "month", "m"}:
            labels = idx.to_period("M")
        else:
            labels = idx.to_period("W-FRI")
        labels = pd.Series(labels, index=idx)
        return labels.ne(labels.shift(1)).fillna(True)

    def _normalize_with_cap(self, raw, target_sum):
        raw = raw.clip(lower=0.0).astype(float)
        if len(raw) == 0 or target_sum <= 0:
            return raw * 0.0
        max_weight = max(float(self.cfg.max_weight), 0.0)
        feasible_target = min(float(target_sum), max_weight * len(raw)) if max_weight > 0 else float(target_sum)
        weights = raw / max(raw.sum(), EPS) * feasible_target
        if max_weight <= 0:
            return weights
        for _ in range(len(weights) + 5):
            over = weights > max_weight + 1e-12
            if not over.any():
                break
            excess = float((weights[over] - max_weight).sum())
            weights[over] = max_weight
            under = ~over
            if excess <= 1e-12 or not under.any():
                break
            base = weights[under]
            if base.sum() <= EPS:
                weights.loc[under] += excess / int(under.sum())
            else:
                weights.loc[under] += excess * base / base.sum()
        if weights.sum() > EPS:
            weights *= feasible_target / weights.sum()
        return weights.clip(upper=max_weight)

    def _build_side_weights(self, scores, vol_row, gross_target):
        if len(scores) == 0 or gross_target <= 0:
            return pd.Series(dtype=float)
        strength = scores.abs().clip(lower=EPS)
        if bool(self.cfg.risk_adjust_weights):
            vol = vol_row.reindex(scores.index).fillna(float(self.cfg.vol_floor)).clip(lower=float(self.cfg.vol_floor))
            strength = strength / vol
        return self._normalize_with_cap(strength, gross_target)

    def _target_weights_for_date(self, signal_row, vol_row, prev_weights):
        scores = signal_row.replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < max(int(self.cfg.min_names), int(self.cfg.top_n) + int(self.cfg.bottom_n)):
            return prev_weights.copy()

        scores = scores.sort_values()
        n_long = min(int(self.cfg.top_n), len(scores))
        n_short = min(int(self.cfg.bottom_n), max(len(scores) - n_long, 0))

        longs = scores.tail(n_long)
        shorts = scores.head(n_short) if bool(self.cfg.long_short) else pd.Series(dtype=float)
        shorts = shorts.loc[~shorts.index.isin(longs.index)]

        target = pd.Series(0.0, index=prev_weights.index, dtype=float)
        gross_leverage = max(float(self.cfg.gross_leverage), 0.0)

        if bool(self.cfg.long_short):
            long_budget = gross_leverage / 2.0
            short_budget = gross_leverage / 2.0
        else:
            long_budget = gross_leverage
            short_budget = 0.0

        long_w = self._build_side_weights(longs, vol_row, long_budget)
        target.loc[long_w.index] = long_w.values

        if bool(self.cfg.long_short) and len(shorts) > 0:
            short_w = self._build_side_weights(shorts, vol_row, short_budget)
            target.loc[short_w.index] = -short_w.values

        return target

    def run(self, prices, alpha_signal, alpha_name="alpha_pro_multi"):
        prices = prices.sort_index().copy()
        alpha_signal = alpha_signal.reindex(prices.index).reindex(columns=prices.columns)
        asset_returns = prices.pct_change().fillna(0.0)
        log_returns = np.log(prices / prices.shift(1))
        asset_vol = log_returns.shift(1).rolling(
            int(self.cfg.vol_lookback),
            min_periods=max(int(self.cfg.vol_lookback) // 2, 1),
        ).std() * np.sqrt(252)

        rebalance_mask = self._rebalance_mask(prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_weights = pd.Series(0.0, index=prices.columns, dtype=float)

        for dt in prices.index:
            if bool(rebalance_mask.loc[dt]):
                prev_weights = self._target_weights_for_date(alpha_signal.loc[dt], asset_vol.loc[dt], prev_weights)
            weights.loc[dt] = prev_weights.values

        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        cost = turnover * max(float(self.cfg.transaction_cost_bps), 0.0) / 10000.0

        strategy_returns = (weights * asset_returns).sum(axis=1) - cost
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("strategy")
        equity = (1.0 + strategy_returns).cumprod().rename("equity")

        availability = prices.notna().astype(float)
        bench_weights = availability.div(availability.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        benchmark_returns = (bench_weights * asset_returns).sum(axis=1).fillna(0.0).rename("benchmark")
        benchmark_equity = (1.0 + benchmark_returns).cumprod().rename("benchmark_equity")

        report = self.risk.full_report(strategy_returns, bench=benchmark_returns, name=f"Portfolio({alpha_name})")
        report.update({
            "avg_turnover": round(float(turnover.mean()), 4),
            "avg_active_names": round(float((weights.abs() > EPS).sum(axis=1).mean()), 2),
            "avg_gross_exposure": round(float(weights.abs().sum(axis=1).mean()), 4),
            "avg_net_exposure": round(float(weights.sum(axis=1).mean()), 4),
            "rebalance_count": int(rebalance_mask.sum()),
            "benchmark_total_return": round(float(benchmark_equity.iloc[-1] - 1.0), 4),
        })

        meta = {
            "rebalance_frequency": self.cfg.rebalance_frequency,
            "top_n": int(self.cfg.top_n),
            "bottom_n": int(self.cfg.bottom_n),
            "long_short": bool(self.cfg.long_short),
        }
        logger.info(
            "Portfolio %s | CAGR=%.2f%% | Sharpe=%.3f | MDD=%.2f%%",
            alpha_name,
            report.get("cagr", 0.0) * 100,
            report.get("sharpe", 0.0),
            report.get("max_drawdown", 0.0) * 100,
        )
        return PortfolioBacktestResult(
            alpha_name=alpha_name,
            weights=weights,
            returns=strategy_returns,
            equity=equity,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
            turnover=turnover.rename("turnover"),
            report=report,
            meta=meta,
        )
