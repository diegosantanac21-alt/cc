"""
SIMULACION MONTE CARLO
Modelos soportados: GBM, Merton, Heston, Regime Switching, Bootstrap, GARCH.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("montecarlo")

EPS = 1e-12
MAX_LOG_PRICE = 50.0
MIN_POSITIVE_PRICE = 1e-8


def _safe_exp(x):
    return np.exp(np.clip(x, -MAX_LOG_PRICE, MAX_LOG_PRICE))


def _safe_prices_from_log_returns(log_rets, s0):
    log_paths = np.cumsum(log_rets, axis=1)
    prices = s0 * _safe_exp(np.hstack([np.zeros((log_rets.shape[0], 1)), log_paths]))
    return np.maximum(prices, MIN_POSITIVE_PRICE)


def _safe_pct_returns(paths):
    prev = np.maximum(paths[:, :-1], MIN_POSITIVE_PRICE)
    return paths[:, 1:] / prev - 1.0


def _validate_params(p):
    checks = {
        "n_paths": getattr(p, "n_paths", 0),
        "n_days": getattr(p, "n_days", 0),
        "dt": getattr(p, "dt", 0),
        "sigma": getattr(p, "sigma", 0),
        "S0": getattr(p, "S0", 0),
    }
    if checks["n_paths"] <= 0:
        raise ValueError("n_paths debe ser > 0")
    if checks["n_days"] <= 0:
        raise ValueError("n_days debe ser > 0")
    if checks["dt"] <= 0:
        raise ValueError("dt debe ser > 0")
    if checks["sigma"] < 0:
        raise ValueError("sigma no puede ser negativo")
    if checks["S0"] <= 0:
        raise ValueError("S0 debe ser > 0")
    if hasattr(p, "rho") and abs(p.rho) >= 1:
        raise ValueError("rho debe estar entre -1 y 1")
    if hasattr(p, "garch_alpha") and hasattr(p, "garch_beta"):
        if p.garch_alpha < 0 or p.garch_beta < 0:
            raise ValueError("garch_alpha y garch_beta deben ser >= 0")
        if p.garch_alpha + p.garch_beta >= 1.0:
            logger.warning("GARCH alpha+beta >= 1; se aplicara estabilizacion numerica")


class BaseSimulator:

    def _rng(self, p):
        _validate_params(p)
        return np.random.default_rng(p.seed)


class GBMSimulator(BaseSimulator):
    """Geometric Brownian Motion."""

    def simulate(self, p):
        rng = self._rng(p)
        mu_d = (p.mu - 0.5 * p.sigma ** 2) * p.dt
        sig_d = p.sigma * np.sqrt(p.dt)
        z = rng.standard_normal((p.n_paths, p.n_days))
        log_rets = mu_d + sig_d * z
        return _safe_prices_from_log_returns(log_rets, p.S0)


class MertonJumpSimulator(BaseSimulator):
    """GBM con saltos de Poisson."""

    def simulate(self, p):
        rng = self._rng(p)
        dt = p.dt
        k = np.exp(p.jump_mu + 0.5 * p.jump_sigma ** 2) - 1.0
        mu_adj = (p.mu - p.jump_lambda * k - 0.5 * p.sigma ** 2) * dt
        sig_d = p.sigma * np.sqrt(dt)
        lam_d = max(p.jump_lambda * dt, 0.0)

        z = rng.standard_normal((p.n_paths, p.n_days))
        n_jumps = rng.poisson(lam_d, (p.n_paths, p.n_days))
        jump_sizes = np.zeros_like(z)
        mask = n_jumps > 0
        n_mask = int(mask.sum())
        if n_mask > 0:
            per_jump = rng.normal(p.jump_mu, p.jump_sigma, n_mask)
            jump_sizes[mask] = per_jump * n_jumps[mask]

        log_rets = mu_adj + sig_d * z + jump_sizes
        return _safe_prices_from_log_returns(log_rets, p.S0)


class HestonSimulator(BaseSimulator):
    """Volatilidad estocastica con esquema full truncation."""

    def simulate(self, p):
        rng = self._rng(p)
        n_paths, n_days = p.n_paths, p.n_days
        dt = p.dt
        prices = np.full((n_paths, n_days + 1), p.S0, dtype=float)
        var = np.full(n_paths, max(p.v0, EPS), dtype=float)

        rho_term = np.sqrt(max(1.0 - p.rho ** 2, 0.0))
        for t in range(n_days):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            z_var = z1
            z_price = p.rho * z1 + rho_term * z2

            var_pos = np.maximum(var, 0.0)
            sqrt_var = np.sqrt(np.maximum(var_pos, EPS))
            dv = (
                p.kappa * (p.theta - var_pos) * dt
                + p.xi * sqrt_var * np.sqrt(dt) * z_var
            )
            var = np.maximum(var_pos + dv, EPS)
            log_ret = (p.mu - 0.5 * var_pos) * dt + np.sqrt(var_pos * dt) * z_price
            prices[:, t + 1] = np.maximum(prices[:, t] * _safe_exp(log_ret), MIN_POSITIVE_PRICE)
        return prices


class RegimeSwitchingSimulator(BaseSimulator):
    """Markov Regime Switching con calibracion por dt."""

    def simulate(self, p):
        rng = self._rng(p)
        n_paths, n_days = p.n_paths, p.n_days
        prices = np.full((n_paths, n_days + 1), p.S0, dtype=float)
        state = np.ones(n_paths, dtype=int)

        mu_step = np.array([p.mu_bear * p.dt, p.mu_bull * p.dt], dtype=float)
        sig_step = np.array([p.sig_bear * np.sqrt(p.dt), p.sig_bull * np.sqrt(p.dt)], dtype=float)
        p_bull2bear = np.clip(p.p_bull2bear, 0.0, 1.0)
        p_bear2bull = np.clip(p.p_bear2bull, 0.0, 1.0)

        for t in range(n_days):
            u = rng.random(n_paths)
            state[(state == 1) & (u < p_bull2bear)] = 0
            state[(state == 0) & (u < p_bear2bull)] = 1
            z = rng.standard_normal(n_paths)
            log_ret = mu_step[state] - 0.5 * sig_step[state] ** 2 + sig_step[state] * z
            prices[:, t + 1] = np.maximum(prices[:, t] * _safe_exp(log_ret), MIN_POSITIVE_PRICE)
        return prices


class BootstrapSimulator(BaseSimulator):
    """Bootstrap por bloques con saneamiento para historicos cortos."""

    def simulate(self, hist_returns, p):
        rng = self._rng(p)
        hist = np.asarray(getattr(hist_returns, "dropna", lambda: hist_returns)(), dtype=float).reshape(-1)
        hist = hist[np.isfinite(hist)]
        if hist.size < 5:
            raise ValueError("historical_returns debe tener al menos 5 observaciones validas")

        block = int(min(max(getattr(p, "bootstrap_block", 20), 2), max(hist.size - 1, 2), p.n_days))
        n_blocks = int(np.ceil(p.n_days / block))
        prices = np.full((p.n_paths, p.n_days + 1), p.S0, dtype=float)

        max_start = max(hist.size - block + 1, 1)
        for i in range(p.n_paths):
            starts = rng.integers(0, max_start, size=n_blocks)
            sampled = np.concatenate([hist[s:s + block] for s in starts])[:p.n_days]
            prices[i, 1:] = np.maximum(p.S0 * np.cumprod(1.0 + sampled), MIN_POSITIVE_PRICE)
        return prices


class GARCHSimulator(BaseSimulator):
    """GARCH(1,1) con varianza estacionaria estabilizada."""

    def simulate(self, p):
        rng = self._rng(p)
        n_paths, n_days = p.n_paths, p.n_days
        alpha = max(p.garch_alpha, 0.0)
        beta = max(p.garch_beta, 0.0)
        persistence = min(alpha + beta, 0.999)
        long_var = max((p.sigma ** 2) * p.dt, EPS)
        omega = max(p.garch_omega, long_var * (1.0 - persistence))

        prices = np.full((n_paths, n_days + 1), p.S0, dtype=float)
        var = np.full(n_paths, long_var, dtype=float)

        for t in range(n_days):
            z = rng.standard_t(df=8, size=n_paths) / np.sqrt(8.0 / 6.0)
            eps = np.sqrt(np.maximum(var, EPS)) * z
            log_ret = (p.mu - 0.5 * var / max(p.dt, EPS)) * p.dt + eps
            prices[:, t + 1] = np.maximum(prices[:, t] * _safe_exp(log_ret), MIN_POSITIVE_PRICE)
            var = omega + alpha * eps ** 2 + beta * var
            var = np.clip(var, EPS, 4.0)
        return prices


@dataclass
class MCResults:
    model_name: str
    n_paths: int
    n_days: int
    paths: np.ndarray
    final_rets: np.ndarray = field(init=False)
    daily_rets: np.ndarray = field(init=False)
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.paths = np.asarray(self.paths, dtype=float)
        if self.paths.ndim != 2 or self.paths.shape[1] < 2:
            raise ValueError("paths debe tener shape (n_paths, n_days+1)")
        self.paths = np.maximum(self.paths, MIN_POSITIVE_PRICE)
        self.final_rets = self.paths[:, -1] / self.paths[:, 0] - 1.0
        self.daily_rets = _safe_pct_returns(self.paths)
        self.metrics = self._compute()

    def _compute(self):
        r = self.final_rets[np.isfinite(self.final_rets)]
        if r.size == 0:
            raise ValueError("No hay retornos finales validos para calcular metricas")

        n = max(self.n_days, 1)
        ann_factor = 252.0 / n
        cagrs = np.power(np.maximum(1.0 + r, EPS), ann_factor) - 1.0

        peak = np.maximum.accumulate(self.paths, axis=1)
        drawdowns = self.paths / np.maximum(peak, MIN_POSITIVE_PRICE) - 1.0
        mdds = drawdowns.min(axis=1)

        daily = self.daily_rets[np.isfinite(self.daily_rets)]
        daily_mean = float(np.mean(daily)) if daily.size else 0.0
        daily_std = float(np.std(daily)) if daily.size else 0.0
        sharpe = daily_mean / (daily_std + EPS) * np.sqrt(252.0)

        var95 = float(np.percentile(r, 5))
        var99 = float(np.percentile(r, 1))
        tail95 = r[r <= var95]
        tail99 = r[r <= var99]

        return {
            "mean_return": float(np.mean(r)),
            "median_return": float(np.median(r)),
            "std_return": float(np.std(r)),
            "min_return": float(np.min(r)),
            "max_return": float(np.max(r)),
            "skewness": float(scipy_stats.skew(r, bias=False)) if r.size > 2 else 0.0,
            "kurtosis": float(scipy_stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else 0.0,
            "mean_cagr": float(np.mean(cagrs)),
            "median_cagr": float(np.median(cagrs)),
            "p1_cagr": float(np.percentile(cagrs, 1)),
            "p5_cagr": float(np.percentile(cagrs, 5)),
            "p25_cagr": float(np.percentile(cagrs, 25)),
            "p75_cagr": float(np.percentile(cagrs, 75)),
            "p95_cagr": float(np.percentile(cagrs, 95)),
            "p99_cagr": float(np.percentile(cagrs, 99)),
            "var_95": var95,
            "var_99": var99,
            "cvar_95": float(np.mean(tail95)) if tail95.size else var95,
            "cvar_99": float(np.mean(tail99)) if tail99.size else var99,
            "prob_loss": float(np.mean(r < 0.0)),
            "prob_loss_10": float(np.mean(r < -0.10)),
            "prob_loss_20": float(np.mean(r < -0.20)),
            "prob_gain_20": float(np.mean(r > 0.20)),
            "mean_mdd": float(np.mean(mdds)),
            "median_mdd": float(np.median(mdds)),
            "p95_mdd": float(np.percentile(mdds, 95)),
            "worst_mdd": float(np.min(mdds)),
            "daily_sharpe": float(sharpe),
            "pct_above_0": float(np.mean(r > 0.0)),
        }

    def print_summary(self):
        m = self.metrics
        sep = "-" * 50
        print(f"\n{'=' * 50}")
        print(f"  MONTE CARLO: {self.model_name}")
        print(f"  {self.n_paths:,} paths x {self.n_days} dias")
        print(f"{'=' * 50}")
        print("\n  RETORNOS:")
        print(f"  {sep}")
        print(f"  Media:        {m['mean_return'] * 100:>8.2f}%")
        print(f"  Mediana:      {m['median_return'] * 100:>8.2f}%")
        print(f"  Std:          {m['std_return'] * 100:>8.2f}%")
        print(f"  Min / Max:    {m['min_return'] * 100:>8.2f}% / {m['max_return'] * 100:>8.2f}%")
        print(f"  Sharpe diario:{m['daily_sharpe']:>8.3f}")
        print(f"  Skewness:     {m['skewness']:>8.3f}")
        print(f"  Kurtosis:     {m['kurtosis']:>8.3f}")
        print("\n  CAGR:")
        print(f"  {sep}")
        for pct, key in [(1, "p1_cagr"), (5, "p5_cagr"), (50, "median_cagr"), (95, "p95_cagr"), (99, "p99_cagr")]:
            print(f"  P{pct:<3}:         {m[key] * 100:>8.2f}%")
        print("\n  RIESGO:")
        print(f"  {sep}")
        print(f"  VaR 95%:      {m['var_95'] * 100:>8.2f}%")
        print(f"  CVaR 95%:     {m['cvar_95'] * 100:>8.2f}%")
        print(f"  VaR 99%:      {m['var_99'] * 100:>8.2f}%")
        print(f"  CVaR 99%:     {m['cvar_99'] * 100:>8.2f}%")
        print(f"  P(perdida):   {m['prob_loss'] * 100:>8.1f}%")
        print(f"  P(>10% loss): {m['prob_loss_10'] * 100:>8.1f}%")
        print(f"  P(>20% gain): {m['prob_gain_20'] * 100:>8.1f}%")
        print("\n  DRAWDOWN:")
        print(f"  {sep}")
        print(f"  MDD medio:    {m['mean_mdd'] * 100:>8.2f}%")
        print(f"  MDD mediano:  {m['median_mdd'] * 100:>8.2f}%")
        print(f"  MDD P95:      {m['p95_mdd'] * 100:>8.2f}%")
        print(f"  MDD peor:     {m['worst_mdd'] * 100:>8.2f}%")
        print(f"{'=' * 50}\n")


class MonteCarloEngine:

    MODELS = {
        "gbm": GBMSimulator,
        "merton": MertonJumpSimulator,
        "heston": HestonSimulator,
        "regime": RegimeSwitchingSimulator,
        "garch": GARCHSimulator,
    }

    def run(self, model, params, historical_returns=None):
        model = model.lower()
        logger.info(f"MC: {model} | {params.n_paths} paths x {params.n_days} dias")
        _validate_params(params)

        if model == "bootstrap":
            if historical_returns is None:
                raise ValueError("bootstrap requiere historical_returns")
            paths = BootstrapSimulator().simulate(historical_returns, params)
        else:
            sim_cls = self.MODELS.get(model)
            if sim_cls is None:
                raise ValueError(f"Modelo desconocido: {model}")
            paths = sim_cls().simulate(params)

        return MCResults(
            model_name=model,
            n_paths=params.n_paths,
            n_days=params.n_days,
            paths=paths,
        )

    def compare_models(self, params, models=None, historical_returns=None):
        models = models or list(self.MODELS.keys())
        results = {}
        for model in models:
            try:
                results[model] = self.run(model, params, historical_returns=historical_returns)
                logger.info(f"  {model}: mean_cagr={results[model].metrics['mean_cagr'] * 100:.2f}%")
            except Exception as exc:
                logger.error(f"  {model} error: {exc}")
        return results

    def run_strategy_mc(self, strategy_fn, params, model="gbm", cost_bps=5.0, historical_returns=None):
        mc_raw = self.run(model, params, historical_returns=historical_returns)
        cost = cost_bps / 10000.0
        strat_paths = np.full((params.n_paths, params.n_days + 1), params.S0, dtype=float)

        for i in range(params.n_paths):
            prices = mc_raw.paths[i]
            signal = np.asarray(strategy_fn(prices), dtype=float)
            if signal.shape[0] != prices.shape[0]:
                raise ValueError("strategy_fn debe devolver una senal del mismo largo que prices")
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            signal = np.clip(signal, -1.0, 1.0)

            equity = params.S0
            pos = 0.0
            for t in range(1, params.n_days + 1):
                new_pos = float(signal[t])
                price_ret = (prices[t] - prices[t - 1]) / max(prices[t - 1], MIN_POSITIVE_PRICE)
                tc = abs(new_pos - pos) * cost
                step_ret = np.clip(pos * price_ret - tc, -0.99, 10.0)
                equity *= (1.0 + step_ret)
                equity = max(equity, MIN_POSITIVE_PRICE)
                strat_paths[i, t] = equity
                pos = new_pos

        strat = MCResults(
            model_name=f"Strategy({model})",
            n_paths=params.n_paths,
            n_days=params.n_days,
            paths=strat_paths,
        )
        return {"buy_and_hold": mc_raw, "strategy": strat}
