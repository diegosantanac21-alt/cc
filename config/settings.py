from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import os

ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_FEAT = ROOT / "data" / "features"
LOGS_DIR  = ROOT / "logs"
REPORT_DIR= ROOT / "research" / "reports"

for d in [DATA_RAW, DATA_PROC, DATA_FEAT, LOGS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class UniverseConfig:
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA",
        "JPM","BAC","GS","JNJ","UNH","XOM","CVX",
        "WMT","PG","KO","MCD","CAT","HON",
    ])
    start_date: str = "2015-01-01"
    end_date:   str = "2024-12-31"
    min_adv_usd: float = 1_000_000
    min_price:   float = 5.0


@dataclass
class DataProviderConfig:
    provider: str = os.getenv("DATA_PROVIDER", "yfinance")
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_api_secret: str = os.getenv("ALPACA_API_SECRET", "")
    alpaca_feed: str = os.getenv("ALPACA_FEED", "iex")
    alpaca_adjustment: str = os.getenv("ALPACA_ADJUSTMENT", "all")
    universe_source: str = os.getenv("UNIVERSE_SOURCE", "config")
    alpaca_asset_class: str = os.getenv("ALPACA_ASSET_CLASS", "us_equity")
    alpaca_tradable_only: bool = os.getenv("ALPACA_TRADABLE_ONLY", "true").lower() == "true"
    alpaca_fractionable_only: bool = os.getenv("ALPACA_FRACTIONABLE_ONLY", "false").lower() == "true"
    alpaca_limit: int = int(os.getenv("ALPACA_LIMIT", "0"))


@dataclass
class EdgeConfig:
    ic_min_mean:     float = 0.03
    ic_min_icir:     float = 0.50
    ic_min_tstat:    float = 2.0
    fdr_q:           float = 0.05
    n_bootstrap:     int   = 3000
    bootstrap_block: int   = 20
    dsr_threshold:   float = 0.95
    wf_train_months: int   = 36
    wf_test_months:  int   = 12
    wf_min_folds:    int   = 4
    tvt_train_ratio: float = 0.60
    tvt_valid_ratio: float = 0.20
    tvt_min_periods: int   = 126
    purged_embargo_days: int = 5
    purged_n_splits: int = 5


@dataclass
class RiskConfig:
    target_vol_annual:  float = 0.10
    max_drawdown_limit: float = 0.15
    min_sharpe:         float = 1.0
    min_sortino:        float = 1.5
    min_calmar:         float = 0.5
    var_confidence:     float = 0.95


@dataclass
class PortfolioConfig:
    alpha_name: str = "alpha_pro_multi"
    rebalance_frequency: str = "weekly"
    top_n: int = 4
    bottom_n: int = 4
    long_short: bool = True
    gross_leverage: float = 1.0
    max_weight: float = 0.20
    transaction_cost_bps: float = 5.0
    risk_adjust_weights: bool = True
    vol_lookback: int = 21
    vol_floor: float = 0.10
    min_names: int = 6


@dataclass
class ScanConfig:
    n_seeds:    int   = 5
    days:       int   = 504
    drift:      float = 0.08
    volatility: float = 0.20
    cost_bps:   float = 5.0
    commission_bps: float = 2.0
    slippage_bps:   float = 3.0
    execution_lag:  int   = 1
    target_vol_annual: float = 0.10
    max_leverage:   float = 1.0
    stop_loss_pct:  float = 0.08
    take_profit_pct: float = 0.15
    optuna_trials:  int   = 50
    optuna_timeout_sec: int = 120
    optuna_metric:  str   = "score"
    regimes: List[str] = field(default_factory=lambda: [
        "normal","trending","ranging","crash","volatile"
    ])


@dataclass
class SimParams:
    n_paths:     int   = 1000
    n_days:      int   = 252
    dt:          float = 1/252
    mu:          float = 0.08
    sigma:       float = 0.20
    jump_lambda: float = 5.0
    jump_mu:     float = -0.02
    jump_sigma:  float = 0.04
    kappa:       float = 2.0
    theta:       float = 0.04
    xi:          float = 0.3
    rho:         float = -0.7
    v0:          float = 0.04
    mu_bull:     float = 0.15
    mu_bear:     float = -0.10
    sig_bull:    float = 0.12
    sig_bear:    float = 0.30
    p_bull2bear: float = 0.02
    p_bear2bull: float = 0.05
    garch_omega: float = 1e-6
    garch_alpha: float = 0.09
    garch_beta:  float = 0.90
    S0:          float = 100.0
    seed:        int   = 42


UNIVERSE_CFG = UniverseConfig()
DATA_CFG     = DataProviderConfig()
EDGE_CFG     = EdgeConfig()
RISK_CFG     = RiskConfig()
PORTFOLIO_CFG = PortfolioConfig()
SCAN_CFG     = ScanConfig()
