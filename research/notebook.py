"""
NOTEBOOK DE INVESTIGACION
run_demo, run_param_scan, run_optuna_scan, run_signal_investigation,
run_full_pipeline, run_portfolio_alpha_strategy
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from config.settings import UNIVERSE_CFG, SCAN_CFG, REPORT_DIR

DEFAULT_TEST_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","XOM","JNJ","WMT","PG"]


def _get_prices(tickers, start=None, end=None):
    from src.ingestion.pipeline import DataStore
    store=DataStore()
    prices=store.load_panel("processed",tickers,"Close",start,end)
    prices=prices.dropna(axis=1,thresh=max(len(prices)//3,50))
    return store, prices


def _build_signal(prices, name):
    from src.features.indicators import (MomentumSignals, MeanReversionSignals,
                                          VolatilitySignals, HybridSignals,
                                          ProfessionalAlphaSignals, CrossSectionNormalizer)
    mo=MomentumSignals(); mr=MeanReversionSignals()
    vo=VolatilitySignals(); hy=HybridSignals(); pa=ProfessionalAlphaSignals(); n=CrossSectionNormalizer()
    lr=np.log(prices/prices.shift(1))
    CATALOG={
        "momentum_12_1": lambda: n.rank(mo.momentum_n(prices,252,21)),
        "momentum_3m":   lambda: n.rank(mo.momentum_n(prices,63,5)),
        "momentum_1m":   lambda: n.rank(mo.momentum_n(prices,21,1)),
        "trend_ma200":   lambda: n.full(mo.trend_vs_ma(prices,200)),
        "ma_cross_10_50":lambda: n.full(mo.ma_cross(prices,10,50)),
        "triple_ma":     lambda: n.full(mo.triple_ma(prices,8,21,100)),
        "zscore_20d":    lambda: n.full(mr.zscore(prices,20)),
        "zscore_60d":    lambda: n.full(mr.zscore(prices,60)),
        "bb_pct_b":      lambda: n.full(mr.bollinger_pct_b(prices,20,2.0)),
        "rsi_14":        lambda: n.full(mr.rsi_signal(prices,14)),
        "rsi_7":         lambda: n.full(mr.rsi_signal(prices,7)),
        "low_vol":       lambda: n.full(vo.low_vol_signal(lr,63)),
        "vol_momentum":  lambda: n.full(vo.vol_momentum(lr)),
        "trend_rsi_lowvol": lambda: n.full(hy.trend_rsi_lowvol(prices,150,63,14,21)),
        "alpha_trend_quality": lambda: n.full(pa.trend_quality(prices,63,252,200,63)),
        "alpha_pullback_trend": lambda: n.full(pa.pullback_in_trend(prices,150,10,14)),
        "alpha_compression_breakout": lambda: n.full(pa.compression_breakout(prices,55,10,63)),
        "alpha_pro_multi": lambda: n.full(pa.pro_multi_alpha(prices)),
    }
    if name not in CATALOG:
        print(f"  Senales disponibles: {list(CATALOG.keys())}")
        raise ValueError(f"Senal '{name}' no encontrada")
    return CATALOG[name]()


def run_demo():
    TICKERS=DEFAULT_TEST_TICKERS
    START="2018-01-01"; END="2023-12-31"
    print(f"\n{'='*60}")
    print(f"  DEMO — QUANT RESEARCH FRAMEWORK")
    print(f"{'='*60}")
    print(f"  Tickers: {TICKERS}")
    print(f"  Periodo: {START} -> {END}")

    from src.ingestion.pipeline import IngestionPipeline
    IngestionPipeline().run(tickers=TICKERS,start=START,end=END)

    _, prices = _get_prices(TICKERS,START,END)
    if prices.empty or prices.shape[1] < 2:
        print("  No hay suficientes datos para continuar. Revisa la descarga.")
        return

    from src.models.edge_finder import EdgeFinder
    finder=EdgeFinder(); reports={}
    signals_to_test=[
        ("momentum_12_1","Momentum 12-1 meses — Jegadeesh & Titman (1993)"),
        ("zscore_60d",   "Mean-reversion Z-score 60 dias"),
        ("low_vol",      "Low-volatility anomaly — Ang et al. (2006)"),
    ]
    for sig_name,hyp in signals_to_test:
        try:
            sig=_build_signal(prices,sig_name)
            rpt=finder.investigate(signal=sig,prices=prices,signal_name=sig_name,
                                    hypothesis=hyp,horizon=21,m_trials=len(signals_to_test))
            reports[sig_name]=rpt
        except Exception as e:
            print(f"  Error en {sig_name}: {e}")

    print("\n  MINI PARAM SCAN...")
    from src.models.param_scanner import ParamScanner, FAMILIES
    mini=[ f for f in FAMILIES if f.id in ("momentum","rsi_rev","bb_rev")]
    from config.settings import ScanConfig
    cfg=ScanConfig(n_seeds=3,regimes=["normal","trending","ranging"])
    scan_df=ParamScanner(cfg).scan(families=mini,verbose=True)
    if len(scan_df)>0:
        ParamScanner().print_leaderboard(scan_df,n=10)

    _save_report(reports,scan_df,"demo_report.txt")
    print(f"\n  Demo completa. Ver research/reports/demo_report.txt\n")
    return reports, scan_df


def run_param_scan(regimes=None, n_seeds=None, days=None):
    from src.models.param_scanner import ParamScanner, FAMILIES
    from config.settings import ScanConfig
    cfg=ScanConfig(n_seeds=n_seeds or SCAN_CFG.n_seeds,
                    days=days or SCAN_CFG.days,
                    regimes=regimes or SCAN_CFG.regimes)
    scanner=ParamScanner(cfg)
    print(f"  Total combinaciones: {scanner.total_combos()}")
    df=scanner.scan(verbose=True)
    scanner.print_leaderboard(df,n=25)
    out=REPORT_DIR/"param_scan_results.csv"
    cols=[c for c in df.columns if c not in ("equity","param_dict")]
    df[cols].to_csv(out)
    print(f"  Guardado: {out}")
    return df


def run_optuna_scan(regimes=None, n_seeds=None, days=None, n_trials=None, timeout=None):
    from src.models.optuna_scanner import OptunaScanner
    from config.settings import ScanConfig
    cfg=ScanConfig(n_seeds=n_seeds or SCAN_CFG.n_seeds,
                    days=days or SCAN_CFG.days,
                    regimes=regimes or SCAN_CFG.regimes,
                    optuna_trials=n_trials or SCAN_CFG.optuna_trials,
                    optuna_timeout_sec=timeout or SCAN_CFG.optuna_timeout_sec)
    scanner=OptunaScanner(cfg)
    df=scanner.optimize(verbose=True)
    scanner.print_leaderboard(df,n=20)
    out=REPORT_DIR/"optuna_scan_results.csv"
    cols=[c for c in df.columns if c not in ("equity","param_dict")]
    df[cols].to_csv(out)
    print(f"  Guardado: {out}")
    return df


def run_signal_investigation(signal_name="momentum_12_1", hypothesis="",
                               tickers=None, start=None, end=None,
                               horizon=21, m_trials=1):
    tickers=tickers or UNIVERSE_CFG.tickers[:20]
    start=start or "2015-01-01"; end=end or "2024-12-31"
    HYP_MAP={
        "momentum_12_1":"Momentum 12-1 (Jegadeesh & Titman 1993)",
        "momentum_3m":"Momentum 3 meses",
        "trend_ma200":"Tendencia vs MA200",
        "zscore_60d":"Mean-reversion Z-score 60 dias",
        "rsi_14":"RSI 14 senal de reversion continua",
        "low_vol":"Low-volatility anomaly",
        "trend_rsi_lowvol":"Tendencia con confirmacion RSI y filtro de baja volatilidad",
        "alpha_trend_quality":"Momentum de calidad: tendencia persistente ajustada por volatilidad",
        "alpha_pullback_trend":"Comprar retrocesos cortos dentro de tendencia alcista",
        "alpha_compression_breakout":"Breakout desde compresion de volatilidad",
        "alpha_pro_multi":"Alpha compuesto profesional de tendencia, pullback y breakout",
    }
    hyp=hypothesis or HYP_MAP.get(signal_name,"Hipotesis no documentada")
    from src.ingestion.pipeline import IngestionPipeline
    IngestionPipeline().run(tickers=tickers,start=start,end=end)
    _,prices=_get_prices(tickers,start,end)
    if prices.empty:
        print("  Sin datos. Revisa la descarga."); return
    signal=_build_signal(prices,signal_name)
    from src.models.edge_finder import EdgeFinder
    rpt=EdgeFinder().investigate(signal=signal,prices=prices,
                                  signal_name=signal_name,hypothesis=hyp,
                                  horizon=horizon,m_trials=m_trials)
    if rpt.decay_df is not None:
        print("  SIGNAL DECAY:")
        print(rpt.decay_df[["ic_mean","icir","viable"]].to_string())
    return rpt


def run_full_pipeline(tickers=None, start=None, end=None):
    tickers=tickers or UNIVERSE_CFG.tickers
    start=start or UNIVERSE_CFG.start_date
    end=end or UNIVERSE_CFG.end_date
    print(f"\n{'='*60}\n  FULL PIPELINE\n{'='*60}")
    from src.ingestion.pipeline import IngestionPipeline
    quality=IngestionPipeline().run(tickers=tickers,start=start,end=end)
    good=tickers[:10]
    if "passed" in quality.columns and "ticker" in quality.columns:
        passed=quality[quality["passed"]==True]
        good=passed["ticker"].tolist() if len(passed)>0 else tickers[:10]
    from config.settings import ScanConfig
    from src.models.param_scanner import ParamScanner, FAMILIES
    cfg=ScanConfig(n_seeds=3,regimes=["normal","trending","ranging"])
    scan_df=ParamScanner(cfg).scan(verbose=True)
    _,prices=_get_prices(good,start,end)
    signals={nm:_build_signal(prices,nm) for nm in
             ["momentum_12_1","zscore_60d","low_vol","trend_ma200","rsi_14",
              "trend_rsi_lowvol","alpha_trend_quality","alpha_pullback_trend",
              "alpha_compression_breakout","alpha_pro_multi"]
             if not prices.empty}
    batch = pd.DataFrame()
    if signals:
        from src.models.edge_finder import EdgeFinder
        batch=EdgeFinder().batch_investigate(
            signals=signals,prices=prices,
            hypotheses={"momentum_12_1":"Momentum","zscore_60d":"ZScore MR",
                        "low_vol":"Low Vol","trend_ma200":"Trend MA200",
                        "rsi_14":"RSI Rev","trend_rsi_lowvol":"Trend + RSI + Low Vol",
                        "alpha_trend_quality":"Trend quality alpha",
                        "alpha_pullback_trend":"Pullback in trend alpha",
                        "alpha_compression_breakout":"Compression breakout alpha",
                        "alpha_pro_multi":"Professional multi-alpha stack"})
    else:
        print("  Sin precios suficientes para batch_investigate; se omite esa etapa.")
    _save_report({},scan_df,"full_pipeline_report.txt",batch=batch)
    return scan_df,batch


def run_portfolio_alpha_strategy(alpha_name="alpha_pro_multi", tickers=None, start=None, end=None):
    tickers = tickers or DEFAULT_TEST_TICKERS
    start = start or UNIVERSE_CFG.start_date
    end = end or UNIVERSE_CFG.end_date

    print(f"\n{'='*60}")
    print(f"  PORTFOLIO ALPHA STRATEGY")
    print(f"{'='*60}")
    print(f"  Alpha:   {alpha_name}")
    print(f"  Tickers: {tickers}")
    print(f"  Periodo: {start} -> {end}")

    from src.ingestion.pipeline import IngestionPipeline
    IngestionPipeline().run(tickers=tickers, start=start, end=end)

    _, prices = _get_prices(tickers, start, end)
    if prices.empty or prices.shape[1] < 4:
        print("  No hay suficientes precios para construir una cartera.")
        return None

    alpha_signal = _build_signal(prices, alpha_name)
    from src.portfolio.backtester import AlphaPortfolioBacktester
    result = AlphaPortfolioBacktester().run(prices, alpha_signal, alpha_name=alpha_name)
    result.print_summary()

    equity_path = REPORT_DIR / f"{alpha_name}_portfolio_equity.csv"
    weights_path = REPORT_DIR / f"{alpha_name}_portfolio_weights.csv"
    report_path = REPORT_DIR / f"{alpha_name}_portfolio_report.txt"

    out_df = pd.concat([result.equity, result.benchmark_equity, result.returns, result.benchmark_returns], axis=1)
    out_df.to_csv(equity_path)
    result.weights.to_csv(weights_path)
    _save_portfolio_report(result, report_path)

    print(f"  Equity:   {equity_path}")
    print(f"  Weights:  {weights_path}")
    print(f"  Reporte:  {report_path}")
    return result


def _save_report(reports, scan_df, filename, batch=None):
    lines=["="*60,"  QUANT RESEARCH FRAMEWORK — REPORTE","="*60,""]
    if reports:
        lines.append("SENALES INVESTIGADAS:")
        for name,r in reports.items():
            lines.append(f"  {name}: {r.verdict}")
        lines.append("")
    if scan_df is not None and len(scan_df)>0:
        lines.append("TOP 10 PARAM SCAN:")
        cols=[c for c in ["family","params","sharpe","mdd","score","bias_robust"] if c in scan_df.columns]
        lines.append(scan_df[cols].head(10).to_string())
        lines.append("")
    if batch is not None and len(batch)>0:
        lines.append("BATCH IC:")
        cols=[c for c in ["signal","ic_mean","icir","bh_significant"] if c in batch.columns]
        lines.append(batch[cols].to_string())
    path=REPORT_DIR/filename
    path.write_text("\n".join(lines),encoding="utf-8")
    print(f"  Reporte: {path}")


def _save_portfolio_report(result, path):
    report = result.report
    lines = [
        "=" * 60,
        f"  PORTFOLIO REPORT | {result.alpha_name}",
        "=" * 60,
        "",
        f"rebalance_frequency: {result.meta.get('rebalance_frequency')}",
        f"long_short:          {result.meta.get('long_short')}",
        f"top_n:               {result.meta.get('top_n')}",
        f"bottom_n:            {result.meta.get('bottom_n')}",
        "",
    ]
    for key in [
        "total_return", "benchmark_total_return", "cagr", "vol_annual", "sharpe",
        "sortino", "max_drawdown", "calmar", "var_95", "cvar_95", "beta", "ir",
        "avg_turnover", "avg_active_names", "avg_gross_exposure", "avg_net_exposure",
        "rebalance_count",
    ]:
        if key in report:
            lines.append(f"{key}: {report[key]}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "demo"
    if mode=="demo": run_demo()
    elif mode=="scan": run_param_scan()
    elif mode=="optuna": run_optuna_scan()
    elif mode=="full": run_full_pipeline()
    elif mode=="portfolio":
        nm=sys.argv[2] if len(sys.argv)>2 else "alpha_pro_multi"
        run_portfolio_alpha_strategy(nm)
    elif mode=="signal":
        nm=sys.argv[2] if len(sys.argv)>2 else "momentum_12_1"
        run_signal_investigation(nm)
