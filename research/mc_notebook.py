"""
MONTE CARLO NOTEBOOK
compare_price_models, mc_strategy_vs_bnh, mc_bias_validation,
mc_portfolio_frontier, mc_sensitivity, mc_historical_bootstrap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from config.settings import SimParams, REPORT_DIR
from src.montecarlo.simulator import MonteCarloEngine, MCResults


def compare_price_models(n_paths=1000, n_days=252):
    print(f"\n{'='*60}")
    print(f"  COMPARACION DE MODELOS MC | {n_paths} paths x {n_days} dias")
    print(f"{'='*60}\n")
    params=SimParams(n_paths=n_paths,n_days=n_days,mu=0.08,sigma=0.20,seed=42)
    engine=MonteCarloEngine()
    results=engine.compare_models(params,["gbm","merton","heston","regime","garch"])
    print(f"  {'Modelo':<12} {'CAGR%':>8} {'P5%':>8} {'P95%':>8} {'VaR95%':>8} {'MDD%':>8} {'Skew':>7}")
    print("  "+"─"*65)
    for m,res in results.items():
        mt=res.metrics
        print(f"  {m:<12} {mt['mean_cagr']*100:>8.2f} {mt['p5_cagr']*100:>8.2f} "
              f"{mt['p95_cagr']*100:>8.2f} {mt['var_95']*100:>8.2f} "
              f"{mt['mean_mdd']*100:>8.2f} {mt['skewness']:>7.3f}")
    return results


def mc_strategy_vs_bnh(n_paths=500, n_days=504):
    print(f"\n{'='*60}")
    print(f"  ESTRATEGIA vs BUY-AND-HOLD | {n_paths} paths x {n_days} dias")
    print(f"{'='*60}")
    params=SimParams(n_paths=n_paths,n_days=n_days,mu=0.10,sigma=0.22,seed=42)
    engine=MonteCarloEngine()

    def _ema_np(p, w):
        k=2.0/(w+1); e=np.empty(len(p)); e[0]=p[0]
        for i in range(1,len(p)): e[i]=p[i]*k+e[i-1]*(1-k)
        return e

    def ma_cross(prices):
        f=_ema_np(prices,10); s=_ema_np(prices,50)
        sig=np.zeros(len(prices))
        for i in range(50,len(prices)):
            sig[i]=1.0 if f[i-1]>s[i-1] else -1.0
        return sig

    results=engine.run_strategy_mc(ma_cross,params,"gbm",5.0)
    bnh=results["buy_and_hold"]; strat=results["strategy"]
    print(f"\n  {'Metrica':<28} {'BnH':>12} {'MA Cross':>12}")
    print("  "+"─"*55)
    for label,bv,sv in [
        ("CAGR medio",bnh.metrics["mean_cagr"],strat.metrics["mean_cagr"]),
        ("CAGR P5",bnh.metrics["p5_cagr"],strat.metrics["p5_cagr"]),
        ("CAGR P95",bnh.metrics["p95_cagr"],strat.metrics["p95_cagr"]),
        ("VaR 95%",bnh.metrics["var_95"],strat.metrics["var_95"]),
        ("CVaR 95%",bnh.metrics["cvar_95"],strat.metrics["cvar_95"]),
        ("P(perdida)",bnh.metrics["prob_loss"],strat.metrics["prob_loss"]),
        ("MDD medio",bnh.metrics["mean_mdd"],strat.metrics["mean_mdd"]),
    ]:
        print(f"  {label:<28} {bv*100:>10.2f}%  {sv*100:>10.2f}%")
    beat=float(np.mean(strat.final_rets>bnh.final_rets[:len(strat.final_rets)]))
    print(f"\n  P(estrategia > BnH): {beat*100:.1f}%")
    return results


def mc_bias_validation(observed_sharpe=1.5, n_sim=300, n_days=504):
    print(f"\n{'='*60}")
    print(f"  MC BIAS VALIDATION | Sharpe obs={observed_sharpe} | {n_sim} sims")
    print(f"{'='*60}")
    params=SimParams(n_paths=n_sim,n_days=n_days,mu=0.08,sigma=0.20,seed=42)
    engine=MonteCarloEngine()

    def _ema_np(p, w):
        k=2.0/(w+1); e=np.empty(len(p)); e[0]=p[0]
        for i in range(1,len(p)): e[i]=p[i]*k+e[i-1]*(1-k)
        return e

    def ma_cross(prices):
        f=_ema_np(prices,10); s=_ema_np(prices,50)
        sig=np.zeros(len(prices))
        for i in range(50,len(prices)): sig[i]=1.0 if f[i-1]>s[i-1] else -1.0
        return sig

    mc=engine.run("gbm",params); cost=5/10000; sim_srs=[]
    for i in range(n_sim):
        prices=mc.paths[i]; sig=ma_cross(prices); rets=[]; pos=0.0
        for t in range(1,len(prices)):
            np_=float(sig[t]); pr=(prices[t]-prices[t-1])/(prices[t-1]+1e-10)
            tc=abs(np_-pos)*cost; rets.append(pos*pr-tc); pos=np_
        r=np.array(rets); mu_r=r.mean(); sig_r=r.std()+1e-10
        sim_srs.append(float(mu_r/sig_r*np.sqrt(252)))

    sims=np.array(sim_srs); pct=float(np.mean(sims<observed_sharpe))
    print(f"\n  Sharpe observado:    {observed_sharpe:.4f}")
    print(f"  Media Sharpe (sim):  {sims.mean():.4f}")
    print(f"  P5  Sharpe (sim):    {np.percentile(sims,5):.4f}")
    print(f"  P95 Sharpe (sim):    {np.percentile(sims,95):.4f}")
    print(f"  Percentil rank:      {pct*100:.1f}%")
    print(f"  P-value:             {1-pct:.4f}")
    sig_="SI ✓" if pct>0.95 else "NO ✗"
    print(f"  Significativo:       {sig_}")
    verdict=("VENTAJA REAL" if pct>0.95 else "POSIBLE AZAR")
    print(f"  VEREDICTO: {verdict} — percentil {pct*100:.0f}%")
    return {"observed":observed_sharpe,"percentile":pct*100,"significant":pct>0.95}


def mc_sensitivity_analysis(n_paths=500):
    print(f"\n{'='*60}")
    print(f"  ANALISIS DE SENSIBILIDAD MC")
    print(f"{'='*60}")
    base=SimParams(n_paths=n_paths,n_days=252,seed=42)
    engine=MonteCarloEngine()
    analyses=[
        ("gbm","sigma",[0.10,0.15,0.20,0.25,0.30,0.40]),
        ("gbm","mu",[-0.05,0.00,0.05,0.08,0.12,0.20]),
        ("merton","jump_lambda",[1.0,3.0,5.0,10.0,20.0]),
        ("heston","xi",[0.1,0.2,0.3,0.5,0.8]),
    ]
    for model,param,values in analyses:
        print(f"\n  [{model.upper()}] Sensibilidad a '{param}':")
        print(f"  {'─'*60}")
        rows=[]
        for v in values:
            kw={**{f:getattr(base,f) for f in base.__dataclass_fields__},param:v}
            p2=SimParams(**kw)
            res=engine.run(model,p2); m=res.metrics
            rows.append({param:v,"CAGR%":round(m["mean_cagr"]*100,2),
                          "P5%":round(m["p5_cagr"]*100,2),"VaR%":round(m["var_95"]*100,2),
                          "MDD%":round(m["mean_mdd"]*100,2),"P(loss)%":round(m["prob_loss"]*100,1)})
        print(f"  {pd.DataFrame(rows).set_index(param).to_string()}")


def run_full_mc_report(n_paths=500, n_days=252):
    print(f"\n{'='*60}\n  MONTE CARLO FULL REPORT\n{'='*60}")
    print("\n[1/4] Comparando modelos...")
    compare_price_models(n_paths, n_days)
    print("\n[2/4] Estrategia vs BnH...")
    mc_strategy_vs_bnh(min(n_paths,300), n_days)
    print("\n[3/4] Bias validation...")
    mc_bias_validation(1.5, min(n_paths,200), n_days)
    print("\n[4/4] Sensibilidad...")
    mc_sensitivity_analysis(min(n_paths,300))
    path=REPORT_DIR/"mc_report.txt"
    path.write_text("Monte Carlo report completado.\n",encoding="utf-8")
    print(f"\n  Reporte: {path}\n")


if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "all"
    if mode=="models": compare_price_models()
    elif mode=="strategy": mc_strategy_vs_bnh()
    elif mode=="bias": mc_bias_validation()
    elif mode=="sensitivity": mc_sensitivity_analysis()
    elif mode=="all": run_full_mc_report()
    else: print("Uso: python mc_notebook.py [models|strategy|bias|sensitivity|all]")
