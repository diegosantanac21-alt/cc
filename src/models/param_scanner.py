"""
GRID SEARCH DE PARAMETROS
12 familias x sus grids x semillas x regimenes
"""
import logging
from dataclasses import dataclass, field
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config.settings import SCAN_CFG

logger = logging.getLogger("scanner")


def _lcg(seed):
    x = seed & 0xFFFFFFFF
    def _r():
        nonlocal x
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        return x / 0xFFFFFFFF
    return _r

def _rn(r):
    u1 = r() + 1e-10
    return float(np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*r()))

def generate_prices(days, mu_a, sig_a, regime, seed):
    rand  = _lcg(seed * 7919 + 3571)
    mu    = mu_a / 252
    sigma = sig_a / np.sqrt(252)
    p     = [100.0]
    for i in range(1, days):
        t = i / days
        d, v = mu, sigma
        if regime == "trending":
            d = mu*(1+0.9*np.sin(t*np.pi)); v = sigma*0.65
        elif regime == "ranging":
            d = mu*0.15; v = sigma*1.1
        elif regime == "crash":
            if 0.35<t<0.50: d=-0.003; v=sigma*3
            elif 0.50<t<0.70: d=mu*1.5; v=sigma*1.5
        elif regime == "volatile":
            v = sigma*(1.6+np.sin(i*0.04)*0.9)
        z = _rn(rand)
        p.append(p[-1]*np.exp(d + v*z))
    return np.array(p)

def _np_ema(p, w):
    k = 2.0/(w+1); e = np.empty(len(p)); e[0]=p[0]
    for i in range(1,len(p)): e[i]=p[i]*k+e[i-1]*(1-k)
    return e

def _np_sma(p, w):
    out = np.full(len(p), np.nan)
    for i in range(w-1, len(p)):
        out[i] = p[i-w+1:i+1].mean()
    return out

def _np_std(p, w):
    out = np.full(len(p), np.nan)
    for i in range(w-1, len(p)):
        out[i] = p[i-w+1:i+1].std()
    return out

def _np_rsi(p, w):
    n=len(p); gains=np.maximum(np.diff(p),0); losses=np.maximum(-np.diff(p),0)
    rsi=np.full(n,np.nan)
    if n<=w: return rsi
    ag=gains[:w].mean(); al=losses[:w].mean()
    for i in range(w,n-1):
        ag=(ag*(w-1)+gains[i])/w; al=(al*(w-1)+losses[i])/w
        rsi[i+1]=100-100/(1+ag/(al+1e-10))
    return rsi


# Funciones de señal
def _s_ma(p,fast,slow):
    if fast>=slow: return None
    f=_np_ema(p,fast); s=_np_ema(p,slow)
    return np.where(np.roll(f,1)>np.roll(s,1),1.0,-1.0)

def _s_triple(p,s,m,l):
    if s>=m or m>=l: return None
    e1=_np_ema(p,s); e2=_np_ema(p,m); e3=_np_ema(p,l)
    p1=np.roll(e1,1); p2=np.roll(e2,1); p3=np.roll(e3,1)
    return np.where((p1>p2)&(p2>p3),1.0,np.where((p1<p2)&(p2<p3),-1.0,0.0))

def _s_rsi_rev(p,w,lo,hi):
    r=_np_rsi(p,w); sig=np.zeros(len(p)); pos=0
    for i in range(1,len(p)):
        if not np.isnan(r[i]):
            if r[i]<lo and pos<=0: pos=1
            elif r[i]>hi and pos>=0: pos=-1
        sig[i]=pos
    return sig

def _s_rsi_trend(p,w,th,maW):
    r=_np_rsi(p,w); m=_np_sma(p,maW); sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(r[i]) or np.isnan(m[i]): continue
        sig[i]=1.0 if(p[i-1]>m[i] and r[i]>th) else(-1.0 if(p[i-1]<m[i] and r[i]<100-th) else 0.0)
    return sig

def _s_bb_rev(p,w,k):
    m=_np_sma(p,w); s=_np_std(p,w); pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(m[i]): continue
        lo=m[i]-k*s[i]; hi=m[i]+k*s[i]; pv=p[i-1]
        if pv<lo and pos<=0: pos=1
        elif pv>hi and pos>=0: pos=-1
        elif abs(pv-m[i])<s[i]*0.25: pos=0
        sig[i]=pos
    return sig

def _s_bb_brk(p,w,k):
    m=_np_sma(p,w); s=_np_std(p,w); pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(m[i]): continue
        pv=p[i-1]
        if pv>m[i]+k*s[i] and pos<=0: pos=1
        elif pv<m[i]-k*s[i] and pos>=0: pos=-1
        sig[i]=pos
    return sig

def _s_macd(p,f,s,sp):
    if f>=s: return None
    ef=_np_ema(p,f); es=_np_ema(p,s); mac=ef-es; sl=_np_ema(mac,sp)
    return np.where(np.roll(mac,1)>np.roll(sl,1),1.0,-1.0)

def _s_mom(p,lb,sk):
    sig=np.zeros(len(p)); lbt=lb+sk
    if lbt>=len(p): return None
    for i in range(lbt,len(p)):
        sig[i]=1.0 if p[i-1]>p[i-lbt+sk] else -1.0
    return sig

def _s_zscore(p,w,thr):
    m=_np_sma(p,w); s=_np_std(p,w); pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(m[i]): continue
        z=(p[i-1]-m[i])/(s[i]+1e-10)
        if z<-thr and pos<=0: pos=1
        elif z>thr and pos>=0: pos=-1
        elif abs(z)<0.3: pos=0
        sig[i]=pos
    return sig

def _s_ma_rsi(p,fast,slow,rsiW,rsiTh):
    if fast>=slow: return None
    f=_np_ema(p,fast); s=_np_ema(p,slow); r=_np_rsi(p,rsiW)
    pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(r[i]): continue
        if f[i-1]>s[i-1] and r[i]>rsiTh and pos<=0: pos=1
        elif f[i-1]<s[i-1] and r[i]<100-rsiTh and pos>=0: pos=-1
        sig[i]=pos
    return sig

def _s_bb_rsi(p,bbW,bbK,rsiW):
    m=_np_sma(p,bbW); s=_np_std(p,bbW); r=_np_rsi(p,rsiW)
    pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(m[i]) or np.isnan(r[i]): continue
        pv=p[i-1]; lo=m[i]-bbK*s[i]; hi=m[i]+bbK*s[i]
        if pv<lo and r[i]<35 and pos<=0: pos=1
        elif pv>hi and r[i]>65 and pos>=0: pos=-1
        sig[i]=pos
    return sig

def _s_macd_rsi(p,f,s,sp,rsiW):
    if f>=s: return None
    ef=_np_ema(p,f); es=_np_ema(p,s); mac=ef-es; sl=_np_ema(mac,sp); r=_np_rsi(p,rsiW)
    pos=0; sig=np.zeros(len(p))
    for i in range(1,len(p)):
        if np.isnan(r[i]): continue
        up=mac[i-1]>sl[i-1]
        if up and r[i]>50 and pos<=0: pos=1
        elif not up and r[i]<50 and pos>=0: pos=-1
        sig[i]=pos
    return sig

def _s_trend_rsi_vol(p, fast, slow, rsiW, volW, volThr):
    if fast >= slow:
        return None
    fast_ma = _np_ema(p, fast)
    slow_ma = _np_ema(p, slow)
    rsi = _np_rsi(p, rsiW)
    rets = np.diff(np.log(p + 1e-10), prepend=np.log(max(p[0], 1e-10)))
    vol = np.full(len(p), np.nan)
    for i in range(volW, len(p)):
        vol[i] = np.std(rets[i - volW + 1:i + 1]) * np.sqrt(252)
    sig = np.zeros(len(p))
    pos = 0
    for i in range(1, len(p)):
        if np.isnan(rsi[i]) or np.isnan(vol[i]):
            continue
        trend_up = fast_ma[i - 1] > slow_ma[i - 1]
        trend_down = fast_ma[i - 1] < slow_ma[i - 1]
        low_vol = vol[i] <= volThr
        if trend_up and low_vol and 50 <= rsi[i] <= 72 and pos <= 0:
            pos = 1
        elif trend_down and low_vol and 28 <= rsi[i] <= 50 and pos >= 0:
            pos = -1
        elif not low_vol or (rsi[i] > 78) or (rsi[i] < 22):
            pos = 0
        sig[i] = pos
    return sig


@dataclass
class StrategyFamily:
    id: str
    name: str
    color: str
    params: List[dict]
    signal_fn: object = field(repr=False)

    def all_combos(self):
        names  = [p["name"]   for p in self.params]
        values = [p["values"] for p in self.params]
        for combo in product(*values):
            yield dict(zip(names, combo))

    def n_combos(self):
        n = 1
        for p in self.params: n *= len(p["values"])
        return n


FAMILIES = [
    StrategyFamily("ma_cross",  "MA Cross",      "#00d4ff",
        [{"name":"fast","values":[5,8,10,13,21]},
         {"name":"slow","values":[20,30,40,50,60,100]}], _s_ma),
    StrategyFamily("triple_ma", "Triple MA",     "#64ffda",
        [{"name":"s","values":[5,8,10,13]},
         {"name":"m","values":[21,30,40,50]},
         {"name":"l","values":[60,100,150,200]}], _s_triple),
    StrategyFamily("rsi_rev",   "RSI Mean Rev",  "#00e676",
        [{"name":"w","values":[7,9,14,21,28]},
         {"name":"lo","values":[20,25,30,35]},
         {"name":"hi","values":[65,70,75,80]}], _s_rsi_rev),
    StrategyFamily("rsi_trend", "RSI Trend",     "#76ff03",
        [{"name":"w","values":[9,14,21]},
         {"name":"th","values":[45,50,55]},
         {"name":"maW","values":[20,50,100]}], _s_rsi_trend),
    StrategyFamily("bb_rev",    "BB Rev",        "#ff6d00",
        [{"name":"w","values":[10,15,20,30,40]},
         {"name":"k","values":[1.5,2.0,2.5,3.0]}], _s_bb_rev),
    StrategyFamily("bb_break",  "BB Breakout",   "#ff9100",
        [{"name":"w","values":[10,20,30,50]},
         {"name":"k","values":[1.0,1.5,2.0,2.5]}], _s_bb_brk),
    StrategyFamily("macd",      "MACD",          "#e040fb",
        [{"name":"f","values":[8,12,16]},
         {"name":"s","values":[21,26,34]},
         {"name":"sig","values":[7,9,12]}], _s_macd),
    StrategyFamily("momentum",  "Momentum",      "#f48fb1",
        [{"name":"lb","values":[5,10,20,40,60,80,120]},
         {"name":"sk","values":[0,1,5,10]}], _s_mom),
    StrategyFamily("zscore",    "Z-Score Rev",   "#80deea",
        [{"name":"w","values":[20,40,60,80,120]},
         {"name":"thr","values":[1.0,1.5,2.0,2.5]}], _s_zscore),
    StrategyFamily("ma_rsi",    "MA + RSI",      "#ffcc02",
        [{"name":"fast","values":[5,10,15,20]},
         {"name":"slow","values":[30,50,80,100]},
         {"name":"rsiW","values":[9,14,21]},
         {"name":"rsiTh","values":[45,50,55]}], _s_ma_rsi),
    StrategyFamily("bb_rsi",    "BB + RSI",      "#b9f6ca",
        [{"name":"bbW","values":[15,20,30]},
         {"name":"bbK","values":[1.5,2.0,2.5]},
         {"name":"rsiW","values":[9,14,21]}], _s_bb_rsi),
    StrategyFamily("macd_rsi",  "MACD + RSI",    "#cf6679",
        [{"name":"f","values":[8,12]},
         {"name":"s","values":[21,26]},
         {"name":"sp","values":[7,9]},
         {"name":"rsiW","values":[9,14]}], _s_macd_rsi),
    StrategyFamily("trend_rsi_vol", "Trend + RSI + Vol", "#4db6ac",
        [{"name":"fast","values":[10,20,30]},
         {"name":"slow","values":[50,100,150]},
         {"name":"rsiW","values":[7,14,21]},
         {"name":"volW","values":[10,21,42]},
         {"name":"volThr","values":[0.18,0.25,0.35]}], _s_trend_rsi_vol),
]


def run_backtest(prices, signal, cost_bps=None, cfg=None):
    cfg = cfg or SCAN_CFG
    total_cost_bps=float(cfg.cost_bps if cost_bps is None else cost_bps)
    total_cost_bps=max(total_cost_bps,0.0)
    commission_bps=min(total_cost_bps, float(cfg.commission_bps))
    slippage_bps=max(total_cost_bps-commission_bps,0.0)
    if cost_bps is None and total_cost_bps == 0.0:
        commission_bps=float(cfg.commission_bps)
        slippage_bps=float(cfg.slippage_bps)
        total_cost_bps=commission_bps+slippage_bps
    total_cost=total_cost_bps/10000

    execution_lag=max(int(cfg.execution_lag),0)
    target_vol=max(float(cfg.target_vol_annual),1e-6)
    max_leverage=max(float(cfg.max_leverage),0.0)
    stop_loss_pct=max(float(cfg.stop_loss_pct),0.0)
    take_profit_pct=max(float(cfg.take_profit_pct),0.0)

    prices=np.asarray(prices,dtype=float)
    signal=np.asarray(signal,dtype=float)
    n=len(prices)
    equity=np.ones(n); rets=np.zeros(n-1); pos=0.0; trades=0
    turnover_sum=0.0; trade_returns=[]; trade_holding=[]; current_trade=[]
    entry_price=None
    for i in range(1,n):
        sig_idx=max(i-execution_lag,0)
        raw_sig=float(signal[sig_idx]) if not np.isnan(signal[sig_idx]) else 0.0
        if i >= 21:
            recent=np.diff(np.log(prices[max(0,i-21):i]))
            realized_vol=float(np.std(recent)*np.sqrt(252)) if len(recent) else target_vol
        else:
            realized_vol=target_vol
        size_scalar=min(target_vol/max(realized_vol,1e-6), max_leverage)
        target_pos=float(np.clip(raw_sig,-1.0,1.0))*size_scalar
        pr=(prices[i]-prices[i-1])/(prices[i-1]+1e-10)
        pnl_from_entry=0.0
        if pos != 0.0 and entry_price:
            pnl_from_entry=((prices[i-1]/entry_price)-1.0)*np.sign(pos)
        stop_triggered=pos != 0.0 and stop_loss_pct > 0 and pnl_from_entry <= -stop_loss_pct
        take_triggered=pos != 0.0 and take_profit_pct > 0 and pnl_from_entry >= take_profit_pct
        new_pos=0.0 if (stop_triggered or take_triggered) else target_pos
        tc=abs(new_pos-pos)*total_cost
        r=pos*pr-tc
        turnover_sum += abs(new_pos-pos)
        if pos != 0.0:
            current_trade.append(r)
        if new_pos!=pos:
            trades+=1
            if pos != 0.0 and current_trade:
                trade_returns.append(float(np.prod(1 + np.array(current_trade)) - 1))
                trade_holding.append(len(current_trade))
                current_trade=[]
            if new_pos != 0.0:
                entry_price=float(prices[i])
            else:
                entry_price=None
        equity[i]=equity[i-1]*(1+r)
        rets[i-1]=r
        pos=new_pos
    if pos != 0.0 and current_trade:
        trade_returns.append(float(np.prod(1 + np.array(current_trade)) - 1))
        trade_holding.append(len(current_trade))
    mu=rets.mean(); sigma=rets.std()+1e-10
    sharpe=(mu/sigma)*np.sqrt(252)
    down=rets[rets<0]; dsig=np.sqrt((down**2).mean()) if len(down)>1 else 1e-10
    sortino=(mu/dsig)*np.sqrt(252)
    peak=np.maximum.accumulate(equity); mdd=(equity/peak-1).min()
    yrs=(n-1)/252; cagr=equity[-1]**(1/max(yrs,1e-3))-1
    calmar=cagr/abs(mdd) if abs(mdd)>1e-6 else 0.0
    wr=(rets>0).mean(); tpd=trades/(n-1)
    trade_arr=np.array(trade_returns,dtype=float) if trade_returns else np.array([],dtype=float)
    wins=trade_arr[trade_arr>0]; losses=trade_arr[trade_arr<0]
    avg_win=float(wins.mean()) if len(wins) else 0.0
    avg_loss=float(losses.mean()) if len(losses) else 0.0
    hit_ratio=float((trade_arr>0).mean()) if len(trade_arr) else 0.0
    hold_days=float(np.mean(trade_holding)) if trade_holding else 0.0
    turnover=float(turnover_sum/max(n-1,1))
    turnover_ann=float(turnover*252)
    win_loss_ratio=float(avg_win/abs(avg_loss)) if avg_loss<0 else np.inf
    return {"sharpe":float(sharpe),"sortino":float(sortino),"mdd":float(mdd),
            "cagr":float(cagr),"calmar":float(calmar),"wr":float(wr),
            "tpd":float(tpd),"n_trades":int(len(trade_returns)),
            "turnover":turnover,"turnover_annual":turnover_ann,
            "hit_ratio":hit_ratio,"avg_win":avg_win,"avg_loss":avg_loss,
            "win_loss_ratio":win_loss_ratio,"holding_period":hold_days,
            "commission_bps":float(commission_bps),
            "slippage_bps":float(slippage_bps),
            "execution_lag":int(execution_lag),
            "target_vol":float(target_vol),
            "max_leverage":float(max_leverage),
            "stop_loss_pct":float(stop_loss_pct),
            "take_profit_pct":float(take_profit_pct),
            "skewness":float(scipy_stats.skew(rets)) if len(rets) > 2 else 0.0,
            "kurtosis":float(scipy_stats.kurtosis(rets)) if len(rets) > 3 else 0.0,
            "var_95":float(np.percentile(rets,5)) if len(rets) else 0.0,
            "cvar_95":float(rets[rets<=np.percentile(rets,5)].mean()) if len(rets) else 0.0,
            "equity":equity}


def composite_score(r):
    sharpe_score = min(max(r["sharpe"], 0.0), 3.0) / 3.0
    dd_score = max(1.0 - min(abs(r["mdd"]) / 0.20, 1.0), 0.0)
    calmar_score = min(max(r["calmar"], 0.0), 3.0) / 3.0
    hit_score = min(max(r["hit_ratio"], 0.0), 1.0)

    # Penaliza estrategias demasiado activas y favorece holding periods mas swing-friendly.
    turnover_score = max(1.0 - min(r["turnover"] / 0.08, 1.0), 0.0)
    trade_freq_score = max(1.0 - min(r["tpd"] / 0.35, 1.0), 0.0)
    holding_score = min(max((r["holding_period"] - 2.0) / 8.0, 0.0), 1.0)

    return (
        0.28 * sharpe_score +
        0.22 * dd_score +
        0.18 * calmar_score +
        0.10 * hit_score +
        0.10 * turnover_score +
        0.06 * trade_freq_score +
        0.06 * holding_score
    )


def bias_check(sharpe_arr):
    arr=np.array(sharpe_arr); mu=arr.mean(); sd=arr.std()
    cv=sd/(abs(mu)+1e-10); ppp=(arr>0.3).mean()
    return {"mu_sharpe":float(mu),"cv_sharpe":float(cv),
            "pct_pos":float(ppp),"robust":bool(mu>0.15 and cv<0.9 and ppp>0.55)}


class ParamScanner:

    def __init__(self, cfg=None):
        self.cfg = cfg or SCAN_CFG

    def total_combos(self, families=None):
        return sum(f.n_combos() for f in (families or FAMILIES))

    def _sort_results(self, df):
        if len(df) == 0:
            return df
        sort_cols = [c for c in ["bias_robust", "score", "sharpe", "calmar", "holding_period"] if c in df.columns]
        ascending = [False] * len(sort_cols)
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    def _style_tag(self, row):
        hold = float(row.get("holding_period", 0.0))
        turnover = float(row.get("turnover", 0.0))
        tpd = float(row.get("tpd", 0.0))
        if hold >= 5.0 and turnover <= 0.08 and tpd <= 0.25:
            return "SWING"
        if hold >= 2.0 and turnover <= 0.12 and tpd <= 0.40:
            return "BALANCED"
        return "FAST"

    def scan(self, families=None, verbose=True):
        fams=families or FAMILIES
        n_seeds=self.cfg.n_seeds; days=self.cfg.days
        mu_a=self.cfg.drift; sig_a=self.cfg.volatility
        regimes=self.cfg.regimes; cost_bps=self.cfg.cost_bps
        total=self.total_combos(fams); results=[]; done=0

        print(f"\n{'='*60}")
        print(f"  PARAM SCANNER  |  {total} combinaciones  |  {n_seeds} seeds x {len(regimes)} regimenes")
        print(f"{'='*60}\n")

        for fam in fams:
            fam_results = []
            for combo in fam.all_combos():
                sharpe_seeds=[]; best_bt=None
                for seed in range(1, n_seeds+1):
                    for reg in regimes:
                        prices=generate_prices(days,mu_a,sig_a,reg,seed*997+done*13)
                        sig=fam.signal_fn(prices,**combo)
                        if sig is None: continue
                        bt=run_backtest(prices,sig,cost_bps,cfg=self.cfg)
                        sharpe_seeds.append(bt["sharpe"])
                        if best_bt is None or bt["sharpe"]>best_bt["sharpe"]:
                            best_bt=bt
                if not best_bt or len(sharpe_seeds)<2:
                    done+=1; continue
                bias=bias_check(sharpe_seeds); score=composite_score(best_bt)
                label=", ".join(f"{k}={v}" for k,v in combo.items())
                row={"family":fam.name,"fam_id":fam.id,"color":fam.color,
                     "params":label,**best_bt,"score":round(score,4),
                     "bias_robust":bias["robust"],"bias_cv":round(bias["cv_sharpe"],3),
                     "bias_pct_pos":round(bias["pct_pos"],3),"param_dict":combo}
                row.pop("equity",None)
                fam_results.append(row); results.append(row); done+=1
                if verbose and done%50==0:
                    best_sr=max(r["sharpe"] for r in results) if results else 0
                    print(f"  [{done:>4}/{total}] {done/total*100:.1f}% | best Sharpe: {best_sr:.3f}")

            if fam_results and verbose:
                fam_ranked = sorted(
                    fam_results,
                    key=lambda r: (bool(r.get("bias_robust")), r["score"], r["sharpe"], r["holding_period"]),
                    reverse=True,
                )
                top=fam_ranked[0]
                print(f"  [{fam.name:<14}] top: {top['params'][:30]} "
                      f"Sharpe={top['sharpe']:.3f} MDD={top['mdd']*100:.1f}% "
                      f"Bias={'OK' if top['bias_robust'] else 'FLAG'}")

        df=pd.DataFrame(results)
        if len(df)>0:
            df=self._sort_results(df)
            df.index+=1
        return df

    def print_leaderboard(self, df, n=20):
        if len(df) == 0:
            print("\nNo hay resultados para mostrar.\n")
            return

        ranked = self._sort_results(df.copy())
        top = ranked.head(n).copy()

        robust_count = int(ranked["bias_robust"].sum()) if "bias_robust" in ranked.columns else 0
        print(f"\n{'='*78}")
        print(f"  TOP {n} ESTRATEGIAS | ranking swing-friendly")
        print(f"{'='*78}")
        print(f"  Estrategias robustas: {robust_count}/{len(ranked)}")
        if robust_count == 0:
            print("  Nota: ninguna paso el filtro de robustez. Toma este ranking como exploracion, no validacion final.")

        summary = pd.DataFrame({
            "family": top["family"],
            "params": top["params"].map(lambda x: x if len(str(x)) <= 22 else str(x)[:22] + "..."),
            "sharpe": top["sharpe"].map(lambda x: f"{x:.2f}"),
            "cagr%": top["cagr"].map(lambda x: f"{x*100:.1f}%"),
            "mdd%": top["mdd"].map(lambda x: f"{x*100:.1f}%"),
            "calmar": top["calmar"].map(lambda x: f"{x:.2f}"),
            "score": top["score"].map(lambda x: f"{x:.3f}"),
            "style": top.apply(self._style_tag, axis=1),
            "robust": top["bias_robust"].map(lambda x: "OK" if bool(x) else "FLAG"),
            "bias_cv": top["bias_cv"].map(lambda x: f"{x:.3f}"),
        })
        print(f"\n  RESUMEN")
        print(summary.to_string(index=False))

        profile = pd.DataFrame({
            "family": top["family"],
            "hold_d": top["holding_period"].map(lambda x: f"{x:.1f}"),
            "turn": top["turnover"].map(lambda x: f"{x:.3f}"),
            "trades": top["n_trades"].map(lambda x: f"{int(x)}"),
            "hit%": top["hit_ratio"].map(lambda x: f"{x*100:.1f}%"),
            "tpd": top["tpd"].map(lambda x: f"{x:.3f}"),
            "wr_day%": top["wr"].map(lambda x: f"{x*100:.1f}%"),
        })
        print(f"\n  PERFIL OPERATIVO")
        print(profile.to_string(index=False))

        risk = pd.DataFrame({
            "family": top["family"],
            "sortino": top["sortino"].map(lambda x: f"{x:.2f}"),
            "var95": top["var_95"].map(lambda x: f"{x*100:.2f}%"),
            "cvar95": top["cvar_95"].map(lambda x: f"{x*100:.2f}%"),
            "skew": top["skewness"].map(lambda x: f"{x:.2f}"),
            "kurt": top["kurtosis"].map(lambda x: f"{x:.2f}"),
        })
        print(f"\n  RIESGO Y DISTRIBUCION")
        print(risk.to_string(index=False))

        print(f"\n{'='*78}")
        if "family" in ranked.columns:
            print("\n  MEJOR POR FAMILIA:")
            family_best = self._sort_results(ranked).groupby("family", as_index=False).first()
            for _,row in family_best.iterrows():
                bias_tag = "OK" if row.get("bias_robust") else "FLAG"
                print(
                    f"  {row['family']:<16} Sharpe={row['sharpe']:.2f} "
                    f"CAGR={row['cagr']*100:.1f}% MDD={row['mdd']*100:.1f}% "
                    f"Hold={row['holding_period']:.1f}d Style={self._style_tag(row)} "
                    f"Score={row['score']:.3f} Bias={bias_tag}"
                )
        print()
