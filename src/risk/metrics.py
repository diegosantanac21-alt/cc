"""
METRICAS DE RIESGO, PERFORMANCE Y TRADING
Sharpe, Sortino, CAGR, MDD, VaR, CVaR, turnover, hit ratio, holding period
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from config.settings import RISK_CFG

logger = logging.getLogger("risk")


class RiskMetrics:

    def _as_series(self, arr, name):
        if isinstance(arr, pd.Series):
            return arr.dropna()
        return pd.Series(arr, name=name).dropna()

    def _extract_trades(self, ret, positions):
        ret = self._as_series(ret, "returns")
        positions = self._as_series(positions, "position").reindex(ret.index).fillna(0.0)
        trades = []
        current_pos = 0.0
        entry_idx = None
        trade_returns = []

        for idx in ret.index:
            pos = float(positions.loc[idx])
            day_ret = float(ret.loc[idx])

            if current_pos != 0.0:
                trade_returns.append(day_ret)

            if pos != current_pos:
                if current_pos != 0.0 and trade_returns:
                    compounded = float(np.prod(1 + np.array(trade_returns)) - 1)
                    trades.append({
                        "entry": entry_idx,
                        "exit": idx,
                        "position": current_pos,
                        "holding_period": len(trade_returns),
                        "return": compounded,
                    })
                    trade_returns = []
                if pos != 0.0:
                    entry_idx = idx
                else:
                    entry_idx = None
                current_pos = pos

        if current_pos != 0.0 and trade_returns:
            compounded = float(np.prod(1 + np.array(trade_returns)) - 1)
            trades.append({
                "entry": entry_idx,
                "exit": ret.index[-1],
                "position": current_pos,
                "holding_period": len(trade_returns),
                "return": compounded,
            })
        return trades

    def sharpe(self, ret, rf=0.0, freq=252):
        ex=ret-rf/freq
        return float(np.sqrt(freq)*ex.mean()/ex.std()) if ex.std()>1e-10 else 0.0

    def sortino(self, ret, rf=0.0, freq=252):
        ex=ret-rf/freq; down=ex[ex<0]
        dstd=np.sqrt((down**2).mean()) if len(down)>1 else 1e-10
        return float(np.sqrt(freq)*ex.mean()/dstd)

    def max_drawdown(self, equity):
        return float((equity/equity.cummax()-1).min())

    def drawdown_series(self, equity):
        return (equity/equity.cummax()-1).rename("drawdown")

    def calmar(self, ret, freq=252):
        eq=( 1+ret).cumprod()
        cagr=eq.iloc[-1]**(freq/len(ret))-1
        mdd=abs(self.max_drawdown(eq))
        return float(cagr/mdd) if mdd>1e-6 else 0.0

    def var_hist(self, ret, conf=None):
        c=conf or RISK_CFG.var_confidence
        return float(np.percentile(ret.dropna(),(1-c)*100))

    def cvar(self, ret, conf=None):
        c=conf or RISK_CFG.var_confidence
        var=self.var_hist(ret,c); tail=ret[ret<=var]
        return float(tail.mean()) if len(tail)>0 else float(var)

    def beta(self, ret, bench):
        common=ret.dropna().index.intersection(bench.dropna().index)
        r,b=ret.loc[common],bench.loc[common]
        return float(np.cov(r,b)[0,1]/b.var()) if b.var()>1e-10 else np.nan

    def information_ratio(self, ret, bench, freq=252):
        common=ret.dropna().index.intersection(bench.dropna().index)
        active=ret.loc[common]-bench.loc[common]; te=active.std()
        return float(np.sqrt(freq)*active.mean()/te) if te>1e-10 else 0.0

    def omega(self, ret, threshold=0.0):
        ex=ret-threshold
        g=ex[ex>0].sum(); l=abs(ex[ex<0].sum())
        return float(g/l) if l>1e-10 else np.inf

    def cagr(self, ret, freq=252):
        ret = self._as_series(ret, "returns")
        if len(ret) == 0:
            return 0.0
        eq=(1+ret).cumprod()
        years=len(ret)/freq
        return float(eq.iloc[-1]**(1/max(years,1e-3))-1)

    def turnover(self, positions, freq=252):
        positions = self._as_series(positions, "position")
        if len(positions) < 2:
            return {"turnover": 0.0, "turnover_annual": 0.0}
        daily = positions.diff().abs().fillna(0.0)
        avg_turnover = float(daily.mean())
        return {
            "turnover": avg_turnover,
            "turnover_annual": float(avg_turnover * freq),
        }

    def hit_ratio(self, ret=None, trade_returns=None):
        series = trade_returns if trade_returns is not None else ret
        series = self._as_series(series, "series")
        if len(series) == 0:
            return 0.0
        return float((series > 0).mean())

    def avg_win_loss(self, ret=None, trade_returns=None):
        series = trade_returns if trade_returns is not None else ret
        series = self._as_series(series, "series")
        wins = series[series > 0]
        losses = series[series < 0]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        win_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else np.inf
        return {
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
        }

    def holding_period(self, ret=None, positions=None, freq=252):
        if positions is None:
            ret = self._as_series(ret, "returns")
            return {
                "avg_holding_period_days": 1.0 if len(ret) else 0.0,
                "avg_holding_period_annualized": 1.0 / freq if len(ret) else 0.0,
            }
        trades = self._extract_trades(ret, positions)
        if not trades:
            return {
                "avg_holding_period_days": 0.0,
                "avg_holding_period_annualized": 0.0,
            }
        avg_days = float(np.mean([t["holding_period"] for t in trades]))
        return {
            "avg_holding_period_days": avg_days,
            "avg_holding_period_annualized": float(avg_days / freq),
        }

    def full_report(self, ret, bench=None, rf=0.0, name="Strategy", positions=None):
        ret = self._as_series(ret, "returns")
        if len(ret) == 0:
            rpt = {"strategy": name, "days": 0,
                   "total_return": 0.0, "cagr": 0.0, "vol_annual": 0.0,
                   "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                   "calmar": 0.0, "var_95": 0.0, "cvar_95": 0.0,
                   "omega": 0.0, "skewness": 0.0, "kurtosis": 0.0,
                   "pct_pos_days": 0.0}
            cfg = RISK_CFG
            rpt["checks"] = {"sharpe_ok": False, "sortino_ok": False,
                             "calmar_ok": False, "drawdown_ok": True}
            rpt["all_pass"] = False
            return rpt
        eq=(1+ret.fillna(0)).cumprod(); n=len(ret)
        cagr=self.cagr(ret)
        rpt={"strategy":name,"days":n,
             "total_return":round(float(eq.iloc[-1]-1),4),
             "cagr":round(cagr,4),
             "vol_annual":round(float(ret.std()*np.sqrt(252)),4),
             "sharpe":round(self.sharpe(ret,rf),3),
             "sortino":round(self.sortino(ret,rf),3),
             "max_drawdown":round(self.max_drawdown(eq),4),
             "calmar":round(self.calmar(ret),3),
             "var_95":round(self.var_hist(ret),4),
             "cvar_95":round(self.cvar(ret),4),
             "omega":round(self.omega(ret),3),
             "skewness":round(float(scipy_stats.skew(ret.dropna())),4),
             "kurtosis":round(float(scipy_stats.kurtosis(ret.dropna())),4),
             "pct_pos_days":round(float((ret>0).mean()),3)}
        if positions is not None:
            trades = self._extract_trades(ret, positions)
            trade_returns = pd.Series([t["return"] for t in trades], dtype=float)
            turnover = self.turnover(positions)
            avg_win_loss = self.avg_win_loss(trade_returns=trade_returns)
            holding = self.holding_period(ret=ret, positions=positions)
            rpt.update({
                "turnover": round(turnover["turnover"], 4),
                "turnover_annual": round(turnover["turnover_annual"], 2),
                "hit_ratio": round(self.hit_ratio(trade_returns=trade_returns), 3),
                "avg_win": round(avg_win_loss["avg_win"], 4),
                "avg_loss": round(avg_win_loss["avg_loss"], 4),
                "win_loss_ratio": round(avg_win_loss["win_loss_ratio"], 3),
                "holding_period_days": round(holding["avg_holding_period_days"], 2),
                "n_trades": int(len(trades)),
            })
        if bench is not None:
            rpt["beta"]=round(self.beta(ret,bench),3)
            rpt["ir"]=round(self.information_ratio(ret,bench),3)
        cfg=RISK_CFG
        rpt["checks"]={"sharpe_ok":rpt["sharpe"]>=cfg.min_sharpe,
                        "sortino_ok":rpt["sortino"]>=cfg.min_sortino,
                        "calmar_ok":rpt["calmar"]>=cfg.min_calmar,
                        "drawdown_ok":abs(rpt["max_drawdown"])<=cfg.max_drawdown_limit}
        rpt["all_pass"]=all(rpt["checks"].values())
        return rpt

    def print_report(self, rpt):
        sep="─"*50
        print(f"\n{'='*50}")
        print(f"  PERFORMANCE: {rpt['strategy']}")
        print(f"{'='*50}")
        print(f"  Dias:{rpt['days']}  Retorno:{rpt['total_return']*100:.2f}%  CAGR:{rpt['cagr']*100:.2f}%")
        print(f"\n  {sep}")
        checks=rpt.get("checks",{})
        rows=[("Sharpe",rpt["sharpe"],checks.get("sharpe_ok",True)),
              ("Sortino",rpt["sortino"],checks.get("sortino_ok",True)),
              ("Max Drawdown",rpt["max_drawdown"],checks.get("drawdown_ok",True)),
              ("Calmar",rpt["calmar"],checks.get("calmar_ok",True)),
              ("Vol Anual",rpt["vol_annual"],True),
              ("VaR 95%",rpt["var_95"],True),
              ("CVaR 95%",rpt["cvar_95"],True),
              ("Skewness",rpt["skewness"],True),
              ("Kurtosis",rpt["kurtosis"],True),
              ("Omega",rpt["omega"],True),
              ("Win Rate",rpt["pct_pos_days"],True)]
        for name,val,ok in rows:
            print(f"  {name:<20} {val:>10.4f}  {'OK' if ok else 'FAIL'}")
        if "turnover" in rpt:
            print(f"\n  {'TRADING':<20}")
            print(f"  {'Turnover':<20} {rpt['turnover']:>10.4f}")
            print(f"  {'Turnover Ann.':<20} {rpt['turnover_annual']:>10.2f}")
            print(f"  {'Hit Ratio':<20} {rpt['hit_ratio']:>10.4f}")
            print(f"  {'Avg Win':<20} {rpt['avg_win']:>10.4f}")
            print(f"  {'Avg Loss':<20} {rpt['avg_loss']:>10.4f}")
            print(f"  {'Win/Loss':<20} {rpt['win_loss_ratio']:>10.4f}")
            print(f"  {'Holding Days':<20} {rpt['holding_period_days']:>10.2f}")
            print(f"  {'# Trades':<20} {rpt['n_trades']:>10}")
        print(f"\n  {'ALL PASS' if rpt.get('all_pass') else 'CHECKS FAILED'}")
        print(f"{'='*50}\n")
