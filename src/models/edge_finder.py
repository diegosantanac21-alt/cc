"""
VALIDACION ESTADISTICA DE VENTAJA
IC, DSR, Bootstrap, Walk-forward, BH/FDR
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import norm
from config.settings import EDGE_CFG
from src.features.indicators import ICAnalyzer

logger = logging.getLogger("edge")


class MultipleHypothesisTester:

    def benjamini_hochberg(self, p_values, q=0.05):
        m=len(p_values)
        indexed=sorted(enumerate(p_values),key=lambda x:x[1])
        thresh=[(i+1)/m*q for i in range(m)]
        last=-1
        for i,((_, p),t) in enumerate(zip(indexed,thresh)):
            if p<=t: last=i
        rejected=[False]*m
        for i,(orig,_) in enumerate(indexed):
            if i<=last: rejected[orig]=True
        return {"rejected":rejected,"n_significant":sum(rejected),"expected_fp":round(sum(rejected)*q,2)}

    def deflated_sharpe(self, sr_obs, T_obs, m_trials, skew=0.0, kurt=0.0):
        sr_d=sr_obs/np.sqrt(252); eg=0.5772156649
        if m_trials>1:
            e_max=((1-eg)*norm.ppf(1-1/m_trials)+eg*norm.ppf(1-1/(m_trials*np.e)))
        else:
            e_max=0.0
        sr_star=e_max/np.sqrt(max(T_obs,1))
        sigma=np.sqrt(max((1-skew*sr_d+(kurt/4)*sr_d**2)/max(T_obs-1,1),1e-10))
        dsr=float(norm.cdf((sr_d-sr_star)/sigma))
        return {"sr_observed":round(sr_obs,4),"sr_star":round(sr_star*np.sqrt(252),4),
                "dsr":round(dsr,4),"significant":dsr>=EDGE_CFG.dsr_threshold}


class BootstrapValidator:

    def __init__(self, n_boot=None, block=None, seed=42):
        self.n_boot=n_boot or EDGE_CFG.n_bootstrap
        self.block=block or EDGE_CFG.bootstrap_block
        self.rng=np.random.default_rng(seed)

    def bootstrap_ic(self, ic_series):
        arr=ic_series.dropna().values; n=len(arr)
        if n == 0:
            return {"ic_observed":0.0,"ci_95_lo":0.0,"ci_95_hi":0.0,
                    "p_value":1.0,"significant":False}
        mu=arr.mean()
        boot=np.array([self.rng.choice(arr,size=n,replace=True).mean()
                        for _ in range(self.n_boot)])
        pct=float(np.mean(boot<mu))
        return {"ic_observed":round(float(mu),4),
                "ci_95_lo":round(float(np.percentile(boot,2.5)),4),
                "ci_95_hi":round(float(np.percentile(boot,97.5)),4),
                "p_value":round(1-pct if mu>0 else pct,4),
                "significant":pct>0.95 if mu>0 else pct<0.05}


class WalkForwardTester:

    def __init__(self, train_mo=None, test_mo=None):
        self.train_mo=train_mo or EDGE_CFG.wf_train_months
        self.test_mo=test_mo or EDGE_CFG.wf_test_months
        self.ic_ana=ICAnalyzer()

    def run(self, signal, prices, horizon=21):
        log_ret=np.log(prices/prices.shift(1))
        fwd_ret=log_ret.rolling(horizon).sum().shift(-horizon)
        dates=signal.index; n_tr=self.train_mo*21; n_ts=self.test_mo*21
        rows=[]; t=n_tr; fold=0
        while t+n_ts<=len(dates):
            sl=dates[t:t+n_ts]
            ic=self.ic_ana.compute(signal.loc[sl],fwd_ret.reindex(sl))
            summ=self.ic_ana.summary(ic)
            rows.append({"fold":fold,"oos_start":sl[0],"oos_end":sl[-1],
                          "ic_mean":summ["ic_mean"],"icir":summ["icir"],
                          "t_stat":summ["t_stat"],"pct_pos":summ.get("pct_positive",0),
                          "confirmed":summ["ic_mean"]>0.02 and summ.get("pct_positive",0)>0.55})
            t+=n_ts; fold+=1
        return pd.DataFrame(rows)

    def summarize(self, df):
        if len(df)==0: return {"error":"Sin folds"}
        n=len(df); nc=df["confirmed"].sum(); pct=nc/n
        return {"n_folds":n,"n_confirmed":int(nc),"confirm_pct":round(float(pct),3),
                "avg_ic_oos":round(float(df["ic_mean"].mean()),4),
                "avg_icir_oos":round(float(df["icir"].mean()),3),
                "verdict":("VENTAJA CONFIRMADA" if pct>=0.60 else
                           "DEBIL — mas validacion" if pct>=0.40 else "SIN VENTAJA")}


class TimeSeriesValidator:

    def __init__(self, train_ratio=None, valid_ratio=None, min_periods=None):
        self.train_ratio=train_ratio or EDGE_CFG.tvt_train_ratio
        self.valid_ratio=valid_ratio or EDGE_CFG.tvt_valid_ratio
        self.min_periods=min_periods or EDGE_CFG.tvt_min_periods
        self.ic_ana=ICAnalyzer()

    def split(self, index):
        n=len(index)
        tr_end=max(int(n*self.train_ratio), self.min_periods)
        va_end=max(int(n*(self.train_ratio+self.valid_ratio)), tr_end+self.min_periods)
        tr_end=min(tr_end, max(n-self.min_periods*2, self.min_periods))
        va_end=min(va_end, n-self.min_periods)
        return {
            "train": index[:tr_end],
            "validation": index[tr_end:va_end],
            "test": index[va_end:],
        }

    def run(self, signal, prices, horizon=21):
        log_ret=np.log(prices/prices.shift(1))
        fwd_ret=log_ret.rolling(horizon).sum().shift(-horizon)
        splits=self.split(signal.index)
        rows=[]
        for name, idx in splits.items():
            if len(idx) < self.min_periods:
                rows.append({"split":name,"n_obs":len(idx),"confirmed":False,"viable":False})
                continue
            ic=self.ic_ana.compute(signal.loc[idx],fwd_ret.reindex(idx))
            summ=self.ic_ana.summary(ic)
            rows.append({
                "split":name,
                "start":idx[0],
                "end":idx[-1],
                "n_obs":len(idx),
                **summ,
                "confirmed":bool(summ.get("viable",False) and summ.get("pct_positive",0)>=0.55),
            })
        return pd.DataFrame(rows)

    def summarize(self, df):
        if len(df)==0 or "split" not in df.columns:
            return {"error":"Sin split temporal"}
        out={}
        for split in ("train","validation","test"):
            row=df[df["split"]==split]
            if len(row)==0:
                continue
            rec=row.iloc[0].to_dict()
            prefix=f"{split}_"
            for key in ["n_obs","ic_mean","icir","t_stat","pct_positive","confirmed"]:
                if key in rec:
                    out[prefix+key]=rec[key]
        out["oos_confirmed"]=bool(out.get("validation_confirmed",False) and out.get("test_confirmed",False))
        out["verdict"]=("OOS OK" if out["oos_confirmed"] else
                        "VALIDACION PARCIAL" if out.get("validation_confirmed",False) else
                        "OOS FALLA")
        return out


class PurgedWalkForwardValidator:

    def __init__(self, n_splits=None, embargo_days=None):
        self.n_splits=n_splits or EDGE_CFG.purged_n_splits
        self.embargo_days=embargo_days or EDGE_CFG.purged_embargo_days
        self.ic_ana=ICAnalyzer()

    def run(self, signal, prices, horizon=21):
        log_ret=np.log(prices/prices.shift(1))
        fwd_ret=log_ret.rolling(horizon).sum().shift(-horizon)
        dates=signal.index; n=len(dates)
        if n < self.n_splits*20:
            return pd.DataFrame()
        min_test_periods=max(63, horizon*3)
        fold_size=n//self.n_splits
        rows=[]
        for fold in range(self.n_splits):
            test_start=fold*fold_size
            test_end=n if fold==self.n_splits-1 else min((fold+1)*fold_size,n)
            test_idx=dates[test_start:test_end]
            if len(test_idx)==0:
                continue
            embargo_start=max(test_start-self.embargo_days,0)
            embargo_end=min(test_end+self.embargo_days,n)
            train_idx=dates[:embargo_start].append(dates[embargo_end:])
            if len(train_idx) < EDGE_CFG.tvt_min_periods or len(test_idx) < min_test_periods:
                continue
            train_ic=self.ic_ana.compute(signal.loc[train_idx],fwd_ret.reindex(train_idx))
            test_ic=self.ic_ana.compute(signal.loc[test_idx],fwd_ret.reindex(test_idx))
            train_sum=self.ic_ana.summary(train_ic)
            test_sum=self.ic_ana.summary(test_ic)
            rows.append({
                "fold":fold,
                "train_start":train_idx[0],
                "train_end":train_idx[-1],
                "test_start":test_idx[0],
                "test_end":test_idx[-1],
                "train_ic_mean":train_sum.get("ic_mean",0.0),
                "test_ic_mean":test_sum.get("ic_mean",0.0),
                "test_icir":test_sum.get("icir",0.0),
                "test_t_stat":test_sum.get("t_stat",0.0),
                "confirmed":bool(test_sum.get("viable",False) and test_sum.get("pct_positive",0)>=0.55),
            })
        return pd.DataFrame(rows)

    def summarize(self, df):
        if len(df)==0:
            return {"error":"Sin folds purgados"}
        pct=float(df["confirmed"].mean())
        return {
            "n_purged_folds":int(len(df)),
            "n_confirmed":int(df["confirmed"].sum()),
            "confirm_pct":round(pct,3),
            "avg_test_ic":round(float(df["test_ic_mean"].mean()),4),
            "avg_test_icir":round(float(df["test_icir"].mean()),3),
            "avg_test_t_stat":round(float(df["test_t_stat"].mean()),3),
            "embargo_days":int(self.embargo_days),
            "verdict":"PURGED OK" if pct>=0.60 else "PURGED DEBIL",
        }


@dataclass
class ResearchReport:
    signal_name: str
    hypothesis:  str
    ic_summary:  dict = field(default_factory=dict)
    split_summary: dict = field(default_factory=dict)
    purged_summary: dict = field(default_factory=dict)
    wf_summary:  dict = field(default_factory=dict)
    dsr:         dict = field(default_factory=dict)
    bootstrap:   dict = field(default_factory=dict)
    decay_df:    Optional[pd.DataFrame] = None
    verdict:     str  = "PENDIENTE"
    notes:       List[str] = field(default_factory=list)

    def _score(self):
        s=0
        if self.ic_summary.get("viable"):                  s+=2
        if self.split_summary.get("oos_confirmed"):        s+=2
        if self.purged_summary.get("confirm_pct",0)>=0.60: s+=2
        if self.wf_summary.get("confirm_pct",0)>=0.60:    s+=2
        if self.dsr.get("significant"):                    s+=2
        if self.bootstrap.get("significant"):              s+=1
        if self.ic_summary.get("pct_positive",0)>=0.55:   s+=1
        return s

    def finalize(self):
        sc=self._score()
        self.verdict=("PRODUCCION ✓ — backtest completo" if sc>=6 else
                      "INVESTIGAR MAS — prometedora" if sc>=4 else
                      "MARGINAL — evidencia debil" if sc>=2 else "DESCARTAR")

    def print(self):
        sep="─"*55
        print(f"\n{'='*55}")
        print(f"  RESEARCH REPORT: {self.signal_name}")
        print(f"  Hipotesis: {self.hypothesis}")
        print(f"{'='*55}")
        for sec,data in [("IC",self.ic_summary),("TRAIN/VAL/TEST",self.split_summary),
                          ("PURGED CV",self.purged_summary),
                          ("WALK-FWD",self.wf_summary),
                          ("DSR",self.dsr),("BOOTSTRAP",self.bootstrap)]:
            if data:
                print(f"\n  {sec}\n  {sep}")
                for k,v in data.items():
                    print(f"  {str(k):<22}: {v}")
        if self.notes:
            print(f"\n  NOTAS")
            for n in self.notes: print(f"  • {n}")
        print(f"\n{'='*55}")
        print(f"  VEREDICTO: {self.verdict}")
        print(f"{'='*55}\n")


class EdgeFinder:

    def __init__(self):
        self.mht=MultipleHypothesisTester(); self.boot=BootstrapValidator()
        self.wf=WalkForwardTester(); self.ic=ICAnalyzer(); self.splitter=TimeSeriesValidator()
        self.purged=PurgedWalkForwardValidator()

    def investigate(self, signal, prices, signal_name, hypothesis,
                    horizon=21, m_trials=1, verbose=True):
        report=ResearchReport(signal_name=signal_name,hypothesis=hypothesis)
        logger.info(f"Investigando: '{signal_name}'")
        log_ret=np.log(prices/prices.shift(1))
        fwd_ret=log_ret.rolling(horizon).sum().shift(-horizon)
        ic_series=self.ic.compute(signal,fwd_ret)
        report.ic_summary=self.ic.summary(ic_series)
        try:
            report.decay_df=self.ic.decay(signal,prices)
        except Exception:
            pass
        T=len(ic_series.dropna())
        sr_ap=abs(report.ic_summary.get("ic_mean",0))*np.sqrt(252/max(horizon,1))
        report.dsr=self.mht.deflated_sharpe(sr_ap,T,max(m_trials,1),
                                              float(ic_series.skew()),float(ic_series.kurtosis()))
        report.bootstrap=self.boot.bootstrap_ic(ic_series)
        split_df=self.splitter.run(signal,prices,horizon)
        report.split_summary=self.splitter.summarize(split_df)
        purged_df=self.purged.run(signal,prices,horizon)
        report.purged_summary=self.purged.summarize(purged_df)
        wf_df=self.wf.run(signal,prices,horizon)
        report.wf_summary=self.wf.summarize(wf_df)
        if not report.ic_summary.get("viable"):
            report.notes.append(f"IC={report.ic_summary.get('ic_mean',0):.4f} no alcanza el umbral.")
        if not report.split_summary.get("oos_confirmed"):
            report.notes.append("Train/validation/test no confirma robustez fuera de muestra.")
        if report.purged_summary.get("confirm_pct",0) < 0.60:
            report.notes.append("Purged CV no confirma estabilidad suficiente.")
        if not report.dsr.get("significant"):
            report.notes.append(f"DSR={report.dsr.get('dsr',0):.3f} — podria ser azar con m_trials={m_trials}.")
        report.finalize()
        if verbose: report.print()
        return report

    def batch_investigate(self, signals, prices, hypotheses, horizon=21):
        m=len(signals); log_ret=np.log(prices/prices.shift(1))
        fwd_ret=log_ret.rolling(horizon).sum().shift(-horizon)
        rows=[]; p_vals=[]
        for name,sig in signals.items():
            ic=self.ic.compute(sig,fwd_ret); summary=self.ic.summary(ic)
            n=max(summary.get("n_periods",2),2); t=abs(summary.get("t_stat",0))
            p=float(2*(1-scipy_stats.t.cdf(t,df=n-1)))
            p_vals.append(p)
            rows.append({"signal":name,"hypothesis":hypotheses.get(name,"—"),
                          **summary,"p_value":round(p,5)})
        if not rows:
            print(f"\n  BATCH ({m} senales, BH q={EDGE_CFG.fdr_q})")
            print("  Sin senales validas para investigar.")
            return pd.DataFrame(columns=[
                "signal", "hypothesis", "n_periods", "ic_mean", "icir",
                "t_stat", "pct_positive", "viable", "p_value", "bh_significant",
            ])
        bh=self.mht.benjamini_hochberg(p_vals,q=EDGE_CFG.fdr_q)
        for i,r in enumerate(rows): r["bh_significant"]=bh["rejected"][i]
        df=pd.DataFrame(rows).sort_values("ic_mean",ascending=False,key=abs)
        print(f"\n  BATCH ({m} senales, BH q={EDGE_CFG.fdr_q})")
        cols=[c for c in ["signal","ic_mean","icir","t_stat","p_value","bh_significant"] if c in df.columns]
        print(df[cols].to_string())
        return df
