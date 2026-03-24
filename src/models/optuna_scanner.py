"""
OPTUNA PARAMETER OPTIMIZATION
Busqueda bayesiana sobre familias de estrategias con validacion multi-regimen.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import SCAN_CFG
from src.models.param_scanner import (
    FAMILIES,
    ParamScanner,
    bias_check,
    composite_score,
    generate_prices,
    run_backtest,
)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger("optuna_scanner")


class OptunaScanner:

    def __init__(self, cfg=None):
        self.cfg = cfg or SCAN_CFG

    def _suggest_param(self, trial, name, values):
        return trial.suggest_categorical(name, values)

    def _sample_params(self, trial, family):
        params = {}
        for spec in family.params:
            params[spec["name"]] = self._suggest_param(trial, spec["name"], spec["values"])
        return params

    def _evaluate_family(self, family, params):
        sharpe_seeds = []
        rows = []
        for seed in range(1, self.cfg.n_seeds + 1):
            for reg in self.cfg.regimes:
                prices = generate_prices(
                    self.cfg.days,
                    self.cfg.drift,
                    self.cfg.volatility,
                    reg,
                    seed * 997 + len(rows) * 13,
                )
                sig = family.signal_fn(prices, **params)
                if sig is None:
                    continue
                bt = run_backtest(prices, sig, cfg=self.cfg)
                rows.append({"regime": reg, "seed": seed, **bt})
                sharpe_seeds.append(bt["sharpe"])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        best_idx = int(df[self.cfg.optuna_metric].idxmax()) if self.cfg.optuna_metric in df.columns else int(df["sharpe"].idxmax())
        best_row = df.loc[best_idx].to_dict()
        robustness = bias_check(sharpe_seeds)
        best_row["score"] = round(float(composite_score(best_row)), 4)
        best_row["bias_robust"] = robustness["robust"]
        best_row["bias_cv"] = round(robustness["cv_sharpe"], 3)
        best_row["bias_pct_pos"] = round(robustness["pct_pos"], 3)
        best_row["params"] = ", ".join(f"{k}={v}" for k, v in params.items())
        best_row["param_dict"] = params
        best_row["n_trials_seen"] = len(df)
        return best_row

    def optimize(self, families=None, n_trials=None, timeout=None, verbose=True):
        if not HAS_OPTUNA:
            raise ImportError("optuna no instalado. Agrega 'optuna' a tu entorno.")
        families = families or FAMILIES
        n_trials = n_trials or self.cfg.optuna_trials
        timeout = timeout or self.cfg.optuna_timeout_sec
        results = []

        for family in families:
            if verbose:
                print(f"\n[OPTUNA] {family.name} | trials={n_trials} timeout={timeout}s")

            best_payload = {"row": None}

            def objective(trial):
                params = self._sample_params(trial, family)
                row = self._evaluate_family(family, params)
                if row is None:
                    return -1e9
                trial.set_user_attr("row", row)
                score = float(row.get(self.cfg.optuna_metric, row.get("score", -1e9)))
                if best_payload["row"] is None or score > float(best_payload["row"].get(self.cfg.optuna_metric, best_payload["row"].get("score", -1e9))):
                    best_payload["row"] = row
                return score

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

            row = best_payload["row"] or study.best_trial.user_attrs.get("row")
            if row is None:
                continue
            row["family"] = family.name
            row["fam_id"] = family.id
            row["color"] = family.color
            row["optuna_best_value"] = round(float(study.best_value), 4)
            row["optuna_trials"] = len(study.trials)
            results.append(row)

            if verbose:
                print(
                    f"  best: {row['params'][:40]} | "
                    f"score={row.get('score', 0):.3f} sharpe={row.get('sharpe', 0):.3f} "
                    f"bias={'OK' if row.get('bias_robust') else 'FLAG'}"
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("optuna_best_value", ascending=False).reset_index(drop=True)
            df.index += 1
        return df

    def print_leaderboard(self, df, n=20):
        ParamScanner(self.cfg).print_leaderboard(df, n=n)
