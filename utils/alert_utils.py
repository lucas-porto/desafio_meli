"""
Módulo de alerta
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from .business_utils import cm_safe


def apply_simple_threshold(proba_or_pred, thr=None):
    """
    Aplica threshold.
    Se 'thr' for None, assume que 'proba_or_pred' já é 0/1.
    Retorna vetor 0/1.
    """
    if thr is not None:
        return (proba_or_pred >= thr).astype(int)
    else:
        return (proba_or_pred > 0).astype(int)


def find_threshold_for_rate(p, target_rate):
    """Encontra threshold que produz target_rate de alertas"""
    ps = np.sort(np.unique(p))
    lo, hi = 0, len(ps) - 1
    best = ps[hi]
    
    while lo <= hi:
        mid = (lo + hi) // 2
        thr = float(ps[mid])
        preds = (p >= thr).astype(int)
        ar = preds.mean()
        if ar >= target_rate:
            best = thr
            lo = mid + 1
        else:
            hi = mid - 1

    return float(best)


def threshold_for_target_alert_rate(p, target_rate):
    """Encontra threshold que produz target_rate de alertas"""
    return find_threshold_for_rate(p, target_rate)


def cost_simple(y_true, proba, thr):
    """
    Calcula custo 4*FN + 1*FP sem cooldown.
    """
    preds = apply_simple_threshold(proba, thr=thr)
    tn, fp, fn, tp = cm_safe(y_true, preds).ravel()
    cost = 4 * fn + 1 * fp
    return cost, (tn, fp, fn, tp), preds


def tune_alert_budget_simple(
    y_val,
    p_val,
    budgets=(0.01, 0.02, 0.03, 0.05),
):
    """
    Varre budgets (teto de alertas) e escolhe o threshold por quantil que minimiza custo na validação.
    """
    best = {"budget": None, "thr": None, "cost": float("inf"), "cm": None}
    for b in budgets:
        q = 1.0 - b
        thr_b = float(np.quantile(p_val, q))
        cost, cm, _ = cost_simple(y_val, p_val, thr_b)
        if cost < best["cost"]:
            best.update({"budget": b, "thr": thr_b, "cost": float(cost), "cm": cm})
    print(
        f"[budget tuner] best_alert_rate={best['budget'] * 100:.1f}% | thr={best['thr']:.3f} | val_cost={best['cost']:.1f} | cm={best['cm']}"
    )
    return best


def tune_budget_simple(
    y_val,
    p_val,
    budgets=(0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02),
    min_precision=0.20,
):
    """
    Tuner: otimiza budget para maior precisão
    """
    print("Tuner: Budget...")

    best = None
    best_cost = float("inf")

    for budget in budgets:
        # Encontrar threshold para este budget
        q = 1.0 - budget
        thr = float(np.quantile(p_val, q))

        # Aplicar threshold
        cost, cm_tuple, preds = cost_simple(y_val, p_val, thr)
        pr = precision_score(y_val, preds, zero_division=0)
        
        if pr >= min_precision and cost < best_cost:
            best_cost = cost
            tn, fp, fn, tp = cm_tuple
            best = {
                "budget": budget,
                "threshold": thr,
                "cost": cost,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "alerts": (p_val >= thr).sum(),
            }

    if best:
        print(
            f"Melhor: budget={best['budget'] * 100:.2f}%, "
            f"thr={best['threshold']:.4f}, val_cost={best['cost']:.0f}"
        )

    return best


def topk_simple(proba, k=10):
    """
    Estratégia Top-K
    """
    # Encontrar os k maiores scores
    top_k_indices = np.argsort(proba)[-k:]
    preds = np.zeros(len(proba), dtype=np.int8)
    preds[top_k_indices] = 1
    return preds


def event_metrics(original_df, dates, devices, preds, horizon_days=10):
    """
    Métricas por evento: cobertura de falhas reais
    """
    ev = original_df[["date", "device", "failure"]].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["failure"] == 1]

    al = pd.DataFrame(
        {"date": pd.to_datetime(dates), "device": devices, "alert": preds.astype(int)}
    )
    al = al[al["alert"] == 1]

    covered = 0
    for dev, d_fail in ev[["device", "date"]].itertuples(index=False):
        win_start = d_fail - pd.Timedelta(days=horizon_days)
        hit = (
            (al["device"] == dev) & (al["date"] >= win_start) & (al["date"] < d_fail)
        ).any()
        covered += int(hit)

    coverage = covered / max(1, len(ev))
    alerts_per_100dev_week = 100 * (
        al.shape[0]
        / (original_df["device"].nunique() * (original_df["date"].nunique() / 7))
    )
    return {
        "event_coverage": coverage,
        "alerts_per_100dev_week": alerts_per_100dev_week,
    }
