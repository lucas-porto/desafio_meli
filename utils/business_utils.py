"""
Módulo para métricas de negócio e ROI
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score


def cm_safe(y_true, y_pred):
    """
    Confusion matrix robusta que funciona mesmo com classes ausentes
    """
    present = set(unique_labels(y_true))
    labels = [0, 1]

    # Se faltar alguma classe, force a dimensão 2x2
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Garantir shape 2x2
    if cm.shape != (2, 2):
        # Preenche faltante
        full = np.zeros((2, 2), dtype=int)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if li in present and lj in present:
                    full[i, j] = cm[labels.index(li), labels.index(lj)]
        cm = full

    return cm


def business_roi(cm, c_fn=100_000, c_fp=25_000):
    """
    Calcular ROI em reais (falha custa 4x mais que manutenção preventiva)
    """
    tn, fp, fn, tp = cm.ravel()
    baseline = (tp + fn) * c_fn
    with_model = fn * c_fn + fp * c_fp
    roi_abs = baseline - with_model
    roi_pct = 1 - with_model / baseline if baseline > 0 else 0
    return roi_abs, roi_pct


def subset_events(original_df, dates, devices):
    """
    Filtra original_df para o mesmo recorte de avaliação (datas e dispositivos)
    """
    dmin = pd.to_datetime(dates).min()
    dmax = pd.to_datetime(dates).max()
    devs = set(pd.Series(devices))
    df = original_df.loc[
        (pd.to_datetime(original_df["date"]).between(dmin, dmax))
        & (original_df["device"].isin(devs)),
        ["device", "date", "failure"],
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def pair_alerts_and_failures(original_df, dates, devices, preds, horizon_days=10):
    """
    Faz matching 1:1 entre alertas e falhas por dispositivo.
    Um alerta conta como TP se existir uma falha futura no mesmo device em < horizon_days.
    Cada falha pode ser coberta por no máximo 1 alerta; alertas restantes viram FP.
    Retorna: tp_events, fn_events, fp_alerts
    """
    df_fail = original_df.loc[original_df["failure"] == 1, ["device", "date"]].assign(
        date=lambda d: pd.to_datetime(d["date"])
    )
    df_alert = pd.DataFrame(
        {"device": devices, "date": pd.to_datetime(dates), "alert": preds.astype(int)}
    )
    df_alert = df_alert.loc[df_alert["alert"] == 1, ["device", "date"]]

    tp, fn, fp = 0, 0, 0
    for dev, grp_fail in df_fail.groupby("device"):
        fails = sorted(grp_fail["date"].tolist())
        alerts = sorted(df_alert.loc[df_alert["device"] == dev, "date"].tolist())

        used = set()
        # para cada falha, procurar o último alerta dentro da janela
        for f in fails:
            lo = f - pd.Timedelta(days=horizon_days)
            # encontre o alerta mais recente em [lo, f)
            cand_idx = None
            for i in range(len(alerts) - 1, -1, -1):
                if i in used:
                    continue
                if lo <= alerts[i] < f:
                    cand_idx = i
                    break
            if cand_idx is not None:
                tp += 1
                used.add(cand_idx)
            else:
                fn += 1

        # alertas não usados contam como FP
        fp += len(alerts) - len(used)
    return tp, fn, fp


def business_roi_event_based(
    original_df, dates, devices, preds, c_fn=100_000, c_fp=25_000, horizon_days=10
):
    """
    Calcula ROI baseado em eventos (1:1 alerta-falha)
    """
    tp, fn, fp = pair_alerts_and_failures(
        original_df, dates, devices, preds, horizon_days
    )
    baseline = (tp + fn) * c_fn
    with_model = fn * c_fn + fp * c_fp
    roi_abs = baseline - with_model
    roi_pct = 0.0 if baseline == 0 else 1 - with_model / baseline
    precision_event = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall_event = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    return {
        "tp_events": tp,
        "fn_events": fn,
        "fp_alerts": fp,
        "roi_abs": roi_abs,
        "roi_pct": roi_pct,
        "precision_event": precision_event,
        "recall_event": recall_event,
    }


def calculate_business_metrics(y_true, y_pred, probabilities, model_name):
    """
    Calcula métricas de negócio importantes para early warning.
    """
    # Métricas básicas
    tn, fp, fn, tp = cm_safe(y_true, y_pred).ravel()

    # Métricas de negócio
    cost = 4 * fn + fp
    economy = 4 * (tp + fn) - cost
    alert_rate = y_pred.mean()

    # AUC-PR  - mais estável em dados raros
    auc_pr = average_precision_score(y_true, probabilities)

    # Recall@H (detecção dentro do horizonte)
    recall_h = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Métricas de custo-benefício
    tps_avoided = tp
    fps_generated = fp
    fns_remaining = fn

    return {
        "model": model_name,
        "auc_pr": auc_pr,
        "recall_h": recall_h,
        "alert_rate": alert_rate,
        "cost": cost,
        "economy": economy,
        "tps_avoided": tps_avoided,
        "fps_generated": fps_generated,
        "fns_remaining": fns_remaining,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def break_even_analysis(models_results, c_fn=100_000, c_fp=25_000):
    """
    Análise de break-even: qual precisão mínima para ROI positivo
    """
    print("\n=== ANÁLISE DE BREAK-EVEN (DIÁRIO) ===")
    print(
        "AVISO: Cálculo diário; use break_even_analysis_event_based como métrica oficial."
    )

    # Break-even: 4 * TP - FP >= 0 => FP/TP <= 4 => Precisão >= MIN_PREC_FOR_ROI
    min_precision = c_fp / (c_fp + c_fn)
    print(f"Precisão mínima para break-even: {min_precision:.1%}")
    print(f"Ratio custo FN/FP: {c_fn / c_fp:.1f}")

    for name, results in models_results.items():
        cm = results["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        roi_abs, roi_pct = business_roi(cm, c_fn, c_fp)

        print(f"\n{name}:")
        print(
            f"  Precisão: {precision:.1%} ({'[OK]' if precision >= min_precision else '[FAIL]'})"
        )
        print(f"  ROI: {roi_pct:.1%}")
        print(f"  Economia: R$ {roi_abs:,.0f}")

        if precision < min_precision:
            r = c_fn / c_fp
            needed_tp = math.ceil(fp / r)  # mínimo de TPs p/ break-even
            delta_tp = max(0, needed_tp - tp)
            print(
                f"    Precisa de pelo menos {needed_tp} TPs para break-even (faltam {delta_tp})."
            )

    return min_precision


def break_even_analysis_event_based(models_results, c_fn=100_000, c_fp=25_000):
    """
    Análise de break-even por evento para cada modelo
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE BREAK-EVEN POR EVENTO")
    print("=" * 60)

    min_prec = c_fp / (c_fp + c_fn)
    cost_ratio = c_fn / c_fp

    print(f"Precisão mínima para break-even: {min_prec:.1%}")
    print(f"Ratio custo FN/FP: {cost_ratio:.1f}")
    print()

    for name, results in models_results.items():
        if "business_metrics_event" not in results:
            print(f"{name}: Métricas por evento não disponíveis")
            continue

        ev = results["business_metrics_event"]
        tp_events = ev["tp_events"]
        fn_events = ev["fn_events"]
        fp_alerts = ev["fp_alerts"]
        precision_event = ev["precision_event"]
        roi_pct = ev["roi_pct"]

        print(f"{name}:")
        print(f"  [EVENTO] TP={tp_events}, FN={fn_events}, FP={fp_alerts}")
        print(
            f"  [EVENTO] Precisão: {precision_event:.1%} {'[OK]' if precision_event >= min_prec else '[FAIL]'}"
        )
        print(f"  [EVENTO] ROI: {roi_pct:.1%}")

        if precision_event < min_prec:
            needed_tp = int(np.ceil(fp_alerts * (min_prec / (1 - min_prec))))
            print(
                f"    Precisa de pelo menos {needed_tp} TPs para break-even (faltam {needed_tp - tp_events})."
            )
        print()


def compare_models(models_results, y_test, use_holdout=False):
    """
    Comparar performance dos modelos (versão simplificada)
    """
    # Escolher qual conjunto de predições usar
    pred_key = "predictions_hold" if use_holdout else "predictions"

    # Métricas essenciais apenas
    comparison_df = pd.DataFrame(
        {
            "Modelo": list(models_results.keys()),
            "AUC-ROC": [results["auc"] for results in models_results.values()],
            "Custo": [
                results["business_metrics"]["cost"]
                for results in models_results.values()
            ],
            "Precision": [
                precision_score(y_test, results[pred_key], zero_division=0)
                for results in models_results.values()
            ],
            "Recall": [
                recall_score(y_test, results[pred_key], zero_division=0)
                for results in models_results.values()
            ],
        }
    )

    # Ordenar por métrica apropriada
    if use_holdout:
        # Para holdout, usar custo (menor é melhor) como critério principal
        comparison_df = comparison_df.sort_values("Custo", ascending=True)
    else:
        # Para teste, usar AUC-ROC
        comparison_df = comparison_df.sort_values("AUC-ROC", ascending=False)

    # Mostrar apenas tabela essencial
    print("\nCOMPARAÇÃO DE MODELOS:")
    print(comparison_df.round(4))

    return comparison_df
