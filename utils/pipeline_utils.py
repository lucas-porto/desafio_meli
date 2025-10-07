"""
Módulo para funções do pipeline principal de treinamento e avaliação
"""

import warnings
import pickle
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


from .model_utils import (
    apply_undersampling,
    pick_threshold_with_budget,
    undersample_hard_negatives,
    as_df,
    pick_threshold_on_val,
    timeaware_calibration,
)
from .alert_utils import (
    apply_simple_threshold,
    tune_budget_simple,
    tune_alert_budget_simple,
)
from .business_utils import (
    business_roi_event_based,
    cm_safe,
    compare_models,
    calculate_business_metrics,
)

warnings.filterwarnings("ignore")


def pick_threshold_min_precision(
    y_val,
    p_val,
    min_precision=0.20,
):
    """
    Selecionar threshold com restrição de precisão mínima
    """
    # varrer thresholds (concentrando no topo)
    thrs = np.unique(
        np.r_[np.quantile(p_val, np.linspace(0.90, 1.0, 200)), p_val.max()]
    )
    best_t, best_pr = None, -1.0

    for t in thrs[::-1]:
        preds = apply_simple_threshold(p_val, thr=float(t))
        pr = precision_score(y_val, preds, zero_division=0)
        if pr >= min_precision:
            return float(t), float(preds.mean())  # achou um que atende ROI
        if pr > best_pr:
            best_pr = pr
            best_t = float(t)

    # fallback: o de maior precisão alcançada (ainda que < min_precision)
    preds = apply_simple_threshold(p_val, thr=best_t)
    return best_t, float(preds.mean())


def tune_neg_per_pos(
    X_train,
    y_train,
    X_val,
    y_val,
    candidates=(5, 8, 10, 12, 15, 20, 50, 100),
    random_state=42,
    max_alert_rate=0.05,
    c_fn=100_000,
    c_fp=25_000,
):
    """Escolhe neg_per_pos que minimiza custo na validação com orçamento de alertas."""
    print("\n=== TUNING NEG_PER_POS ===")
    best = {"neg_per_pos": None, "cost": float("inf"), "thr": None, "alert_rate": None}

    min_pr = c_fp / (c_fp + c_fn)

    for r in candidates:
        print(f"Testando neg_per_pos={r}...")
        X_tr_us, y_tr_us = apply_undersampling(
            X_train, y_train, random_state=random_state, neg_per_pos=r
        )

        # modelo rápido p/ tuner
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            scale_pos_weight=1.0,
            verbosity=-1,
        )
        model.fit(
            X_tr_us,
            y_tr_us,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )

        proba_val = model.predict_proba(X_val)[:, 1]
        thr, alert_rate, method = pick_threshold_with_budget(
            y_val, proba_val, fn_cost=4, fp_cost=1, max_alert_rate=max_alert_rate
        )
        pred_val = (proba_val >= thr).astype(int)

        pr = precision_score(y_val, pred_val, zero_division=0)
        tn, fp, fn, tp = cm_safe(y_val, pred_val).ravel()
        cost = 4 * fn + fp

        # penaliza soluções que não atingem a precisão mínima
        if pr < min_pr:
            cost = np.inf

        print(
            f"  neg_per_pos={r}: custo={cost if np.isfinite(cost) else 'INF'}, "
            f"thr={thr:.3f}, alertas={alert_rate:.1%}, precisão={pr:.1%}, método={method}"
        )

        if cost < best["cost"]:
            best.update(
                {
                    "neg_per_pos": int(r),
                    "cost": float(cost),
                    "thr": float(thr),
                    "alert_rate": float(alert_rate),
                }
            )

    print(
        f"[tuner] melhor neg_per_pos={best['neg_per_pos']} | custo_val={best['cost']:.1f} | thr_val={best['thr']:.3f} | alertas={best['alert_rate']:.1%}"
    )
    return best


def train_evaluate_with_presplit(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    min_pr=0.2,
    X_val_hold=None,
    y_val_hold=None,
    dates_train=None,
    devices_train=None,
    dates_val=None,
    devices_val=None,
    dates_test=None,
    devices_test=None,
    random_state=42,
    apply_balance=True,
    neg_per_pos=50,
    max_neg_cap=None,
    calibrate=True,
    use_alert_budget=True,
    alert_budgets=(0.01, 0.015, 0.02, 0.03),
    feature_names=None,
    selected_features=None,
):
    """
    Treina XGB e LGBM com (opcional) undersampling, calibração isotônica,
    seleção de threshold por orçamento de alertas + custo na validação.
    """
    print("\n=== TREINANDO MODELOS COM CONJUNTOS PR-SPLIT (BUDGET+CALIBRAÇÃO) ===")

    # Garantir alinhamento de colunas - fonte única da verdade
    feature_names = (
        selected_features if selected_features is not None else feature_names
    )
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Converter para DataFrames consistentes
    X_train_df = as_df(X_train, feature_names)

    # Undersampling no treino
    if apply_balance:
        # Tentar usar undersampling de negativos duros se metadados disponíveis
        if dates_train is not None and devices_train is not None:
            try:
                X_train_use, y_train_use = undersample_hard_negatives(
                    X_train_df,
                    y_train,
                    dates_train,
                    devices_train,
                    neg_per_pos=neg_per_pos,
                    random_state=random_state,
                )

                print(
                    f"Undersampling hard negatives aplicado: {X_train_use.shape[0]} amostras"
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}..."
                print(
                    f"Erro no undersampling hard negatives; usando padrão. Detalhe: {msg}"
                )
                X_train_use, y_train_use = apply_undersampling(
                    X_train_df,
                    y_train,
                    random_state=random_state,
                    neg_per_pos=neg_per_pos,
                    max_neg_cap=max_neg_cap,
                )
        else:
            X_train_use, y_train_use = apply_undersampling(
                X_train_df,
                y_train,
                random_state=random_state,
                neg_per_pos=neg_per_pos,
                max_neg_cap=max_neg_cap,
            )
        spw = 1.0  # Não reponderar após undersampling
    else:
        X_train_use, y_train_use = X_train_df, y_train
        pos = y_train_use.sum()
        neg = len(y_train_use) - pos
        spw = (neg / pos) if pos > 0 else 1.0

    print(f"Scale pos weight: {spw:.2f}")
    print(
        f"Train: {X_train_use.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}"
    )

    # === XGBoost ==============================================================
    print("\n=== TREINANDO XGBOOST ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        early_stopping_rounds=10,
        scale_pos_weight=spw,
        verbosity=0,
        n_jobs=-1,
    )
    # Garantir alinhamento consistente usando feature_names
    X_val_aligned = as_df(X_val, feature_names)
    X_test_aligned = as_df(X_test, feature_names)

    # Se undersampling retornou array, reembrulhe com colunas corretas
    if not hasattr(X_train_use, "columns"):
        X_train_use = pd.DataFrame(X_train_use, columns=feature_names)

    xgb_model.fit(
        X_train_use, y_train_use, eval_set=[(X_val_aligned, y_val)], verbose=False
    )

    # probabilidades
    if calibrate:
        xgb_p_val, xgb_p_test, xgb_cals = timeaware_calibration(
            xgb_model,
            X_val_aligned,
            y_val,
            X_test_aligned,
            n_splits=5,
            method="isotonic",
        )
        print("Calibrao time-aware aplicada (XGB).")
    else:
        xgb_p_val = xgb_model.predict_proba(X_val_aligned)[:, 1]
        xgb_p_test = xgb_model.predict_proba(X_test)[:, 1]

    # threshold (budget na validação)
    if use_alert_budget:
        min_pr = min_pr
        best_xgb = tune_budget_simple(y_val, xgb_p_val, min_precision=min_pr)
        if best_xgb is not None:
            xgb_thr = best_xgb["threshold"]
            xgb_best_budget = best_xgb["budget"]
            print(f"[XGB] threshold com precisão mínima: {best_xgb['precision']:.1%}")
        else:
            # Fallback: usar threshold com maior precisão possível
            xgb_thr, xgb_alert_rate = pick_threshold_min_precision(
                y_val, xgb_p_val, min_precision=0.0
            )
            xgb_best_budget = xgb_alert_rate
            print(
                f"[XGB] fallback - melhor precisão disponível (thr={xgb_thr:.3f}, alertas={xgb_alert_rate:.1%})"
            )
    else:
        xgb_thr, _ = pick_threshold_on_val(y_val, xgb_p_val, 4, 1)
        xgb_best_budget = 0.01
        print(f"[XGB] thr val (custo)={xgb_thr:.3f}")

    # preds no teste
    xgb_pred_test = apply_simple_threshold(xgb_p_test, thr=xgb_thr)

    xgb_auc = roc_auc_score(y_test, xgb_p_test)
    xgb_ap = average_precision_score(y_test, xgb_p_test)
    xgb_cm = cm_safe(y_test, xgb_pred_test)
    tn, fp, fn, tp = xgb_cm.ravel()
    xgb_cost = 4 * fn + fp
    print(
        f"XGB | AUC={xgb_auc:.4f} AP={xgb_ap:.4f} thr={xgb_thr:.3f} cost={xgb_cost:.1f} | CM={xgb_cm.tolist()}"
    )

    # === LightGBM =============================================================
    print("\n=== TREINANDO LIGHTGBM ===")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=random_state,
        scale_pos_weight=spw,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_train_use,
        y_train_use,
        eval_set=[(X_val_aligned, y_val)],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )

    if calibrate:
        lgb_p_val, lgb_p_test, lgb_cals = timeaware_calibration(
            lgb_model,
            X_val_aligned,
            y_val,
            X_test_aligned,
            n_splits=5,
            method="isotonic",
        )
        print("Calibrao time-aware aplicada (LGB).")
    else:
        lgb_p_val = lgb_model.predict_proba(X_val_aligned)[:, 1]
        lgb_p_test = lgb_model.predict_proba(X_test)[:, 1]

    if use_alert_budget:
        min_pr = min_pr
        best_lgb = tune_budget_simple(y_val, lgb_p_val, min_precision=min_pr)
        if best_lgb is not None:
            lgb_thr = best_lgb["threshold"]
            lgb_best_budget = best_lgb["budget"]
            print(f"[LGB] threshold com precisão mínima: {best_lgb['precision']:.1%}")
        else:
            # Fallback: usar threshold com maior precisão possível
            lgb_thr, lgb_alert_rate = pick_threshold_min_precision(
                y_val, lgb_p_val, min_precision=0.0
            )
            lgb_best_budget = lgb_alert_rate
            print(
                f"[LGB] fallback - melhor precisão disponível (thr={lgb_thr:.3f}, alertas={lgb_alert_rate:.1%})"
            )
    else:
        lgb_thr, _ = pick_threshold_on_val(y_val, lgb_p_val, 4, 1)
        lgb_best_budget = 0.01
        print(f"[LGB] thr val (custo)={lgb_thr:.3f}")

    # preds no teste
    lgb_pred_test = apply_simple_threshold(lgb_p_test, thr=lgb_thr)

    lgb_auc = roc_auc_score(y_test, lgb_p_test)
    lgb_ap = average_precision_score(y_test, lgb_p_test)
    lgb_cm = cm_safe(y_test, lgb_pred_test)
    tn, fp, fn, tp = lgb_cm.ravel()
    lgb_cost = 4 * fn + fp
    print(
        f"LGB | AUC={lgb_auc:.4f} AP={lgb_ap:.4f} thr={lgb_thr:.3f} cost={lgb_cost:.1f} | CM={lgb_cm.tolist()}"
    )

    # Ensemble ponderado por performance
    w_xgb = average_precision_score(y_val, xgb_p_val) + 1e-9
    w_lgb = average_precision_score(y_val, lgb_p_val) + 1e-9
    w_sum = w_xgb + w_lgb

    ensemble_p_val = (w_xgb / w_sum) * xgb_p_val + (w_lgb / w_sum) * lgb_p_val
    ensemble_p_test = (w_xgb / w_sum) * xgb_p_test + (w_lgb / w_sum) * lgb_p_test

    print(f"Ensemble ponderado: XGB={w_xgb / w_sum:.3f}, LGB={w_lgb / w_sum:.3f}")

    # threshold para ensemble (budget na validação)
    if use_alert_budget:
        best_ensemble = tune_alert_budget_simple(
            y_val,
            ensemble_p_val,
            budgets=alert_budgets,
        )
        ensemble_thr = best_ensemble["thr"]
        ensemble_budget = best_ensemble["budget"]
    else:
        ensemble_thr, _ = pick_threshold_on_val(y_val, ensemble_p_val, 4, 1)
        ensemble_budget = 0.01
        print(f"[Ensemble] thr val (custo)={ensemble_thr:.3f}")

    # preds no teste
    ensemble_pred_test = apply_simple_threshold(ensemble_p_test, thr=ensemble_thr)

    ensemble_auc = roc_auc_score(y_test, ensemble_p_test)
    ensemble_ap = average_precision_score(y_test, ensemble_p_test)
    ensemble_cm = cm_safe(y_test, ensemble_pred_test)
    tn, fp, fn, tp = ensemble_cm.ravel()
    ensemble_cost = 4 * fn + fp
    print(
        f"Ensemble | AUC={ensemble_auc:.4f} AP={ensemble_ap:.4f} thr={ensemble_thr:.3f} cost={ensemble_cost:.1f} | CM={ensemble_cm.tolist()}"
    )

    # Calcular predições de holdout se disponível
    holdout_predictions = {}
    if X_val_hold is not None:
        print("Calculando predições de holdout durante o treinamento...")

        # XGBoost holdout
        xgb_proba_hold = xgb_model.predict_proba(X_val_hold)[:, 1]
        if calibrate and xgb_cals is not None:
            cal_preds = [cm.transform(xgb_proba_hold) for cm in xgb_cals]
            xgb_proba_hold = np.mean(cal_preds, axis=0)
        xgb_pred_hold = apply_simple_threshold(xgb_proba_hold, xgb_thr)
        holdout_predictions["XGBoost"] = xgb_pred_hold

        # LightGBM holdout
        lgb_proba_hold = lgb_model.predict_proba(X_val_hold)[:, 1]
        if calibrate and lgb_cals is not None:
            cal_preds = [cm.transform(lgb_proba_hold) for cm in lgb_cals]
            lgb_proba_hold = np.mean(cal_preds, axis=0)
        lgb_pred_hold = apply_simple_threshold(lgb_proba_hold, lgb_thr)
        holdout_predictions["LightGBM"] = lgb_pred_hold

        # Ensemble holdout
        w_xgb = xgb_ap + 1e-9
        w_lgb = lgb_ap + 1e-9
        w_sum = w_xgb + w_lgb
        ensemble_proba_hold = (w_xgb / w_sum) * xgb_proba_hold + (
            w_lgb / w_sum
        ) * lgb_proba_hold
        ensemble_pred_hold = apply_simple_threshold(ensemble_proba_hold, ensemble_thr)
        holdout_predictions["EnsembleAvg"] = ensemble_pred_hold

    # guardar
    models_results = {
        "XGBoost": {
            "model": xgb_model,
            "probabilities": xgb_p_test,
            "predictions": xgb_pred_test,
            "auc": xgb_auc,
            "ap": xgb_ap,
            "confusion_matrix": xgb_cm,
            "threshold": float(xgb_thr),
            "cost": float(xgb_cost),
            "proba_val": xgb_p_val,
            "tuned_budget": float(xgb_best_budget),
            "calibrator": xgb_cals if calibrate else None,
            "predictions_hold": holdout_predictions.get("XGBoost"),
        },
        "LightGBM": {
            "model": lgb_model,
            "probabilities": lgb_p_test,
            "predictions": lgb_pred_test,
            "auc": lgb_auc,
            "ap": lgb_ap,
            "confusion_matrix": lgb_cm,
            "threshold": float(lgb_thr),
            "cost": float(lgb_cost),
            "proba_val": lgb_p_val,
            "tuned_budget": float(lgb_best_budget),
            "calibrator": lgb_cals if calibrate else None,
            "predictions_hold": holdout_predictions.get("LightGBM"),
        },
        "EnsembleAvg": {
            "model": None,
            "xgb_model": xgb_model,  # Modelo XGBoost
            "lgb_model": lgb_model,  # Modelo LightGBM
            "xgb_calibrator": xgb_cals if calibrate else None,  # Calibrador XGBoost
            "lgb_calibrator": lgb_cals if calibrate else None,  # Calibrador LightGBM
            "probabilities": ensemble_p_test,
            "predictions": ensemble_pred_test,
            "auc": ensemble_auc,
            "ap": ensemble_ap,
            "confusion_matrix": ensemble_cm,
            "threshold": float(ensemble_thr),
            "cost": float(ensemble_cost),
            "proba_val": ensemble_p_val,
            "tuned_budget": float(ensemble_budget),
            "predictions_hold": holdout_predictions.get("EnsembleAvg"),
        },
    }
    return models_results


def apply_optimal_thresholds(
    models_results,
    y_val,
    y_test,
    *,
    dates_val,
    devices_val,
    dates_test=None,
    devices_test=None,
    df_features=None,  # DataFrame original para avaliação por evento
    min_precision=0.2,
    c_fn=100_000,
    c_fp=25_000,
    budgets=(0.01, 0.015, 0.02, 0.03),
    horizon_days=10,
):
    """
    Aplicar thresholds ótimos com restrição de precisão mínima (versão simplificada)

    Args:
        models_results: Dicionário com resultados dos modelos
        y_val: Labels de validação
        y_test: Labels de teste
        dates_val: Datas de validação
        devices_val: Dispositivos de validação
        dates_test: Datas de teste
        devices_test: Dispositivos de teste
        df_features: DataFrame original com colunas 'device', 'date', 'failure' para avaliação por evento
        min_precision: Precisão mínima exigida (0.2 por padrão)
        budgets: Lista de budgets para testar
        horizon_days: Horizonte de previsão

    Returns:
        models_results: Dicionário atualizado com thresholds ótimos aplicados
    """
    print("\n" + "=" * 60)
    print(f"APLICAÇÃO DOS THRESHOLDS ÓTIMOS (min_precision={min_precision:.0%})")
    print("=" * 60)

    for name in ["XGBoost", "LightGBM", "EnsembleAvg"]:
        r = models_results[name]
        p_val, p_test = r["proba_val"], r["probabilities"]

        # Usar tuner simples
        best = tune_budget_simple(
            y_val, p_val, budgets=budgets, min_precision=min_precision
        )

        if best is not None:
            thr = best["threshold"]
            budget = best["budget"]
        else:
            # Fallback: usar threshold por custo
            thr, _ = pick_threshold_on_val(y_val, p_val, 4, 1)
            budget = 0.01

        # Aplicar threshold simples
        preds_test = apply_simple_threshold(p_test, thr=thr)
        bm = calculate_business_metrics(y_test, preds_test, p_test, name)

        # Calcular métricas por evento para o teste
        if (
            df_features is not None
            and dates_test is not None
            and devices_test is not None
        ):
            df_events_test = _subset_events(df_features, dates_test, devices_test)
            ev_test = business_roi_event_based(
                original_df=df_events_test,
                dates=dates_test,
                devices=devices_test,
                preds=preds_test,
                c_fn=c_fn,
                c_fp=c_fp,
                horizon_days=horizon_days,
            )
            models_results[name]["business_metrics_event"] = ev_test

        models_results[name].update(
            {
                "threshold": thr,
                "predictions": preds_test,
                "confusion_matrix": np.array(bm["confusion_matrix"]),
                "cost": float(bm["cost"]),
                "alert_rate": float(bm["alert_rate"]),
                "business_metrics": bm,
                "applied_budget": float(budget),
            }
        )

        # Mostrar métricas por evento se disponíveis
        event_info = ""
        if "business_metrics_event" in models_results[name]:
            ev = models_results[name]["business_metrics_event"]
            event_info = f" | prec_event={ev['precision_event']:.1%} | recall_event={ev['recall_event']:.1%} | ROI={ev['roi_pct']:.1%}"

        print(
            f"\n{name}: thr={thr:.4f} | budget~{budget:.3%} | "
            f"alertas={bm['alert_rate']:.2%} | precisão_test={precision_score(y_test, preds_test, zero_division=0):.1%} | custo={bm['cost']:.0f}{event_info}"
        )

    return models_results


def analyze_feature_importance(models_results, feature_cols, top_n=20):
    """
    Analisar importncia das features para ambos os modelos
    """
    print("\n=== ANÁLISE DE FEATURE IMPORTANCE ===")

    for model_name, results in models_results.items():
        print(f"\n{model_name} - Top {top_n} Features:")
        print("=" * 50)

        model = results["model"]

        # Pular EnsembleRank (no tem modelo nico)
        if model is None:
            print("EnsembleRank não tem feature importance individual")
            continue

        # Obter feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "booster_"):
            importance = model.booster_.get_score(importance_type="weight")
            # Converter para array se necessrio
            if isinstance(importance, dict):
                importance = [
                    importance.get(f"f{i}", 0) for i in range(len(feature_cols))
                ]

        # Criar DataFrame com importance
        feature_importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Mostrar top features
        top_features = feature_importance_df.head(top_n)
        for idx, row in top_features.iterrows():
            print(f"{row['importance']:.4f} - {row['feature']}")

        # Salvar para anlise posterior
        results["feature_importance"] = feature_importance_df

        # Análise por tipo de feature

        feature_types_importance = {}
        for _, row in feature_importance_df.iterrows():
            feature = row["feature"]
            importance = row["importance"]

            # Classificar tipo de feature
            if "_lag_" in feature:
                feature_type = "Temporal (Lag)"
            elif "_ma_" in feature:
                feature_type = "Temporal (Média Móvel)"
            elif "_std_" in feature:
                feature_type = "Temporal (Desvio Padrão)"
            elif "_diff" in feature:
                feature_type = "Temporal (Diferença)"
            elif "_slope_" in feature:
                feature_type = "Temporal (Tendência)"
            elif "device" in feature:
                feature_type = "Dispositivo"
            elif feature in [
                "year",
                "month",
                "day",
                "dayofweek",
                "dayofyear",
                "weekofyear",
                "quarter",
            ]:
                feature_type = "Temporal (Data)"
            elif feature.startswith("is_"):
                feature_type = "Sazonalidade"
            elif "_sum" in feature or "_ratio" in feature:
                feature_type = "Interação"
            else:
                feature_type = "Original"

            if feature_type not in feature_types_importance:
                feature_types_importance[feature_type] = []
            feature_types_importance[feature_type].append(importance)

        # Mostrar média de importance por tipo
        for feature_type, importances in feature_types_importance.items():
            avg_importance = np.mean(importances)
            count = len(importances)

    return models_results


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


def calculate_event_metrics(
    models_results, original_df, dates_test, devices_test, horizon_days=10
):
    """
    Calcula métricas por evento para todos os modelos
    """
    print("\n=== MÉTRICAS POR EVENTO ===")

    # Filtrar subset de eventos para teste
    df_events_test = _subset_events(original_df, dates_test, devices_test)

    for k, v in models_results.items():
        evm = event_metrics(
            original_df=df_events_test,
            dates=dates_test,
            devices=devices_test,
            preds=v["predictions"],
            horizon_days=horizon_days,
        )
        models_results[k]["event_metrics"] = evm
        print(
            f"{k} | event_coverage={evm['event_coverage']:.1%} | alerts_per_100dev_week={evm['alerts_per_100dev_week']:.2f}"
        )

    return models_results


def select_best_model(models_results, y_val_hold):
    """
    Seleciona o melhor modelo baseado na performance no conjunto de holdout.

    Args:
        models_results: Dicionário com resultados dos modelos
        y_val_hold: Labels do conjunto de holdout

    Returns:
        tuple: (best_model_name, best_model_results, comparison_df)
    """
    print(f"\n{'=' * 60}")
    print("SELEÇÃO DO MELHOR MODELO (VALIDAÇÃO HOLDOUT)")
    print(f"{'=' * 60}")

    # Comparar modelos usando dados de holdout
    comparison_df = compare_models(models_results, y_val_hold, use_holdout=True)

    # Selecionar o melhor modelo (primeiro da lista ordenada)
    best_model_name = comparison_df.iloc[0]["Modelo"]
    best_model_results = models_results[best_model_name]

    if "predictions_hold" in best_model_results:
        preds_hold = best_model_results["predictions_hold"]
        if preds_hold is not None and len(preds_hold) > 0:
            # Calcular métricas técnicas no holdout (métricas de validação)
            precision_hold = precision_score(y_val_hold, preds_hold, zero_division=0)
            recall_hold = recall_score(y_val_hold, preds_hold, zero_division=0)
            f1_hold = f1_score(y_val_hold, preds_hold, zero_division=0)

            # Adicionar métricas ao best_model_results
            best_model_results["precision"] = precision_hold
            best_model_results["recall"] = recall_hold
            best_model_results["f1"] = f1_hold
            best_model_results["metrics_source"] = "holdout"
        else:
            # Fallback: usar métricas do teste se holdout estiver vazio
            preds_test = best_model_results["predictions"]
            precision_test = precision_score(y_val_hold, preds_test, zero_division=0)
            recall_test = recall_score(y_val_hold, preds_test, zero_division=0)
            f1_test = f1_score(y_val_hold, preds_test, zero_division=0)

            best_model_results["precision"] = precision_test
            best_model_results["recall"] = recall_test
            best_model_results["f1"] = f1_test
            best_model_results["metrics_source"] = "test_fallback"
    else:
        # Fallback: usar métricas do teste
        preds_test = best_model_results["predictions"]
        precision_test = precision_score(y_val_hold, preds_test, zero_division=0)
        recall_test = recall_score(y_val_hold, preds_test, zero_division=0)
        f1_test = f1_score(y_val_hold, preds_test, zero_division=0)

        best_model_results["precision"] = precision_test
        best_model_results["recall"] = recall_test
        best_model_results["f1"] = f1_test
        best_model_results["metrics_source"] = "test_fallback"

    print(f"\nMELHOR MODELO SELECIONADO: {best_model_name}")
    print(f"   Custo: {best_model_results['cost']:.0f}")

    return best_model_name, best_model_results, comparison_df


def _subset_events(original_df, dates, devices):
    """
    Filtra original_df para o mesmo recorte de avaliação
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


def execute_final_model_evaluation(
    best_model_name,
    best_model_results,
    X_test,
    y_test,
    dates_test,
    devices_test,
    df,
    c_fn,
    c_fp,
    horizon_days=10,
):
    """
    Executa a avaliação final do melhor modelo nos dados de teste.

    Args:
        best_model_name: Nome do melhor modelo selecionado
        best_model_results: Resultados do melhor modelo
        X_test: Dados de teste
        y_test: Labels de teste
        dates_test: Datas de teste
        devices_test: Dispositivos de teste
        df: DataFrame original
        horizon_days: Horizonte de predição

    Returns:
        dict: Resultados finais da avaliação
    """
    print("\n" + "=" * 60)
    print(f"EXECUÇÃO FINAL DO MELHOR MODELO: {best_model_name}")
    print("=" * 60)

    # Obter modelo e threshold
    model = best_model_results.get("model")
    threshold = best_model_results.get("threshold", 0.5)
    calibrator = best_model_results.get("calibrator")

    # Fazer predições no teste
    if best_model_name == "EnsembleAvg":
        print("Executando Ensemble (XGBoost + LightGBM)...")
        proba_test = best_model_results.get("probabilities", np.zeros(len(y_test)))
    else:
        print(f"Executando {best_model_name}...")
        proba_test = model.predict_proba(X_test)[:, 1]

        # Aplicar calibração se disponível
        if calibrator is not None:
            print("Aplicando calibração...")
            # Calibrador é uma lista de modelos, usar média
            cal_preds = [cm.transform(proba_test) for cm in calibrator]
            proba_test = np.mean(cal_preds, axis=0)

    # Aplicar threshold
    preds_test = apply_simple_threshold(proba_test, threshold)

    # Calcular métricas de negócio
    business_metrics = calculate_business_metrics(
        y_test, preds_test, proba_test, best_model_name
    )

    # Calcular métricas por evento
    df_events_test = _subset_events(df, dates_test, devices_test)
    ev_metrics = business_roi_event_based(
        original_df=df_events_test,
        dates=dates_test,
        devices=devices_test,
        preds=preds_test,
        c_fn=c_fn,
        c_fp=c_fp,
        horizon_days=horizon_days,
    )

    # Compilar resultados finais
    final_test_results = {
        "model_name": best_model_name,
        "threshold": float(threshold),
        "business_metrics": business_metrics,
        "business_metrics_event": ev_metrics,
        "test_data_info": {
            "total_samples": len(y_test),
            "positive_samples": int(y_test.sum()),
            "negative_samples": int((y_test == 0).sum()),
            "positive_rate": float(y_test.mean()),
        },
    }

    # Imprimir resumo
    print("\nRESULTADOS DO TESTE FINAL:")
    print(f"   Modelo: {best_model_name}")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Alert Rate: {business_metrics['alert_rate']:.3f}")
    print(f"   Recall: {business_metrics['recall_h']:.3f}")
    print(f"   AUC-PR: {business_metrics['auc_pr']:.3f}")
    print("\nMÉTRICAS DE NEGÓCIO:")
    print(f"   ROI por Evento: {ev_metrics['roi_pct']:.1%}")
    print(f"   Precisão por Evento: {ev_metrics['precision_event']:.1%}")
    print(f"   Recall por Evento: {ev_metrics['recall_event']:.1%}")
    print("\nANÁLISE DE ALERTAS:")
    print(f"   Total de Alertas: {preds_test.sum()}")
    print(f"   Taxa de Alertas: {preds_test.mean():.2%}")
    print(f"   Falsos Negativos: {((y_test == 1) & (preds_test == 0)).sum()}")
    print(f"   Falsos Positivos: {((y_test == 0) & (preds_test == 1)).sum()}")

    return final_test_results


def create_final_summary(
    models_results,
    dates_train,
    dates_val_model,
    dates_val_hold,
    dates_test,
    selected_features,
    cv_results,
    cv_summary,
    suspicious_features,
    c_fn=100_000,
    c_fp=25_000,
    min_roi=0.02,
):
    """
    Cria o resumo final com métricas essenciais e ROI por evento
    """
    print("\n" + "=" * 60)
    print("RESUMO FINAL - MÉTRICAS ESSENCIAIS")
    print("=" * 60)

    # Calcular períodos dos conjuntos
    train_period = (
        pd.to_datetime(dates_train).min(),
        pd.to_datetime(dates_train).max(),
    )
    val_all = np.r_[pd.to_datetime(dates_val_model), pd.to_datetime(dates_val_hold)]
    val_period = (val_all.min(), val_all.max())
    test_period = (pd.to_datetime(dates_test).min(), pd.to_datetime(dates_test).max())

    split_info = {
        "train_period": f"{train_period[0].strftime('%Y-%m-%d')} to {train_period[1].strftime('%Y-%m-%d')}",
        "val_period": f"{pd.to_datetime(val_period[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(val_period[1]).strftime('%Y-%m-%d')}",
        "test_period": f"{test_period[0].strftime('%Y-%m-%d')} to {test_period[1].strftime('%Y-%m-%d')}",
        "purge_days": 7,
    }

    # Resumo dos modelos (métricas essenciais)
    print("\nCOMPARAÇÃO DE MODELOS:")
    print("-" * 50)
    for k, v in models_results.items():
        # ROI por evento (métrica principal)
        roi_event = v.get("business_metrics_event", {}).get("roi_pct", 0)
        precision_event = v.get("business_metrics_event", {}).get("precision_event", 0)
        recall_event = v.get("business_metrics_event", {}).get("recall_event", 0)

        print(f"{k}:")
        print(f"  ROI por Evento: {roi_event:.1%}")
        print(f"  Precisão por Evento: {precision_event:.1%}")
        print(f"  Recall por Evento: {recall_event:.1%}")
        print(f"  Custo: {v['cost']:.0f}")
        print()

    # Análise de break-even (simplificada)
    print("ANÁLISE DE BREAK-EVEN:")
    print("-" * 30)
    print(f"Precisão mínima para break-even: {min_roi:.1%}")
    print(f"Ratio custo FN/FP: {c_fn / c_fp:.1f}")
    print()

    for k, v in models_results.items():
        precision_event = v.get("business_metrics_event", {}).get("precision_event", 0)
        roi_event = v.get("business_metrics_event", {}).get("roi_pct", 0)
        status = "OK" if precision_event >= min_roi else "FAIL"

        print(f"{k}:")
        print(f"  Precisão: {precision_event:.1%} [{status}]")
        print(f"  ROI: {roi_event:.1%}")
        print()

    # Resumo estruturado para salvamento
    summary = {
        "models": {
            k: {
                "AUC": v["auc"],
                "AP": v["ap"],
                "threshold": float(v["threshold"]),
                "cost": float(v["cost"]),
                "roi_event": float(
                    v.get("business_metrics_event", {}).get("roi_pct", 0)
                ),
                "precision_event": float(
                    v.get("business_metrics_event", {}).get("precision_event", 0)
                ),
                "recall_event": float(
                    v.get("business_metrics_event", {}).get("recall_event", 0)
                ),
                "confusion_matrix": v["confusion_matrix"].tolist(),
            }
            for k, v in models_results.items()
        },
        "split_info": split_info,
        "selected_features_count": len(selected_features),
        "cv_results": cv_results.to_dict() if cv_results is not None else None,
        "cv_summary": cv_summary,
        "data_leakage_check": max(len(suspicious_features) - 1, 0)
        if suspicious_features is not None
        else 0,
    }

    return summary


def save_model_and_metrics(
    best_model_name,
    best_model_results,
    selected_features,
    summary,
    final_test_results,
    cv_results,
    cv_summary,
    suspicious_features,
    model_name="device_failure_model",
    save_dir="models",
):
    """
    Salva apenas o melhor modelo selecionado e suas métricas.

    Args:
        best_model_name: Nome do melhor modelo selecionado
        best_model_results: Resultados do melhor modelo
        selected_features: Lista de features selecionadas
        summary: Resumo final das métricas
        final_test_results: Resultados do teste final
        cv_results: Resultados do cross-validation
        cv_summary: Resumo do cross-validation
        suspicious_features: Features suspeitas de data leakage
        model_name: Nome do modelo para salvar
        save_dir: Diretório para salvar os arquivos
    """

    # Criar diretório se não existir
    os.makedirs(save_dir, exist_ok=True)

    # Timestamp para versionamento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 60}")
    print(f"SALVANDO MODELO E MÉTRICAS - {timestamp}")
    print(f"{'=' * 60}")

    # 1. Salvar apenas o melhor modelo (pickle)
    best_model_data = {
        "model": best_model_results.get("model"),
        "calibrator": best_model_results.get("calibrator"),
        "threshold": best_model_results.get("threshold"),
        "feature_names": selected_features,
        "model_type": best_model_name,
    }

    # Se for ensemble, salvar também os modelos individuais
    if best_model_name == "EnsembleAvg":
        # Para ensemble, incluir modelos individuais e calibradores
        best_model_data["xgb_model"] = best_model_results.get("xgb_model")
        best_model_data["lgb_model"] = best_model_results.get("lgb_model")
        best_model_data["xgb_calibrator"] = best_model_results.get("xgb_calibrator")
        best_model_data["lgb_calibrator"] = best_model_results.get("lgb_calibrator")
        print(
            "Ensemble salvo com modelos individuais: XGBoost + LightGBM + Calibradores"
        )

    # Salvar melhor modelo
    model_path = os.path.join(
        save_dir, f"{model_name}_{best_model_name.lower()}_{timestamp}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump(best_model_data, f)
    print(f"Melhor modelo salvo: {model_path}")

    # 2. Salvar métricas de performance (JSON)
    performance_metrics = {
        "timestamp": timestamp,
        "model_name": model_name,
        "best_model": best_model_name,
        "performance_summary": {
            "auc": float(best_model_results.get("auc", 0)),
            "ap": float(best_model_results.get("ap", 0)),
            "precision": float(best_model_results.get("precision", 0)),
            "recall": float(best_model_results.get("recall", 0)),
            "f1": float(best_model_results.get("f1", 0)),
            "cost": float(best_model_results.get("cost", 0)),
            "threshold": float(best_model_results.get("threshold", 0.5)),
            "metrics_source": best_model_results.get("metrics_source", "unknown"),
            "total_features": len(selected_features),
            "suspicious_features": len(suspicious_features)
            if suspicious_features is not None and len(suspicious_features) > 0
            else 0,
        },
        "final_test_results": final_test_results,
        "cv_results": cv_results.to_dict() if cv_results is not None else None,
        "cv_summary": cv_summary,
        "selected_features": selected_features,
    }

    # Salvar métricas de performance
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(performance_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"Métricas de performance salvas: {metrics_path}")

    # 3. Salvar resumo executivo (JSON)
    executive_summary = {
        "timestamp": timestamp,
        "model_name": model_name,
        "best_model": best_model_name,
        "performance_summary": {
            "auc": float(best_model_results.get("auc", 0)),
            "ap": float(best_model_results.get("ap", 0)),
            "precision": float(best_model_results.get("precision", 0)),
            "recall": float(best_model_results.get("recall", 0)),
            "f1": float(best_model_results.get("f1", 0)),
            "cost": float(best_model_results.get("cost", 0)),
            "threshold": float(best_model_results.get("threshold", 0.5)),
            "metrics_source": best_model_results.get("metrics_source", "unknown"),
            "total_features": len(selected_features),
            "suspicious_features": len(suspicious_features)
            if suspicious_features is not None and len(suspicious_features) > 0
            else 0,
        },
        "final_test_results": final_test_results,
        "file_paths": {
            "model": model_path,
            "metrics": metrics_path,
        },
    }

    # Salvar resumo executivo
    executive_path = os.path.join(save_dir, f"{model_name}_executive_{timestamp}.json")
    with open(executive_path, "w", encoding="utf-8") as f:
        json.dump(executive_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Resumo executivo salvo: {executive_path}")

    print("\nMelhor modelo e métricas salvos com sucesso!")
    print(f"Diretório: {save_dir}")
    print(f"Timestamp: {timestamp}")
    print("Arquivos criados:")
    print(f"   - {os.path.basename(model_path)} (melhor modelo: {best_model_name})")
    print(f"   - {os.path.basename(metrics_path)} (métricas de performance)")
    print(f"   - {os.path.basename(executive_path)} (resumo executivo)")

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "executive_path": executive_path,
        "timestamp": timestamp,
        "best_model_name": best_model_name,
    }
