"""
Módulo para treinamento e avaliação de modelos
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    average_precision_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from .business_utils import cm_safe
from .alert_utils import apply_simple_threshold


def as_df(X, cols):
    """Converte array para DataFrame com colunas consistentes"""
    return X if hasattr(X, "columns") else pd.DataFrame(X, columns=cols)


def debug_probiles(p, name):
    """Debug rápido de distribuição dos scores"""
    qs = [0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    print(f"\n{'=' * 30}")
    print(name.center(30, "="))
    print({q: float(np.quantile(p, q)) for q in qs})


def pick_threshold_on_val(y_val, proba_val, fn_cost=4, fp_cost=1):
    """
    Encontrar threshold ótimo por custo na VALIDAÇÃO (evita olhar o teste)
    """
    thresholds = np.linspace(0, 1, 101)
    best_cost = float("inf")
    best_thr = 0.5

    for thr in thresholds:
        preds = (proba_val >= thr).astype(int)
        tn, fp, fn, tp = cm_safe(y_val, preds).ravel()
        cost = fn_cost * fn + fp_cost * fp

        if cost < best_cost:
            best_cost = cost
            best_thr = thr

    return best_thr, best_cost


def pick_threshold_with_budget(y_val, p_val, fn_cost=4, fp_cost=1, max_alert_rate=0.05):
    """
    Escolhe threshold otimizando custo com fallback por orçamento de alertas.
    """
    thrs = np.unique(np.r_[0.0, np.quantile(p_val, np.linspace(0, 1, 201)), 1.0])
    costs = []

    for t in thrs:
        pred = (p_val >= t).astype(int)
        tn, fp, fn, tp = cm_safe(y_val, pred).ravel()
        cost = fn_cost * fn + fp_cost * fp
        alert_rate = pred.mean()
        costs.append({"thr": t, "cost": cost, "alert_rate": alert_rate})

    costs_df = pd.DataFrame(costs)
    best_cost_idx = costs_df["cost"].idxmin()
    best_cost_thr = costs_df.loc[best_cost_idx, "thr"]
    best_cost_alert_rate = costs_df.loc[best_cost_idx, "alert_rate"]

    # Se o threshold ótimo gera muitos alertas, usar fallback por orçamento
    if best_cost_alert_rate > max_alert_rate:
        # Encontrar threshold que respeita o orçamento
        budget_thr = float(np.quantile(p_val, 1.0 - max_alert_rate))
        return budget_thr, max_alert_rate, "budget_fallback"
    else:
        return best_cost_thr, best_cost_alert_rate, "cost_optimal"


def apply_undersampling(X, y, random_state=42, neg_per_pos=50, max_neg_cap=None):
    """
    Aplicar undersampling balanceado com controle de taxa neg/pos e cap opcional.
    """
    print("\n=== APLICANDO UNDERSAMPLING ===")
    print(
        f"Dataset original: {len(X)} amostras | dist: {dict(y.value_counts())} | rate={y.mean():.4f}"
    )

    rng = np.random.RandomState(random_state)

    # Encontrar ndices de classes
    pos_mask = y == 1
    neg_mask = y == 0
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0:
        print("WARNING - Sem positivos no treino; pulando undersampling.")
        return X, y

    target_neg = min(n_neg, neg_per_pos * n_pos)
    if max_neg_cap is not None:
        target_neg = min(target_neg, max_neg_cap)

    if target_neg >= n_neg:
        print(
            f"INFO - Nenhum corte aplicado (target_neg={target_neg} >= n_neg={n_neg}). "
            f"Tente reduzir neg_per_pos (ex.: 10/20) ou definir max_neg_cap (ex.: 50_000)."
        )
        X_bal, y_bal = X, y
    else:
        # Amostrar negativos
        sampled_neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)

        # Combinar positivos e negativos amostrados
        keep_idx = np.concatenate([pos_idx, sampled_neg_idx])

        # Selecionar dados mantendo ordem temporal
        if isinstance(X, pd.DataFrame):
            X_bal = X.iloc[keep_idx]
        else:
            X_bal = X[keep_idx]
        if hasattr(y, "iloc"):
            # y  uma Series do pandas
            y_bal = y.iloc[keep_idx]
        else:
            # y  um numpy array
            y_bal = y[keep_idx]

    orig_ratio = (y == 0).sum() / max(1, (y == 1).sum())
    new_ratio = (y_bal == 0).sum() / max(1, (y_bal == 1).sum())
    print(
        f"US: {len(X)} -> {len(X_bal)} | pos={int((y_bal == 1).sum())} neg={int((y_bal == 0).sum())} "
        f"(neg/pos~{new_ratio:.1f}, antes~{orig_ratio:.1f})"
    )
    return X_bal, y_bal


def undersample_hard_negatives(X, y, dates, device, neg_per_pos=10, random_state=42):
    """
    Undersampling com pesos por proximidade da próxima falha.
    Funciona com X DataFrame ou ndarray; y pode ser Series ou ndarray.
    """
    print("\n=== UNDERSAMPLING HARD NEGATIVES ===")
    rng = np.random.default_rng(random_state)

    # normalizar tipos
    y = pd.Series(y).reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    device = pd.Series(device).reset_index(drop=True)

    idx_pos = y[y == 1].index
    idx_neg = y[y == 0].index
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        print("  Aviso: não há positivos/negativos suficientes; retornando original.")
        return X, y

    df_tmp = pd.DataFrame({"date": dates, "device": device, "y": y})
    df_tmp["fail_date"] = df_tmp["date"].where(df_tmp["y"] == 1)
    next_fail = df_tmp.groupby("device")["fail_date"].transform(
        lambda s: s[::-1].ffill()[::-1]
    )
    dist = (next_fail - df_tmp["date"]).dt.days.clip(lower=0).fillna(9_999)
    w = 1.0 / (1.0 + dist)

    n_pos = len(idx_pos)
    target_neg = min(len(idx_neg), neg_per_pos * n_pos)

    prob = (w.loc[idx_neg] / w.loc[idx_neg].sum()).to_numpy()
    sampled_neg = pd.Index(
        rng.choice(idx_neg.to_numpy(), size=target_neg, replace=False, p=prob)
    )
    keep = idx_pos.union(sampled_neg)

    # retornar X no tipo original mantendo ordem temporal
    if isinstance(X, pd.DataFrame):
        return X.loc[keep], y.loc[keep]
    else:
        return X[keep], y.loc[keep]


def calibrate_single_block(model, X_cal, y_cal, X_test, method="logistic"):
    """
    Calibração isotônica ou logística em um único bloco
    """
    p_cal_raw = model.predict_proba(X_cal)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_cal_raw, y_cal.astype(int))
        p_test_cal = cal.transform(p_test_raw)
    else:
        cal = LogisticRegression(max_iter=1000)
        cal.fit(p_cal_raw.reshape(-1, 1), y_cal.astype(int))
        p_test_cal = cal.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    return p_test_cal, cal


def timeaware_calibration(model, X_val, y_val, X_test, n_splits=5, method="logistic"):
    """Mantida para compatibilidade, mas recomenda-se usar calibrate_single_block"""

    # Garante DataFrame para indexação temporal estável
    Xv = pd.DataFrame(X_val) if not hasattr(X_val, "iloc") else X_val
    Xt = pd.DataFrame(X_test) if not hasattr(X_test, "iloc") else X_test

    # Preserva nomes de features se disponíveis
    if hasattr(X_val, "columns"):
        Xv.columns = X_val.columns
    if hasattr(X_test, "columns"):
        Xt.columns = X_test.columns

    yv = y_val if isinstance(y_val, pd.Series) else pd.Series(y_val, index=Xv.index)

    n = len(Xv)
    idx = np.arange(n)  # assume Xv já ordenado no tempo
    folds = np.array_split(idx, n_splits)

    p_val = np.zeros(n, dtype=float)
    cal_models = []

    for val_idx in folds:
        tr_idx = np.setdiff1d(idx, val_idx)
        p_tr_raw = model.predict_proba(Xv.iloc[tr_idx])[:, 1]
        p_va_raw = model.predict_proba(Xv.iloc[val_idx])[:, 1]

        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_tr_raw, yv.iloc[tr_idx].astype(int))
            p_val[val_idx] = ir.transform(p_va_raw)
            cal_models.append(ir)
        else:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(p_tr_raw.reshape(-1, 1), yv.iloc[tr_idx].astype(int))
            p_val[val_idx] = lr.predict_proba(p_va_raw.reshape(-1, 1))[:, 1]
            cal_models.append(lr)

    p_test_raw = model.predict_proba(Xt)[:, 1]
    if method == "isotonic":
        cal_preds = [cm.transform(p_test_raw) for cm in cal_models]
    else:
        cal_preds = [
            cm.predict_proba(p_test_raw.reshape(-1, 1))[:, 1] for cm in cal_models
        ]

    p_test = np.mean(cal_preds, axis=0)
    return p_val, p_test, cal_models


def temporal_splits_with_purge(df, purge_days=7, val_size=0.2, test_size=0.2):
    """
    Split temporal com purge para evitar contágio entre conjuntos
    Esta técnica perdemos dias de purge, porém cria gaps entre nossos splits
    """
    max_date, min_date = df["date"].max(), df["date"].min()
    span_days = (max_date - min_date).days
    test_cut = max_date - pd.Timedelta(days=int(span_days * test_size))
    val_cut = test_cut - pd.Timedelta(days=int(span_days * val_size))

    train_end = val_cut - pd.Timedelta(days=purge_days)
    val_start = val_cut + pd.Timedelta(days=purge_days)
    val_end = test_cut - pd.Timedelta(days=purge_days)
    test_start = test_cut + pd.Timedelta(days=purge_days)

    train = df["date"] <= train_end
    val = (df["date"] >= val_start) & (df["date"] <= val_end)
    test = df["date"] >= test_start

    return train, val, test


def create_pipeline_with_metadata(df_features, target_col="failure"):
    """
    Pipeline seguro que preserva metadados (dates, devices)
    """
    print("\n=== PIPELINE SEGURO COM METADADOS ===")

    # 1. Split temporal PRIMEIRO
    train_mask, val_mask, test_mask = temporal_splits_with_purge(
        df_features, purge_days=7
    )

    # 2. Separar conjuntos preservando metadados
    df_train = df_features[train_mask].copy()
    df_val = df_features[val_mask].copy()
    df_test = df_features[test_mask].copy()

    print("Split temporal:")
    print(f"  Train: {df_train.shape[0]:,} samples")
    print(f"  Val: {df_val.shape[0]:,} samples")
    print(f"  Test: {df_test.shape[0]:,} samples")

    # 3. Extrair os arrays
    dates_train = df_train["date"].values
    devices_train = df_train["device"].values
    dates_val = df_val["date"].values
    devices_val = df_val["device"].values
    dates_test = df_test["date"].values
    devices_test = df_test["device"].values

    # 4. Feature selection no treino
    excluded = ["device", "date", target_col, "failure"]
    feature_cols = [col for col in df_train.columns if col not in excluded]
    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_val, y_val = df_val[feature_cols], df_val[target_col]
    X_test, y_test = df_test[feature_cols], df_test[target_col]

    print(f"Features disponíveis: {len(feature_cols)}")

    # 5. Criando pipeline
    print("Aplicando pipeline...")
    pipeline_seguro = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "sfm",
                SelectFromModel(
                    lgb.LGBMClassifier(n_estimators=400, random_state=42, verbosity=-1),
                    max_features=80,
                    threshold=-np.inf,
                ),
            ),
        ]
    )

    # Selecionando as features com pipeline no treino
    X_train_proc = pipeline_seguro.fit_transform(X_train, y_train)
    X_val_proc = pipeline_seguro.transform(X_val)
    X_test_proc = pipeline_seguro.transform(X_test)

    # Obter features selecionadas
    selected_mask = pipeline_seguro.named_steps["sfm"].get_support()
    selected_features = [
        feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]
    ]

    print(f"Features selecionadas: {len(selected_features)}")

    # 6. Separação do modelo para calibração # 20% modelo e 80% calibração
    n = X_val_proc.shape[0]
    cut = int(n * 0.2)

    X_val_model, y_val_model = X_val_proc[:cut], y_val[:cut]
    X_val_hold, y_val_hold = X_val_proc[cut:], y_val[cut:]

    dates_val_model = dates_val[:cut]
    devices_val_model = devices_val[:cut]
    dates_val_hold = dates_val[cut:]
    devices_val_hold = devices_val[cut:]

    print("Validação separada:")
    print(f"  Modelo: {X_val_model.shape[0]:,} samples")
    print(f"  Calibração: {X_val_hold.shape[0]:,} samples")

    return (
        X_train_proc,
        y_train,
        X_val_model,
        y_val_model,
        X_val_hold,
        y_val_hold,
        X_test_proc,
        y_test,
        selected_features,
        dates_train,
        devices_train,
        dates_val_model,
        devices_val_model,
        dates_val_hold,
        devices_val_hold,
        dates_test,
        devices_test,
    )


def temporal_cv_evaluation(df_features, target_col="failure", n_splits=3, purge_days=7):
    """
    Validação cruzada temporal com purge (sem cooldown)
    """
    print(f"\n=== ROLLING ORIGIN CV (n={n_splits}) COM PURGE ===")

    df = df_features.sort_values(["date"]).reset_index(drop=True)
    max_date, min_date = df["date"].max(), df["date"].min()
    span_days = (max_date - min_date).days

    # Criar splits baseados em percentuais
    rows = []
    for fold in range(1, n_splits + 1):
        # Calcular cortes
        val_size = 0.2
        test_size = 0.2

        # Ajustar para fold específico
        fold_ratio = fold / n_splits
        test_cut = max_date - pd.Timedelta(days=int(span_days * test_size * fold_ratio))
        val_cut = test_cut - pd.Timedelta(days=int(span_days * val_size))

        # Aplicar purge
        train_end = val_cut - pd.Timedelta(days=purge_days)
        val_start = val_cut + pd.Timedelta(days=purge_days)
        val_end = test_cut - pd.Timedelta(days=purge_days)

        # Criar máscaras
        mask_tr = df["date"] <= train_end
        mask_va = (df["date"] >= val_start) & (df["date"] <= val_end)

        dtr = df[mask_tr]
        dva = df[mask_va]

        feat_cols = [c for c in df.columns if c not in ["device", "date", target_col]]
        X_tr, y_tr = dtr[feat_cols], dtr[target_col]
        X_va, y_va = dva[feat_cols], dva[target_col]

        # Pipeline
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "sfm",
                    SelectFromModel(
                        lgb.LGBMClassifier(
                            n_estimators=400, random_state=42, verbosity=-1
                        ),
                        max_features=80,
                        threshold=-np.inf,
                    ),
                ),
            ]
        )

        X_tr_proc = pipeline.fit_transform(X_tr, y_tr)
        X_va_proc = pipeline.transform(X_va)

        mdl = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        mdl.fit(X_tr_proc, y_tr)
        p_va = mdl.predict_proba(X_va_proc)[:, 1]

        # threshold por orçamento de alertas (ex.: 5%)
        thr = float(np.quantile(p_va, 1 - 0.05))
        preds = apply_simple_threshold(p_va, thr=thr)

        tn, fp, fn, tp = cm_safe(y_va, preds).ravel()
        cost = 4 * fn + fp
        ap = average_precision_score(y_va, p_va) if y_va.sum() > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rows.append({"fold": fold, "cost": cost, "ap": ap, "recall": recall})

        print(f"Fold {fold}: cost={cost:.1f} | AP={ap:.4f} | recall={recall:.3f}")

    df_cv = pd.DataFrame(rows)
    print("Médias:", df_cv.mean(numeric_only=True).to_dict())

    # Print do resumo dos conjuntos
    print("\nPipeline:")
    print(f"  Train: {X_tr.shape[0]:,} samples ({y_tr.mean():.4f} failure rate)")
    print(f"  Val: {X_va.shape[0]:,} samples ({y_va.mean():.4f} failure rate)")
    print(f"  Features selecionadas: {X_tr_proc.shape[1]}")

    # Calcular estatísticas resumidas
    cv_summary = {
        "mean_cost": df_cv["cost"].mean(),
        "mean_ap": df_cv["ap"].mean(),
        "mean_recall": df_cv["recall"].mean(),
        "std_cost": df_cv["cost"].std(),
        "std_ap": df_cv["ap"].std(),
        "std_recall": df_cv["recall"].std(),
        "best_fold": df_cv.loc[df_cv["cost"].idxmin(), "fold"],
        "best_cost": df_cv["cost"].min(),
    }

    return {
        "cv_results": df_cv,
        "cv_summary": cv_summary,
        "n_splits": n_splits,
        "purge_days": purge_days,
    }


def load_saved_model(models_path, model_name=None):
    """
    Carrega um modelo salvo para fazer predições

    Args:
        models_path: Caminho para o arquivo de modelos (.pkl)
        model_name: Nome do modelo específico a carregar (se None, retorna todos)

    Returns:
        Dicionário com os modelos carregados
    """
    import pickle

    with open(models_path, "rb") as f:
        models = pickle.load(f)

    if model_name is not None:
        return models.get(model_name)

    return models


def load_metrics(metrics_path):
    """
    Carrega as métricas salvas de um modelo

    Args:
        metrics_path: Caminho para o arquivo de métricas (.json)

    Returns:
        Dicionário com as métricas carregadas
    """
    import json

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics
