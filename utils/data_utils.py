"""
Módulo de load e análise de dados
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def load_and_explore_data():
    """Carregar e explorar os dados"""
    print("=== CARREGANDO DADOS ===")
    df = pd.read_csv(
        "data/full_devices.csv", encoding="utf-8", encoding_errors="ignore"
    )

    print(f"Shape original: {df.shape}")
    print(f"Colunas: {list(df.columns)}")

    # Converter data para datetime
    df["date"] = pd.to_datetime(df["date"])

    # Ordenar por dispositivo e data
    df = df.sort_values(["device", "date"]).reset_index(drop=True)

    # Remover duplicatas se existirem
    duplicates = df.duplicated(subset=["device", "date"]).sum()
    if duplicates > 0:
        print(f"Removendo {duplicates} duplicatas...")
        df = df.drop_duplicates(subset=["device", "date"]).reset_index(drop=True)

    print(f"Shape após limpeza: {df.shape}")

    print("\n")
    print(f"{df.head()}")

    return df


def analyze_data_completeness(df):
    """Analisar completude dos dados"""
    print("\n=== ANÁLISE DE COMPLETUDE DOS DADOS ===")

    # Informações básicas
    total_devices = df["device"].nunique()
    total_dates = df["date"].nunique()
    expected_records = total_devices * total_dates
    actual_records = len(df)

    print(f"Total de registros: {len(df):,}")
    print(f"Número de dispositivos únicos: {total_devices:,}")
    print(f"Número de datas únicas: {total_dates:,}")
    print(f"Período dos dados: {df['date'].min()} at {df['date'].max()}")
    print(
        f"Registros esperados (todos dispositivos em todas as datas): {expected_records:,}"
    )
    print(f"Registros atuais: {actual_records:,}")
    print(f"Diferença: {expected_records - actual_records:,} registros faltando")
    print(f"Taxa de completude: {(actual_records / expected_records) * 100:.2f}%")
    print(f"Taxa de falhas: {df['failure'].mean():.4f}")

    # Análise por dispositivo
    device_counts = df.groupby("device").size().reset_index(name="record_count")
    complete_devices = (device_counts["record_count"] == total_dates).sum()
    print(
        f"\nDispositivos completos: {complete_devices} ({(complete_devices / total_devices) * 100:.2f}%)"
    )

    # Análise por data
    date_counts = df.groupby("date").size().reset_index(name="device_count")
    complete_dates = (date_counts["device_count"] == total_devices).sum()
    print(
        f"Datas completas: {complete_dates} ({(complete_dates / total_dates) * 100:.2f}%)"
    )

    return {
        "total_devices": total_devices,
        "total_dates": total_dates,
        "expected_records": expected_records,
        "actual_records": actual_records,
        "completeness_rate": (actual_records / expected_records) * 100,
        "complete_devices": complete_devices,
        "complete_dates": complete_dates,
    }


def analyze_failure_patterns(df):
    """Analisar padrões de falhas"""
    print("\n=== ANÁLISE DE PADRÕES DE FALHAS ===")

    # Distribuição de falhas
    failure_distribution = df["failure"].value_counts()
    failure_rate = (df["failure"].sum() / len(df)) * 100

    print("Distribuição de falhas:")
    print(f"  Sem falha (0): {failure_distribution[0]:,} ({100 - failure_rate:.2f}%)")
    print(f"  Com falha (1): {failure_distribution[1]:,} ({failure_rate:.2f}%)")

    # Falhas por dispositivo
    device_failures = (
        df.groupby("device")["failure"].agg(["sum", "count", "mean"]).reset_index()
    )
    device_failures.columns = [
        "device",
        "total_failures",
        "total_records",
        "failure_rate",
    ]
    device_failures = device_failures.sort_values("total_failures", ascending=False)

    print(f"\nDispositivos com falhas: {(device_failures['total_failures'] > 0).sum()}")
    print(f"Dispositivos sem falhas: {(device_failures['total_failures'] == 0).sum()}")
    print(
        f"Taxa média de falhas por dispositivo: {device_failures['failure_rate'].mean():.4f}"
    )

    # Análise temporal das falhas
    daily_failures = (
        df.groupby("date")["failure"].agg(["sum", "count", "mean"]).reset_index()
    )
    daily_failures.columns = ["date", "total_failures", "total_devices", "failure_rate"]

    print(f"\nDias com falhas: {(daily_failures['total_failures'] > 0).sum()}")
    print(f"Dias sem falhas: {(daily_failures['total_failures'] == 0).sum()}")
    print(f"Taxa média de falhas por dia: {daily_failures['failure_rate'].mean():.4f}")

    return device_failures, daily_failures


def check_data_drift(df_train, df_test, feature_cols):
    """
    Verificar se há drift nos dados entre treino e teste
    """
    print("\n=== VERIFICAÇÃO DE DATA DRIFT ===")

    drift_detected = []

    for col in feature_cols[:10]:
        try:
            stat, p_value = ks_2samp(df_train[col].dropna(), df_test[col].dropna())
            if p_value < 0.05:
                drift_detected.append((col, stat, p_value))
        except Exception:
            continue

    if drift_detected:
        print("  DRIFT DETECTADO nas features:")
        for col, stat, p_val in drift_detected[:5]:
            print(f"   {col}: KS-stat={stat:.4f}, p-value={p_val:.4f}")
    else:
        print("  Nenhum drift significativo detectado")

    return drift_detected


def remove_drift_features(df, drift_features, keep_normalized=True):
    """
    Remove features com drift forte, mantendo versões normalizadas se disponíveis
    """
    drift_cols = [f[0] for f in drift_features]

    # Features a remover
    to_remove = []
    for col in drift_cols:
        if col.startswith("attribute") and not col.endswith("_normalized"):
            to_remove.append(col)

    if keep_normalized:
        for col in to_remove:
            normalized_col = f"{col}_normalized"
            if normalized_col in df.columns:
                print(f"  Mantendo {normalized_col} (versão normalizada de {col})")

    # Remover features com drift
    df_clean = df.drop(columns=to_remove, errors="ignore")

    print(f"  Removidas {len(to_remove)} features com drift: {to_remove}")
    print(f"  Features restantes: {df_clean.shape[1]}")

    return df_clean


def verify_data_leakage(df_features, target_col):
    """
    Verificar se há data leakage nas features
    """
    print("\n=== VERIFICAÇÃO DE DATA LEAKAGE ===")

    feature_cols = [
        col for col in df_features.columns if col not in ["device", "date", target_col]
    ]

    print(f"Analisando {len(feature_cols)} features contra target '{target_col}'")

    correlations = df_features[feature_cols + [target_col]].corr()[target_col].abs()
    suspicious = correlations[correlations > 0.8].sort_values(ascending=False)

    if len(suspicious) > 1:  # Mais que 1 porque inclui a própria target
        print("  ATENÇÃO: Features com correlação suspeita:")
        for feat, corr in suspicious.items():
            if feat != target_col:
                print(f"   {feat}: {corr:.4f}")
    else:
        print("  Nenhuma correlação suspeita detectada")

    return suspicious


def analyze_false_negatives(df_test_meta, models_results, y_test):
    """
    Analisar quais tipos de falha o modelo está errando
    """
    print("\n=== ANÁLISE DE FALSOS NEGATIVOS ===")

    # Usar XGBoost como referência (melhor modelo)
    xgb_preds = models_results["XGBoost"]["predictions"]
    fn_mask = (y_test == 1) & (xgb_preds == 0)

    if fn_mask.sum() > 0:
        print(f"Falsos Negativos detectados: {fn_mask.sum()}")

        # Obter dados dos FNs usando índices corretos
        fn_indices = np.where(fn_mask)[0]
        fn_df = df_test_meta.iloc[fn_indices]

        # Padrões temporais
        print("\nDistribuição por dia da semana:")
        print(pd.to_datetime(fn_df["date"]).dt.dayofweek.value_counts().sort_index())

        # Padrões por dispositivo
        print("\nDispositivos com mais FNs:")
        print(fn_df["device"].value_counts().head(10))

        # Análise temporal
        print("\nPeríodo dos FNs:")
        print(f"  Início: {pd.to_datetime(fn_df['date']).min()}")
        print(f"  Fim: {pd.to_datetime(fn_df['date']).max()}")

    else:
        print("Nenhum falso negativo detectado")

    return fn_mask


def sensitivity_analysis(models_results, y_test, cost_ratio_range=[2, 3, 4, 5, 6]):
    """
    Analisar sensibilidade ao ratio de custo FN/FP
    """
    print("\n=== ANÁLISE DE SENSIBILIDADE AO CUSTO ===")

    results = []
    for cost_ratio in cost_ratio_range:
        row = {"cost_ratio": cost_ratio}

        for model_name, results_dict in models_results.items():
            cm = results_dict["confusion_matrix"]
            tn, fp, fn, tp = cm.ravel()

            cost = cost_ratio * fn + fp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            row[f"{model_name}_cost"] = cost
            row[f"{model_name}_recall"] = recall

        results.append(row)

    sensitivity_df = pd.DataFrame(results)
    print(sensitivity_df.round(3))

    return sensitivity_df
