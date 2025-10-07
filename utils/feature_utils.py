"""
Módulo para feature engineering e análise de features
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
)
from sklearn.base import BaseEstimator, TransformerMixin


def make_early_warning_labels_safe(df, target_col="failure", horizon_days=10):
    """
    Label de early warning sem perder negativos:
      - 1 se a próxima falha ocorre em <= H dias
      - 0 se sabemos que não ocorre falha nos próximos H dias
      - descarta apenas as últimas (H-1) observações de cada device
    """
    df = df.sort_values(["device", "date"]).copy()

    # próxima data de falha conhecida
    df["failure_date_tmp"] = df["date"].where(df[target_col] == 1)
    df["next_failure_date"] = df.groupby("device")["failure_date_tmp"].transform(
        lambda s: s[::-1].ffill()[::-1]
    )
    df.drop(columns=["failure_date_tmp"], inplace=True)

    delta = (df["next_failure_date"] - df["date"]).dt.days

    # data máxima por device
    last_date_by_dev = df.groupby("device")["date"].transform("max")
    safe_cutoff = last_date_by_dev - pd.Timedelta(days=horizon_days)

    # regra de rotulagem
    pos = (delta.notna()) & (delta <= horizon_days)
    is_failure_day = df[target_col] == 1
    safe_neg = (df["date"] <= safe_cutoff) & (~pos) & (~is_failure_day)

    # mantemos apenas linhas rotuláveis (positivas ou negativas)
    keep = pos | safe_neg
    df = df[keep].copy()

    pos_kept = pos[keep]
    df["early_warn_target"] = np.where(pos_kept, 1, 0)
    df.drop(columns=["next_failure_date"], inplace=True)
    return df, "early_warn_target"


def create_features(
    df, target_col="failure", forecast_horizon=7, enable_interactions=True, max_lags=7
):
    """
    Função completa de feature engineering para dados de sensores esparsos
    """
    print("\n=== INICIANDO FEATURE ENGINEERING ===")

    df_feat = df.copy()

    # 1. FEATURES BÁSICAS
    print("1. Criando features básicas...")

    # Features de data
    df_feat["year"] = df_feat["date"].dt.year
    df_feat["dayofweek"] = df_feat["date"].dt.dayofweek

    # Codificação cíclica para features temporais
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.month / 12)
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.day / 31)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.day / 31)
    df_feat["doy_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.dayofyear / 365)
    df_feat["doy_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.dayofyear / 365)
    df_feat["week_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.isocalendar().week / 53)
    df_feat["week_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.isocalendar().week / 53)
    df_feat["quarter_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.quarter / 4)
    df_feat["quarter_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.quarter / 4)

    # Features de sazonalidade
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)
    df_feat["is_month_start"] = df_feat["date"].dt.is_month_start.astype(int)
    df_feat["is_month_end"] = df_feat["date"].dt.is_month_end.astype(int)
    df_feat["is_quarter_start"] = df_feat["date"].dt.is_quarter_start.astype(int)
    df_feat["is_quarter_end"] = df_feat["date"].dt.is_quarter_end.astype(int)

    # 2. FEATURES DE DISPOSITIVO
    print("2. Criando features de dispositivo...")

    # Histórico de falhas por dispositivo - acumula apenas falhas do passado
    past_fail = df_feat.groupby("device")[target_col].shift(1).fillna(0)
    df_feat["device_total_failures"] = past_fail.groupby(df_feat["device"]).cumsum()

    # Tempo desde a última falha - shift(1) antes do ffill
    df_feat["failure_date"] = df_feat["date"].where(df_feat[target_col] == 1)
    df_feat["failure_date_shifted"] = df_feat.groupby("device")["failure_date"].shift(1)
    df_feat["last_failure_date"] = df_feat.groupby("device")[
        "failure_date_shifted"
    ].ffill()
    df_feat["days_since_last_failure"] = (
        df_feat["date"] - df_feat["last_failure_date"]
    ).dt.days
    df_feat["days_since_last_failure"] = df_feat["days_since_last_failure"].fillna(999)
    df_feat.drop(
        columns=["failure_date", "failure_date_shifted", "last_failure_date"],
        inplace=True,
    )

    # 3. FEATURES TEMPORAIS
    print("3. Criando features temporais...")

    attribute_cols = [col for col in df_feat.columns if col.startswith("attribute")]

    # Lags
    lag_windows = [1, 2, 3, 7] if max_lags <= 7 else [1, 2, 3, 7, 14, 30]

    for attr in attribute_cols:
        # Lags
        for lag in lag_windows:
            df_feat[f"{attr}_lag_{lag}"] = df_feat.groupby("device")[attr].shift(lag)

        # Sempre manter apenas o passado
        past = df_feat.groupby("device")[attr].shift(1)

        # Médias móveis
        for window in [3, 7, 14]:
            df_feat[f"{attr}_ma_{window}"] = (
                past.groupby(df_feat["device"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Desvio padrão móvel
        for window in [7, 14]:
            df_feat[f"{attr}_std_{window}"] = (
                past.groupby(df_feat["device"])
                .rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

        # Diferenças
        df_feat[f"{attr}_diff"] = df_feat.groupby("device")[attr].diff()
        df_feat[f"{attr}_pct_diff"] = (
            df_feat.groupby("device")[attr]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )

    # 4. FEATURES DE CONTEXTO
    print("4. Criando features de contexto...")

    # Densidade de dados por dispositivo
    df_feat["device_obs_so_far"] = (df_feat.groupby("device").cumcount()).astype(
        "uint32"
    )

    # Dias sem dados
    df_feat["days_since_last_record"] = (
        df_feat.groupby("device")["date"].diff().dt.days.fillna(0)
    )

    # Variabilidade dos atributos - selecionei os 4 primeiros apenas por tempo de execução
    for attr in attribute_cols[:4]:
        past = df_feat.groupby("device")[attr].shift(1)
        df_feat[f"{attr}_std_expanding"] = (
            past.groupby(df_feat["device"])
            .expanding(min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

    # 5. FEATURES DE INTERAÇÃO
    if enable_interactions:
        print("5. Criando features de interação...")

        # Apenas pares mais importantes
        important_attrs = attribute_cols[:4]

        for i, attr1 in enumerate(important_attrs):
            for attr2 in important_attrs[i + 1 :]:
                df_feat[f"{attr1}_{attr2}_sum"] = df_feat[attr1] + df_feat[attr2]
                df_feat[f"{attr1}_{attr2}_diff"] = df_feat[attr1] - df_feat[attr2]
                df_feat[f"{attr1}_{attr2}_ratio"] = df_feat[attr1] / (
                    df_feat[attr2] + 1e-8
                )
    else:
        print("5. Pulando features de interação...")

    # 6. FEATURES DE TENDÊNCIA
    print("6. Criando features de tendência...")

    # Apenas para atributos mais importantes e janelas menores
    important_attrs = attribute_cols[:4]
    trend_windows = [7, 14]

    for attr in important_attrs:
        past = df_feat.groupby("device")[attr].shift(1)
        for window in trend_windows:
            col = f"{attr}_slope_{window}"
            df_feat[col] = (
                past.groupby(df_feat["device"])
                .rolling(window, min_periods=2)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                    if np.isfinite(x).sum() >= 2
                    else np.nan,
                    raw=False,
                )
                .reset_index(level=0, drop=True)
            )

    # 7. OTIMIZAÇÃO DOS TIPOS DE DADOS
    print("7. Otimizando tipos de dados...")
    for c in [
        "year",
        "dayofweek",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "doy_sin",
        "doy_cos",
        "week_sin",
        "week_cos",
        "quarter_sin",
        "quarter_cos",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
    ]:
        if c in df_feat:
            if c in ["year"]:
                df_feat[c] = df_feat[c].astype("uint16")
            elif c in [
                "dayofweek",
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
            ]:
                df_feat[c] = df_feat[c].astype("uint8")
            else:
                df_feat[c] = df_feat[c].astype("float32")

    # 8. NORMALIZAÇÃO POR DISPOSITIVO
    print("8. Aplicando normalização por dispositivo...")
    df_feat = normalize_features_by_device(
        df_feat,
        device_col="device",
        features_to_normalize=[f"attribute{i}" for i in range(1, 10)],
        window_size=30,
    )

    # Indicadores de missing para cada atributo original
    for i in range(1, 10):
        col = f"attribute{i}"
        if col in df_feat:
            df_feat[f"{col}_isna"] = df_feat[col].isna().astype("uint8")

    # Regra dos 2 sigma para detectar anomalias
    for i in range(1, 10):
        colz = f"attribute{i}_normalized"
        if colz in df_feat:
            df_feat[f"{colz}_gt2sigma"] = (df_feat[colz] > 2).astype("uint8")
            df_feat[f"{colz}_lt-2sigma"] = (df_feat[colz] < -2).astype("uint8")

    # 9. CRIAÇÃO DA TARGET - HORIZONTE DE PREVISÃO
    print(f"9. Criando target com horizonte de {forecast_horizon} dias...")
    df_feat, new_target_col = make_early_warning_labels_safe(
        df_feat, target_col, forecast_horizon
    )

    # 10. REMOVER VAZAMENTO
    print("10. Removendo vazamento de target...")
    if "failure" in df_feat.columns:
        df_feat.drop(columns=["failure"], inplace=True)
        print("  Removido 'failure' das features (vazamento de target)")

    print("=== FEATURE ENGINEERING FINALIZADO ===")
    print(f"Shape final: {df_feat.shape}")
    print(f"Número de features: {df_feat.shape[1]}")

    return df_feat, new_target_col


def normalize_features_by_device(
    df_features, device_col="device", features_to_normalize=None, window_size=30
):
    """
    Normaliza features por dispositivo com z-score rolling.
    """
    print("=== NORMALIZAÇÃO POR DISPOSITIVO (rolling z-score) ===")

    if features_to_normalize is None:
        features_to_normalize = [
            "attribute1",
            "attribute2",
            "attribute3",
            "attribute4",
            "attribute5",
            "attribute6",
            "attribute7",
            "attribute8",
            "attribute9",
        ]

    df = df_features.copy()

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values([device_col, "date"], inplace=True)

    for feat in features_to_normalize:
        if feat not in df.columns:
            continue

        s = df[feat]

        roll_mean = (
            s.groupby(df[device_col])
            .apply(lambda x: x.shift(1).rolling(window_size, min_periods=5).mean())
            .reset_index(level=0, drop=True)
        )
        roll_std = (
            s.groupby(df[device_col])
            .apply(lambda x: x.shift(1).rolling(window_size, min_periods=5).std())
            .reset_index(level=0, drop=True)
        )

        z = (s - roll_mean) / roll_std
        df[f"{feat}_normalized"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  {feat}: normalizado (rolling={window_size}, min_periods=5)")

    return df


def analyze_feature_correlations(X, y, corr_threshold=0.95, target_threshold=0.01):
    """
    Análise de correlações entre features e com target
    """

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]

    # correlação com target
    tgt = X.assign(_y=y.astype(float))
    tgt_corr = tgt.corr()["_y"].drop("_y").to_dict()
    keep = [
        c
        for c in X.columns
        if (c not in to_drop) and (abs(tgt_corr.get(c, 0)) >= target_threshold)
    ]

    return {
        "features_to_remove": to_drop,
        "target_correlations": tgt_corr,
        "recommended_features": keep,
    }


def analyze_feature_variance(X, variance_threshold=0.01):
    """
    Analisar variância das features
    """
    print("\n=== ANÁLISE DE VARIÂNCIA ===")

    # Calcular variância de cada feature
    feature_variance = X.var()

    # Features com baixa variância
    low_variance_features = feature_variance[
        feature_variance < variance_threshold
    ].index.tolist()

    print(
        f"Features com variância < {variance_threshold}: {len(low_variance_features)}"
    )

    if low_variance_features:
        print("Top 10 features com menor variância:")
        for feat in feature_variance.nsmallest(10).index:
            print(f"  {feat}: {feature_variance[feat]:.6f}")

    return low_variance_features


def statistical_feature_tests(X, y, max_features=2000, random_state=42):
    """
    Retorna scores F-test e Mutual Information em DataFrames ordenados (desc).
    """

    cols = X.columns[:max_features].tolist()
    X_ = X[cols]
    y_ = y.astype(int)

    # F-test (linear)
    f_scores, f_pvalues = f_classif(X_, y_)
    df_f = (
        pd.DataFrame({"feature": cols, "f_score": f_scores, "p_value": f_pvalues})
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )

    # Mutual Information
    mi_scores = mutual_info_classif(
        X_, y_, random_state=random_state, discrete_features="auto"
    )
    df_mi = (
        pd.DataFrame({"feature": cols, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    return df_f, df_mi


class SafeDeviceImputer(BaseEstimator, TransformerMixin):
    """
    Imput seguro por dispositivo usando apenas dados do passado
    """

    def __init__(
        self, device_col="device", window=14, min_periods=3, feature_prefix="attribute"
    ):
        self.device_col = device_col
        self.window = window
        self.min_periods = min_periods
        self.feature_prefix = feature_prefix
        self.cols_ = None

    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns if c.startswith(self.feature_prefix)]
        return self

    def transform(self, X):
        df = X.copy()
        for c in self.cols_:
            past = df.groupby(self.device_col)[c].shift(1)
            med = (
                past.groupby(df[self.device_col])
                .rolling(self.window, min_periods=self.min_periods)
                .median()
                .reset_index(level=0, drop=True)
            )
            df[c] = df[c].fillna(med)
        return df
