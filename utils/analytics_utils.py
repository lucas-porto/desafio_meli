import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega os dados do arquivo CSV.
    
    Args:
        caminho: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    print("Carregando dados...")
    df = pd.read_csv(caminho)
    print(f"Dados carregados com sucesso! Shape: {df.shape}")
    return df


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria todas as features derivadas e KPIs a partir do dataframe original.
    
    Args:
        df: DataFrame original com os dados brutos
        
    Returns:
        DataFrame com todas as features criadas
        
    Features criadas:
        - Temporais: DURATION_HOURS, START_HOUR, WEEKDAY, WEEKEND, DURATION_BUCKET, DAYPART
        - Vendas: STOCK_SOLD, SALES_RATE, PERFORMANCE_CATEGORY
        - Preços: PRICE_PER_UNIT, VALUE_PER_STOCK_UNIT, ASP
        - Potencial: POTENTIAL_VALUE, SALES_EFFICIENCY
        - Overselling: OVERSELLING_MARGIN, OVERSELL_FLAG, OVERSELL_UNITS, STOCKOUT_FLAG
        - Qualidade: STOCK_SOLD_calc, SOLD_UNITS_MISMATCH, SELL_THROUGH, GMV_PER_STOCK
    """
    print("\n## Ajuste dos tipos de dados")
    
    colunas_data = ['OFFER_START_DATE', 'OFFER_START_DTTM', 'OFFER_FINISH_DTTM']
    colunas_numericas = ['INVOLVED_STOCK', 'REMAINING_STOCK_AFTER_END', 'SOLD_AMOUNT', 'SOLD_QUANTITY']
    colunas_categoricas = ['OFFER_TYPE', 'ORIGIN', 'SHIPPING_PAYMENT_TYPE', 'DOM_DOMAIN_AGG1', 'VERTICAL', 'DOMAIN_ID']
    
    df_tratado = df.copy()
    
    # Converter colunas de data
    for col in colunas_data:
        if col in df_tratado.columns:
            try:
                df_tratado[col] = pd.to_datetime(df_tratado[col], errors='coerce')
            except Exception as e:
                print(f'Erro ao converter {col}: {e}')
    
    # Converter colunas numéricas
    for col in colunas_numericas:
        if col in df_tratado.columns:
            try:
                df_tratado[col] = pd.to_numeric(df_tratado[col], errors='coerce')
            except Exception as e:
                print(f'Erro ao converter {col}: {e}')
    
    # Verificar overselling nos dados originais
    if 'REMAINING_STOCK_AFTER_END' in df_tratado.columns:
        negativos = (df_tratado['REMAINING_STOCK_AFTER_END'] < 0).sum()
        if negativos > 0:
            print(f"   Encontrados {negativos:,} valores negativos em REMAINING_STOCK_AFTER_END")
            print(f"   Estes valores indicam overselling (vendas além do estoque disponível)")
    
    # ==== 1. Features Temporais ====
    print("\n## Criando features temporais")
    
    # Duração da oferta
    if 'OFFER_START_DTTM' in df_tratado.columns and 'OFFER_FINISH_DTTM' in df_tratado.columns:
        df_tratado['DURATION_HOURS'] = (df_tratado['OFFER_FINISH_DTTM'] - df_tratado['OFFER_START_DTTM']).dt.total_seconds() / 3600
    
    # Features temporais finas
    if 'OFFER_START_DTTM' in df_tratado.columns:
        df_tratado['START_HOUR'] = df_tratado['OFFER_START_DTTM'].dt.hour
        df_tratado['WEEKDAY'] = df_tratado['OFFER_START_DTTM'].dt.dayofweek  # 0=Mon
        df_tratado['WEEKEND'] = df_tratado['WEEKDAY'].isin([5,6]).astype(int)
    
    # Bucket de duração
    def _bucket_duration(h):
        if pd.isna(h): return np.nan
        if h <= 2: return 'curta(≤2h)'
        if h <= 6: return 'media(2–6h)'
        return 'longa(>6h)'
    
    if 'DURATION_HOURS' in df_tratado.columns:
        df_tratado['DURATION_BUCKET'] = df_tratado['DURATION_HOURS'].apply(_bucket_duration)
    
    # Período do dia (daypart)
    def _daypart(hour):
        if pd.isna(hour): return np.nan
        h = int(hour)
        if 0 <= h < 6: return 'madrugada'
        if 6 <= h < 12: return 'manha'
        if 12 <= h < 18: return 'tarde'
        return 'noite'
    
    if 'START_HOUR' in df_tratado.columns:
        df_tratado['DAYPART'] = df_tratado['START_HOUR'].apply(_daypart)
    
    # ==== 2. Features de Vendas ====
    print("\n## Criando features de vendas")
    
    # Estoque vendido
    if 'INVOLVED_STOCK' in df_tratado.columns and 'REMAINING_STOCK_AFTER_END' in df_tratado.columns:
        df_tratado['STOCK_SOLD'] = df_tratado['INVOLVED_STOCK'] - df_tratado['REMAINING_STOCK_AFTER_END']
    
    # Taxa de venda
    if 'SOLD_QUANTITY' in df_tratado.columns and 'INVOLVED_STOCK' in df_tratado.columns:
        df_tratado['SALES_RATE'] = df_tratado['SOLD_QUANTITY'] / df_tratado['INVOLVED_STOCK']
        df_tratado['SALES_RATE'] = df_tratado['SALES_RATE'].fillna(0)
    
    # Categoria de performance
    if 'SALES_RATE' in df_tratado.columns:
        def categorizar_performance(sales_rate):
            if sales_rate == 0:
                return 'Não_Vendida'
            elif sales_rate < 1.0:
                return 'Venda_Parcial'
            else:  # sales_rate >= 1.0
                return 'Esgotada'
        
        df_tratado['PERFORMANCE_CATEGORY'] = df_tratado['SALES_RATE'].apply(categorizar_performance)
    
    # ==== 3. Features de Preço ====
    print("\n## Criando features de preço")
    
    # Preço unitário
    if 'SOLD_AMOUNT' in df_tratado.columns and 'SOLD_QUANTITY' in df_tratado.columns:
        df_tratado['PRICE_PER_UNIT'] = df_tratado['SOLD_AMOUNT'] / df_tratado['SOLD_QUANTITY']
        df_tratado['PRICE_PER_UNIT'] = df_tratado['PRICE_PER_UNIT'].fillna(0)
    
    # Preço unitário por estoque comprometido
    if 'SOLD_AMOUNT' in df_tratado.columns and 'INVOLVED_STOCK' in df_tratado.columns:
        df_tratado['VALUE_PER_STOCK_UNIT'] = df_tratado['SOLD_AMOUNT'] / df_tratado['INVOLVED_STOCK']
        df_tratado['VALUE_PER_STOCK_UNIT'] = df_tratado['VALUE_PER_STOCK_UNIT'].fillna(0)
    
    # ==== 4. Features de Potencial ====
    print("\n## Criando features de potencial")
    
    if 'SOLD_AMOUNT' in df_tratado.columns and 'INVOLVED_STOCK' in df_tratado.columns and 'SOLD_QUANTITY' in df_tratado.columns:
        # Valor potencial se vendesse todo o estoque
        df_tratado['POTENTIAL_VALUE'] = df_tratado['PRICE_PER_UNIT'] * df_tratado['INVOLVED_STOCK']
        df_tratado['SALES_EFFICIENCY'] = df_tratado['SOLD_AMOUNT'] / df_tratado['POTENTIAL_VALUE']
        df_tratado['SALES_EFFICIENCY'] = df_tratado['SALES_EFFICIENCY'].fillna(0)
    
    # ==== 5. Features de Overselling ====
    print("\n## Criando features de overselling")
    
    # Margem de overselling
    if 'STOCK_SOLD' in df_tratado.columns and 'INVOLVED_STOCK' in df_tratado.columns:
        df_tratado['OVERSELLING_MARGIN'] = df_tratado['STOCK_SOLD'] - df_tratado['INVOLVED_STOCK']
        df_tratado['OVERSELLING_MARGIN'] = df_tratado['OVERSELLING_MARGIN'].clip(lower=0)  # Apenas valores positivos
    
    # ==== 6. KPIs de Qualidade e Consistência ====
    print("\n## Criando KPIs de qualidade e consistência")
    
    # Coerência entre SOLD_QUANTITY e STOCK_SOLD
    if {'INVOLVED_STOCK','REMAINING_STOCK_AFTER_END','SOLD_QUANTITY'}.issubset(df_tratado.columns):
        df_tratado['STOCK_SOLD_calc'] = df_tratado['INVOLVED_STOCK'] - df_tratado['REMAINING_STOCK_AFTER_END']
        # Divergência absoluta em unidades
        df_tratado['SOLD_UNITS_MISMATCH'] = (df_tratado['STOCK_SOLD_calc'] - df_tratado['SOLD_QUANTITY']).abs()
    
    # KPIs principais
    df_tratado['SELL_THROUGH'] = np.where(df_tratado['INVOLVED_STOCK'] > 0,
                                          df_tratado['STOCK_SOLD'] / df_tratado['INVOLVED_STOCK'], np.nan)
    
    df_tratado['STOCKOUT_FLAG'] = (df_tratado['REMAINING_STOCK_AFTER_END'] <= 0).astype(int)
    df_tratado['OVERSELL_FLAG'] = (df_tratado['REMAINING_STOCK_AFTER_END'] < 0).astype(int)
    df_tratado['OVERSELL_UNITS'] = np.where(df_tratado['REMAINING_STOCK_AFTER_END'] < 0,
                                            -df_tratado['REMAINING_STOCK_AFTER_END'], 0)
    
    # Métricas econômicas robustas
    df_tratado['ASP'] = np.where(df_tratado['SOLD_QUANTITY'] > 0,
                                 df_tratado['SOLD_AMOUNT'] / df_tratado['SOLD_QUANTITY'], np.nan)
    
    df_tratado['GMV_PER_STOCK'] = np.where(df_tratado['INVOLVED_STOCK'] > 0,
                                           df_tratado['SOLD_AMOUNT'] / df_tratado['INVOLVED_STOCK'], np.nan)
    
    # Corrigir/retirar VALUE_CONVERSION_RATE (dimensionalmente inconsistente)
    if 'VALUE_CONVERSION_RATE' in df_tratado.columns:
        df_tratado.drop(columns=['VALUE_CONVERSION_RATE'], inplace=True)
        print("   - VALUE_CONVERSION_RATE removida (métrica dimensionalmente inconsistente)")
    
    print("\nFeature engineering completo!")
    print(f"   Total de features criadas: {len(df_tratado.columns) - len(df.columns)}")
    
    return df_tratado



def obter_colunas_por_tipo():
    """
    Retorna dicionário com listas de colunas organizadas por tipo e categoria.
    
    Returns:
        dict: Dicionário com listas de colunas categorizadas
        
    Exemplo de uso:
        >>> cols = obter_colunas_por_tipo()
        >>> print(cols['kpis_principais'])
        ['SALES_RATE', 'SELL_THROUGH', 'ASP', ...]
        
        >>> print(cols['valores_monetarios'])
        ['SOLD_AMOUNT', 'POTENTIAL_VALUE', ...]
    """
    colunas = {
        # Colunas originais do dataset
        'data': ['OFFER_START_DATE', 'OFFER_START_DTTM', 'OFFER_FINISH_DTTM'],
        'numericas_originais': ['INVOLVED_STOCK', 'REMAINING_STOCK_AFTER_END', 'SOLD_AMOUNT', 'SOLD_QUANTITY'],
        'categoricas': ['OFFER_TYPE', 'ORIGIN', 'SHIPPING_PAYMENT_TYPE', 'DOM_DOMAIN_AGG1', 'VERTICAL', 'DOMAIN_ID'],
        
        # Features temporais criadas
        'temporais': ['DURATION_HOURS', 'START_HOUR', 'WEEKDAY', 'WEEKEND', 'DURATION_BUCKET', 'DAYPART'],
        
        # Features de vendas
        'vendas': ['STOCK_SOLD', 'SALES_RATE', 'PERFORMANCE_CATEGORY'],
        
        # Features de preço
        'precos': ['PRICE_PER_UNIT', 'VALUE_PER_STOCK_UNIT', 'ASP'],
        
        # Features de potencial
        'potencial': ['POTENTIAL_VALUE', 'SALES_EFFICIENCY'],
        
        # Features de overselling
        'overselling': ['OVERSELLING_MARGIN', 'OVERSELL_FLAG', 'OVERSELL_UNITS', 'STOCKOUT_FLAG'],
        
        # Features de qualidade
        'qualidade': ['STOCK_SOLD_calc', 'SOLD_UNITS_MISMATCH', 'SELL_THROUGH', 'GMV_PER_STOCK'],
    }
    
    # Consolidações úteis
    colunas['todas_numericas'] = (
        colunas['numericas_originais'] + 
        colunas['temporais'][:4] +  # Excluir DURATION_BUCKET e DAYPART (categóricas)
        colunas['vendas'][:2] +  # Excluir PERFORMANCE_CATEGORY (categórica)
        colunas['precos'] + 
        colunas['potencial'] + 
        colunas['overselling'] +  # Incluir todas: OVERSELLING_MARGIN, FLAGS, OVERSELL_UNITS
        colunas['qualidade']  # Incluir todas de qualidade
    )
    
    # Numéricas para análises (sem flags e duplicatas)
    colunas['numericas_analises'] = (
        colunas['numericas_originais'] + 
        colunas['temporais'][:4] +  # Excluir DURATION_BUCKET e DAYPART (categóricas)
        colunas['vendas'][:2] +  # Excluir PERFORMANCE_CATEGORY (categórica)
        colunas['precos'] + 
        colunas['potencial'] + 
        colunas['overselling'][:1] + colunas['overselling'][2:3] +  # OVERSELLING_MARGIN e OVERSELL_UNITS (sem flags)
        colunas['qualidade'][1:]  # Excluir STOCK_SOLD_calc (duplicata)
    )
    
    colunas['todas_categoricas'] = (
        colunas['categoricas'] + 
        ['DURATION_BUCKET', 'DAYPART', 'PERFORMANCE_CATEGORY']
    )
    
    colunas['flags'] = ['WEEKEND', 'STOCKOUT_FLAG', 'OVERSELL_FLAG']
    
    colunas['kpis_principais'] = [
        'SALES_RATE', 'SELL_THROUGH', 'ASP', 'GMV_PER_STOCK', 
        'STOCKOUT_FLAG', 'OVERSELL_FLAG', 'SALES_EFFICIENCY'
    ]
    
    # Para análises de quantidade e monetário
    colunas['quantidades_estoques'] = [
        'INVOLVED_STOCK', 'REMAINING_STOCK_AFTER_END', 'STOCK_SOLD', 
        'SOLD_QUANTITY', 'OVERSELLING_MARGIN', 'OVERSELL_UNITS'
    ]
    
    colunas['valores_monetarios'] = [
        'SOLD_AMOUNT', 'POTENTIAL_VALUE', 'PRICE_PER_UNIT', 
        'VALUE_PER_STOCK_UNIT', 'ASP', 'GMV_PER_STOCK'
    ]
    
    colunas['taxas_percentuais'] = [
        'SALES_RATE', 'SALES_EFFICIENCY', 'SELL_THROUGH'
    ]
    
    # Grupos para visualização/distribuições
    colunas['viz_quantidades'] = colunas['quantidades_estoques'] + ['SOLD_AMOUNT', 'POTENTIAL_VALUE']
    colunas['viz_taxas'] = colunas['taxas_percentuais'] + ['DURATION_HOURS', 'SELL_THROUGH']
    colunas['viz_precos'] = colunas['precos'] + ['GMV_PER_STOCK', 'VALUE_PER_STOCK_UNIT']
    
    # Grupos para boxplots (escalas similares, sem flags)
    colunas['box_quantidades'] = ['INVOLVED_STOCK', 'REMAINING_STOCK_AFTER_END', 'STOCK_SOLD', 'SOLD_QUANTITY']
    colunas['box_valores'] = ['SOLD_AMOUNT', 'POTENTIAL_VALUE', 'ASP', 'GMV_PER_STOCK']
    colunas['box_taxas'] = colunas['taxas_percentuais'] + ['SELL_THROUGH']
    
    return colunas



def detectar_outliers(df: pd.DataFrame, coluna: str) -> tuple[int, pd.DataFrame]:
    """Detecta outliers usando método IQR"""
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return len(outliers), outliers


def analise_outliers(df_tratado, colunas_numericas):
    """Análise de outliers"""
    print("=== OUTLIERS ===")
    out_lista = []
    out_keys = ['Variável', 'Quantidade', 'Porcentagem (%)']
    for col in colunas_numericas:
        
       num_outliers, outliers = detectar_outliers(df_tratado, col)
       out_lista.append([col, num_outliers, num_outliers/len(df_tratado)*100])

    df_ = pd.DataFrame(out_lista, columns=out_keys)
    print(tabulate(df_, headers='keys', tablefmt='grid', 
                    floatfmt='.2f', showindex=False))



def analise_distribuicoes_graficos(
    df: pd.DataFrame,
    col_num: list[str],
    target_col: str | None = None,
    plot_type: str = "hist",
    log_scale: bool = False,
    log_transform: str = "log1p",
) -> None:
    """
    Gera os gráficos de distribuição para colunas numéricas.

    Parâmetros:
    - df: DataFrame
    - col_num: lista de colunas numéricas
    - target_col: nome da coluna target (opcional, para segmentação)
    - plot_type: 'hist', 'kde' ou 'both'
    - log_scale: se True, aplica escala logarítmica nos eixos
    - log_transform: 'log', 'log1p', 'sqrt' - tipo de transformação
    """
    if target_col:
        print("GRÁFICOS - DISTRIBUIÇÃO DAS VARIÁVEIS NUMÉRICAS POR TARGET")
    else:
        print("GRÁFICOS - DISTRIBUIÇÃO DAS VARIÁVEIS NUMÉRICAS")
    
    if log_scale:
        print(f"(Escala Logarítmica - Transformação: {log_transform.upper()})")
    
    print("=" * 60)
    print()

    n_vars = len(col_num)

    if target_col:
        # 2 plots por variável (original + por target)
        n_cols = 2
        n_rows = n_vars
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        # Garantir que axes seja sempre 2D para facilitar o acesso
        if n_vars == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(col_num):
            # Preparar dados com transformação se necessário
            if log_scale:
                if log_transform == "log1p":
                    data_valid = df[df[col] >= 0][col]  # Aceita zeros
                    data_transformed = np.log1p(data_valid)
                elif log_transform == "sqrt":
                    data_valid = df[df[col] >= 0][col]
                    data_transformed = np.sqrt(data_valid)
                else:  # log normal
                    data_valid = df[df[col] > 0][col]  # Só positivos
                    data_transformed = np.log(data_valid)
                
                if len(data_transformed) == 0:
                    print(f"Aviso: {col} não tem valores válidos para transformação {log_transform}")
                    continue
            else:
                data_transformed = df[col]
            
            # Plot 1: Distribuição geral
            ax1 = axes[i, 0]
            if plot_type == "both":
                sns.histplot(
                    data_transformed,
                    ax=ax1,
                    color=plt.cm.Set3(i / len(col_num)),
                    kde=True,
                    stat="count",
                )
            elif plot_type == "hist":
                sns.histplot(data_transformed, ax=ax1, color=plt.cm.Set3(i / len(col_num)))
            elif plot_type == "kde":
                sns.kdeplot(data_transformed, ax=ax1, color=plt.cm.Set3(i / len(col_num)))

            ax1.set_title(f"{col.upper()} - Distribuição Geral", fontweight="bold")
            ax1.set_ylabel("Distribuição" if plot_type == "kde" else "Quantidade")
            
            if log_scale:
                ax1.set_xlabel(f"{col} ({log_transform.upper()})")

            # Plot 2: Distribuição por target
            ax2 = axes[i, 1]
            target_values = df[target_col].unique()

            if plot_type in ["hist", "both"]:
                for j, target_val in enumerate(target_values):
                    if log_scale:
                        if log_transform == "log1p":
                            data_subset = df[(df[target_col] == target_val) & (df[col] >= 0)][col]
                            data_subset_transformed = np.log1p(data_subset)
                        elif log_transform == "sqrt":
                            data_subset = df[(df[target_col] == target_val) & (df[col] >= 0)][col]
                            data_subset_transformed = np.sqrt(data_subset)
                        else:  # log normal
                            data_subset = df[(df[target_col] == target_val) & (df[col] > 0)][col]
                            data_subset_transformed = np.log(data_subset)
                    else:
                        data_subset_transformed = df[df[target_col] == target_val][col]
                    
                    if len(data_subset_transformed) > 0:
                        sns.histplot(
                            data_subset_transformed,
                            ax=ax2,
                            alpha=0.7,
                            label=f"{target_col}={target_val}",
                            color=plt.cm.Set2(j),
                        )

            if plot_type in ["kde", "both"]:
                for j, target_val in enumerate(target_values):
                    if log_scale:
                        if log_transform == "log1p":
                            data_subset = df[(df[target_col] == target_val) & (df[col] >= 0)][col]
                            data_subset_transformed = np.log1p(data_subset)
                        elif log_transform == "sqrt":
                            data_subset = df[(df[target_col] == target_val) & (df[col] >= 0)][col]
                            data_subset_transformed = np.sqrt(data_subset)
                        else:  # log normal
                            data_subset = df[(df[target_col] == target_val) & (df[col] > 0)][col]
                            data_subset_transformed = np.log(data_subset)
                    else:
                        data_subset_transformed = df[df[target_col] == target_val][col]
                    
                    if len(data_subset_transformed) > 0:
                        sns.kdeplot(
                            data_subset_transformed,
                            ax=ax2,
                            alpha=0.7,
                            label=f"{target_col}={target_val}",
                            color=plt.cm.Set2(j),
                        )

            ax2.set_title(
                f"{col.upper()} - Por {target_col.upper()}", fontweight="bold"
            )
            ax2.set_ylabel("Distribuição" if plot_type == "kde" else "Quantidade")
            ax2.legend()
            
            if log_scale:
                ax2.set_xlabel(f"{col} ({log_transform.upper()})")

    else:
        # Versão sem comparação da target
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        if n_vars == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(col_num):
            if i >= len(axes):
                break

            ax = axes[i]
            
            # Preparar dados com transformação se necessário
            if log_scale:
                if log_transform == "log1p":
                    data_valid = df[df[col] >= 0][col]  # Aceita zeros
                    data_transformed = np.log1p(data_valid)
                elif log_transform == "sqrt":
                    data_valid = df[df[col] >= 0][col]
                    data_transformed = np.sqrt(data_valid)
                else:  # log normal
                    data_valid = df[df[col] > 0][col]  # Só positivos
                    data_transformed = np.log(data_valid)
                
                if len(data_transformed) == 0:
                    print(f"Aviso: {col} não tem valores válidos para transformação {log_transform}")
                    continue
            else:
                data_transformed = df[col]

            if plot_type == "both":
                sns.histplot(
                    data_transformed,
                    ax=ax,
                    color=plt.cm.Set3(i / len(col_num)),
                    kde=True,
                    stat="count",
                )
            elif plot_type == "hist":
                sns.histplot(data_transformed, ax=ax, color=plt.cm.Set3(i / len(col_num)))
            elif plot_type == "kde":
                sns.kdeplot(data_transformed, ax=ax, color=plt.cm.Set3(i / len(col_num)))

            ax.set_title(f"{col.upper()}", fontweight="bold")
            ax.set_ylabel("Distribuição" if plot_type == "kde" else "Quantidade")
            
            if log_scale:
                ax.set_xlabel(f"{col} ({log_transform.upper()})")

        for j in range(len(col_num), len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def analise_boxplots_graficos(df: pd.DataFrame, col_num: list[str]) -> None:
    """
    Gera os gráficos de boxplot para colunas numéricas.
        Parâmetros:
    - df: DataFrame
    - col_num: lista de colunas numéricas
    """
    print("GRÁFICOS - BOXPLOT DAS VARIÁVEIS NUMÉRICAS")
    print("=" * 60)

    n_vars = len(col_num)
    n_cols = n_vars
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, col in enumerate(col_num):
        if i >= len(axes):
            break

        ax = axes[i]

        # Gráfico de boxplot
        sns.boxplot(y=df[col], ax=ax, color=plt.cm.Set3(i / len(col_num)))
        ax.set_title(f"{col.upper()}", fontweight="bold")
        ax.set_ylabel("Valor")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def grafico_correlacao_numericas(df: pd.DataFrame, col_num: list[str]) -> None:
    """
    Gera o gráfico de matrix de correlação das variáveis numéricas
    Parâmetros:
    - df: DataFrame
    - col_num: lista de colunas numéricas
    """
    print("GRÁFICO - MATRIX DE CORRELAÇÃO DAS VARIÁVEIS NUMÉRICAS")
    print("=" * 60)

    correlacao = df[col_num].corr()
    plt.figure(figsize=(15, 8))
    matrix = np.triu(correlacao)
    sns.heatmap(
        correlacao, annot=True, fmt=".1f", center=0, cmap="coolwarm", mask=matrix
    )
    plt.tight_layout()
    plt.show()


def identificar_pares_correlacionados(df: pd.DataFrame, col_num: list[str], threshold: float = 0.7) -> pd.DataFrame:
    """
    Identifica pares de variáveis altamente correlacionadas e indica se são originais ou criadas.
    
    Args:
        df: DataFrame com os dados
        col_num: Lista de colunas numéricas para analisar
        threshold: Limiar de correlação (padrão 0.7)
        
    Returns:
        DataFrame com pares correlacionados e suas características
    """
    # Obter definições de colunas
    cols = obter_colunas_por_tipo()
    colunas_originais = cols['data'] + cols['numericas_originais'] + cols['categoricas']
    
    # Calcular matriz de correlação
    correlacao = df[col_num].corr()
    
    # Extrair pares únicos (triângulo superior, excluindo diagonal)
    pares = []
    for i in range(len(correlacao.columns)):
        for j in range(i+1, len(correlacao.columns)):
            var1 = correlacao.columns[i]
            var2 = correlacao.columns[j]
            corr_val = correlacao.iloc[i, j]
            
            # Filtrar por threshold (valor absoluto)
            if abs(corr_val) >= threshold:
                # Classificar as variáveis
                tipo_var1 = 'Original' if var1 in colunas_originais else 'Criada'
                tipo_var2 = 'Original' if var2 in colunas_originais else 'Criada'
                
                # Identificar categoria da feature criada
                categoria_var1 = _identificar_categoria_feature(var1, cols)
                categoria_var2 = _identificar_categoria_feature(var2, cols)
                
                pares.append({
                    'Variavel_1': var1,
                    'Tipo_1': tipo_var1,
                    'Categoria_1': categoria_var1,
                    'Variavel_2': var2,
                    'Tipo_2': tipo_var2,
                    'Categoria_2': categoria_var2,
                    'Correlacao': corr_val,
                    'Correlacao_Abs': abs(corr_val),
                    'Tipo_Relacao': f"{tipo_var1} × {tipo_var2}"
                })
    
    # Criar DataFrame e ordenar por correlação absoluta
    df_pares = pd.DataFrame(pares)
    if len(df_pares) > 0:
        df_pares = df_pares.sort_values('Correlacao_Abs', ascending=False)
    
    return df_pares


def _identificar_categoria_feature(coluna: str, cols: dict) -> str:
    """
    Identifica a categoria de uma feature (helper interno).
    
    Args:
        coluna: Nome da coluna
        cols: Dicionário de colunas por tipo
        
    Returns:
        Categoria da feature ou 'Original' se não for feature criada
    """
    categorias = {
        'temporais': 'Temporal',
        'vendas': 'Vendas',
        'precos': 'Preço',
        'potencial': 'Potencial',
        'overselling': 'Overselling',
        'qualidade': 'Qualidade'
    }
    
    for cat_key, cat_label in categorias.items():
        if coluna in cols.get(cat_key, []):
            return cat_label
    
    # Se não encontrou, verificar se é original
    if coluna in cols.get('numericas_originais', []):
        return 'Original'
    
    return 'Outra'


def exibir_pares_correlacionados(df: pd.DataFrame, col_num: list[str], threshold: float = 0.7):
    """
    Exibe relatório formatado de pares correlacionados.
    
    Args:
        df: DataFrame com os dados
        col_num: Lista de colunas numéricas
        threshold: Limiar de correlação (padrão 0.7)
    """
    print("\n" + "="*80)
    print(f"PARES DE VARIÁVEIS ALTAMENTE CORRELACIONADAS (|r| >= {threshold})")
    print("="*80)
    
    df_pares = identificar_pares_correlacionados(df, col_num, threshold)
    
    if len(df_pares) == 0:
        print(f"\nNenhum par com correlação |r| >= {threshold} encontrado.")
        return df_pares
    
    print(f"\nTotal de pares encontrados: {len(df_pares)}")
    
    # Agrupar por tipo de relação
    print("\n" + "-"*80)
    print("POR TIPO DE RELAÇÃO:")
    print("-"*80)
    for tipo_rel in df_pares['Tipo_Relacao'].unique():
        count = (df_pares['Tipo_Relacao'] == tipo_rel).sum()
        print(f"   - {tipo_rel}: {count} pares")
    
    # Top 10 correlações
    print("\n" + "-"*80)
    print("TOP 10 CORRELAÇÕES MAIS FORTES:")
    print("-"*80)
    
    for idx, row in df_pares.head(10).iterrows():
        sinal = "+" if row['Correlacao'] > 0 else "-"
        print(f"\n{idx+1}. {row['Variavel_1']} x {row['Variavel_2']}")
        print(f"   Correlação: {sinal}{row['Correlacao_Abs']:.3f}")
        print(f"   {row['Categoria_1']} ({row['Tipo_1']}) x {row['Categoria_2']} ({row['Tipo_2']})")
    
    # Análise por categoria
    print("\n" + "-"*80)
    print("ANÁLISE POR CATEGORIA:")
    print("-"*80)
    
    # Correlações entre originais
    originais = df_pares[df_pares['Tipo_Relacao'] == 'Original × Original']
    if len(originais) > 0:
        print(f"\n[ORIGINAIS x ORIGINAIS] ({len(originais)} pares):")
        print("   ATENÇÃO: Alta correlação entre variáveis originais pode indicar multicolinearidade")
        for _, row in originais.iterrows():
            print(f"   - {row['Variavel_1']} x {row['Variavel_2']}: {row['Correlacao']:.3f}")
    
    # Correlações entre criadas
    criadas = df_pares[df_pares['Tipo_Relacao'] == 'Criada × Criada']
    if len(criadas) > 0:
        print(f"\n[CRIADAS x CRIADAS] ({len(criadas)} pares):")
        print("   INFO: Esperado, pois features derivadas compartilham variáveis base")
        # Mostrar apenas top 3
        for _, row in criadas.head(3).iterrows():
            print(f"   - {row['Variavel_1']} x {row['Variavel_2']}: {row['Correlacao']:.3f}")
        if len(criadas) > 3:
            print(f"   ... e mais {len(criadas)-3} pares")
    
    # Correlações mistas
    mistas = df_pares[df_pares['Tipo_Relacao'] == 'Original × Criada']
    if len(mistas) > 0:
        print(f"\n[ORIGINAIS x CRIADAS] ({len(mistas)} pares):")
        print("   INFO: Features derivadas correlacionadas com suas variáveis base")
        # Mostrar apenas top 3
        for _, row in mistas.head(3).iterrows():
            print(f"   - {row['Variavel_1']} x {row['Variavel_2']}: {row['Correlacao']:.3f}")
        if len(mistas) > 3:
            print(f"   ... e mais {len(mistas)-3} pares")
    
    print("\n" + "="*80)
    
    return df_pares


def graficos_distribuicao_categoricas(
    df: pd.DataFrame, col_cat: list[str]
) -> None:
    """
    Gera os gráficos de distribuição para colunas categóricas.

    Parâmetros:
    - df: DataFrame
    - col_cat: lista de colunas categóricas
    - target_col: nome da coluna target (opcional, para segmentação)
    """

    print("GRÁFICOS - DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS")
    print("=" * 60)

    n_vars = len(col_cat)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, col in enumerate(col_cat):
        if i >= len(axes):
            break

        ax = axes[i]

        sns.countplot(
            data=df,
            x=col,
            ax=ax,
            hue=col,
            legend=False,
            palette="Set3",
            order=df[col].value_counts().index,
        )

        ax.set_title(f"{col.upper()}", fontweight="bold")
        ax.set_ylabel("Quantidade")
        ax.tick_params(axis="x", rotation=45)

    # Esconder axes vazios
    for j in range(len(col_cat), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()



def analisar_categoria_performance(df, categoria, top_n=10):
    """
    Analisa uma categoria específica de performance das ofertas
    
    Parâmetros:
    - df: DataFrame com os dados
    - categoria: 'Não_Vendida', 'Venda_Parcial', 'Esgotada'
    - top_n: número de top resultados para mostrar
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Filtrar dados da categoria
    dados_categoria = df[df['PERFORMANCE_CATEGORY'] == categoria]
    
    print("="*80)
    print(f"ANÁLISE DE OFERTAS: {categoria.upper()}")
    print("="*80)
    
    # Estatísticas básicas
    total_categoria = len(dados_categoria)
    total_geral = len(df)
    percentual = total_categoria / total_geral * 100
    
    print(f"Total de ofertas {categoria.lower()}: {total_categoria:,}")
    print(f"Percentual do total: {percentual:.1f}%")
    
    if total_categoria > 0:
        # Estatísticas específicas
        if categoria == 'Não_Vendida':
            print(f"Todas as ofertas têm SALES_RATE = 0")
        elif categoria == 'Venda_Parcial':
            taxa_media = dados_categoria['SALES_RATE'].mean()
            print(f"Taxa média de venda: {taxa_media:.2%}")
        elif categoria == 'Esgotada':
            taxa_media = dados_categoria['SALES_RATE'].mean()
            print(f"Taxa média de venda: {taxa_media:.2%}")
            overselling = (dados_categoria['SALES_RATE'] > 1.0).sum()
            print(f"Ofertas com overselling: {overselling:,} ({overselling/total_categoria*100:.1f}%)")
        
        # Top ofertas da categoria
        print(f"\nTop {top_n} ofertas {categoria.lower()}:")
        top_ofertas = dados_categoria.nlargest(top_n, 'SALES_RATE')[
            ['VERTICAL', 'DOM_DOMAIN_AGG1', 'INVOLVED_STOCK', 'STOCK_SOLD', 'SALES_RATE', 'SOLD_AMOUNT']
        ]
        print(top_ofertas.to_string(index=False))
        
        # Gráficos
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Por Vertical
        plt.subplot(2, 2, 1)
        vertical_counts = dados_categoria['VERTICAL'].value_counts()
        colors_vertical = [plt.cm.Set3(i/len(vertical_counts)) for i in range(len(vertical_counts))]
        vertical_counts.plot(kind='bar', color=colors_vertical, alpha=0.8)
        plt.title(f'{categoria} por Categoria')
        plt.xlabel('Categoria')
        plt.ylabel('Número de Ofertas')
        plt.xticks(rotation=45)
        
        # Gráfico 2: Por Domínio (Top 10)
        plt.subplot(2, 2, 2)
        dominio_counts = dados_categoria['DOM_DOMAIN_AGG1'].value_counts().head(10)
        colors_dominio = [plt.cm.Set3(i/10) for i in range(len(dominio_counts))]
        dominio_counts.plot(kind='bar', color=colors_dominio, alpha=0.8)
        plt.title(f'{categoria} por Domínio - TOP 10')
        plt.xlabel('Domínio Agregado')
        plt.ylabel('Número de Ofertas')
        plt.xticks(rotation=45)
        
        # Gráfico 3: Distribuição de SALES_RATE
        plt.subplot(2, 2, 3)
        if categoria != 'Não_Vendida':
            plt.hist(dados_categoria['SALES_RATE'], bins=30, alpha=0.8, color=plt.cm.Set3(0.3))
            plt.title(f'Distribuição da Taxa de Venda - {categoria}')
            plt.xlabel('Taxa de Venda')
            plt.ylabel('Frequência')
        else:
            plt.text(0.5, 0.5, 'Todas as ofertas têm\nSALES_RATE = 0', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=plt.cm.Set3(0.1)))
            plt.title(f'Distribuição da Taxa de Venda - {categoria}')
        
        # Gráfico 4: Por Período (se disponível)
        plt.subplot(2, 2, 4)
        if 'OFFER_START_DATE' in dados_categoria.columns:
            periodo_counts = dados_categoria['OFFER_START_DATE'].dt.date.value_counts().sort_index()
            periodo_counts.plot(kind='line', marker='o', color=plt.cm.Set3(0.5), alpha=0.8, linewidth=2)
            plt.title(f'{categoria} por Dia')
            plt.xlabel('Data')
            plt.ylabel('Número de Ofertas')
            plt.xticks(rotation=45)
        else:
            # Se não tiver data, mostrar distribuição por SHIPPING_PAYMENT_TYPE
            shipping_counts = dados_categoria['SHIPPING_PAYMENT_TYPE'].value_counts()
            colors_shipping = [plt.cm.Set3(i/len(shipping_counts)) for i in range(len(shipping_counts))]
            shipping_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors_shipping)
            plt.title(f'{categoria} por Tipo de Frete')
        
        plt.tight_layout()
        plt.show()
        
        # Resumo por vertical
        print(f"\nResumo por Vertical:")
        vertical_summary = dados_categoria.groupby('VERTICAL').agg({
            'SALES_RATE': ['count', 'mean'],
            'SOLD_AMOUNT': 'sum'
        }).round(2)
        vertical_summary.columns = ['Quantidade', 'Taxa_Media', 'Receita_Total']
        print(vertical_summary.sort_values('Quantidade', ascending=False))
        
    else:
        print(f"Nenhuma oferta encontrada na categoria '{categoria}'")
    
    return dados_categoria

def analise_por_vertical_ultra_simples(df):
    """
    Versão ultra simples usando apenas PERFORMANCE_CATEGORY
    """
    # Contagem por vertical e categoria
    contagem = pd.crosstab(df['VERTICAL'], df['PERFORMANCE_CATEGORY'])
    
    # Adicionar totais e percentuais
    contagem['Total'] = contagem.sum(axis=1)
    
    # Taxa média por vertical
    taxa_media = df.groupby('VERTICAL')['SALES_RATE'].mean()
    contagem['Taxa_Media'] = taxa_media
    
    # Percentuais
    for col in contagem.columns[:-2]:  # Excluir Total e Taxa_Media
        contagem[f'%_{col}'] = (contagem[col] / contagem['Total'] * 100).round(1)
    
    return contagem

def analise_completa_por_vertical(df):
    """
    Análise completa por vertical com dados e gráficos
    """
    # 1. Análise de dados
    vertical_analysis = analise_por_vertical_ultra_simples(df)
    
    # 2. Gráficos
    plt.figure(figsize=(16, 12))
    
    # Heatmap
    plt.subplot(2, 2, 1)
    heatmap_data = vertical_analysis[['%_Não_Vendida', '%_Venda_Parcial', '%_Esgotada']].T
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Percentual (%)'}, 
                linewidths=0.5, linecolor='white')
    plt.title('Distribuição por Vertical (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Vertical', fontsize=12)
    plt.ylabel('Categoria', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Top 5 Esgotadas
    plt.subplot(2, 2, 2)
    top5_esgotadas = vertical_analysis.nlargest(5, 'Esgotada')
    colors_esgot = [plt.cm.Set3(i/5) for i in range(len(top5_esgotadas))]
    bars = plt.barh(range(len(top5_esgotadas)), top5_esgotadas['Esgotada'], 
                    color=colors_esgot, alpha=0.8, edgecolor='darkgray', linewidth=1)
    plt.yticks(range(len(top5_esgotadas)), top5_esgotadas.index)
    plt.title('Top 5 - Ofertas Esgotadas', fontsize=14, fontweight='bold')
    plt.xlabel('Quantidade', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Top 5 Taxa Média
    plt.subplot(2, 2, 3)
    top5_taxa = vertical_analysis.nlargest(5, 'Taxa_Media')
    colors_taxa = [plt.cm.Set3(i/5 + 0.5) for i in range(len(top5_taxa))]
    bars = plt.barh(range(len(top5_taxa)), top5_taxa['Taxa_Media'], 
                    color=colors_taxa, alpha=0.8, edgecolor='darkgray', linewidth=1)
    plt.yticks(range(len(top5_taxa)), top5_taxa.index)
    plt.title('Top 5 - Taxa Média de Venda', fontsize=14, fontweight='bold')
    plt.xlabel('Taxa Média', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Comparativo
    plt.subplot(2, 2, 4)
    top5_geral = vertical_analysis.nlargest(5, 'Total')
    x = np.arange(len(top5_geral))
    width = 0.25
    
    bars1 = plt.bar(x - width, top5_geral['%_Não_Vendida'], width, 
                    label='Não Vendidas', color=plt.cm.Set3(0.2), alpha=0.8, edgecolor='darkgray')
    bars2 = plt.bar(x, top5_geral['%_Venda_Parcial'], width, 
                    label='Parciais', color=plt.cm.Set3(0.5), alpha=0.8, edgecolor='darkgray')
    bars3 = plt.bar(x + width, top5_geral['%_Esgotada'], width, 
                    label='Esgotadas', color=plt.cm.Set3(0.8), alpha=0.8, edgecolor='darkgray')
    
    plt.xlabel('Vertical', fontsize=12)
    plt.ylabel('Percentual (%)', fontsize=12)
    plt.title('Top 5 Verticais - Comparativo', fontsize=14, fontweight='bold')
    plt.xticks(x, top5_geral.index, rotation=45, ha='right')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Resumo estatístico
    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO POR VERTICAL")
    print("="*80)
    
    melhor_vertical = vertical_analysis.loc[vertical_analysis['Taxa_Media'].idxmax()]
    pior_vertical = vertical_analysis.loc[vertical_analysis['Taxa_Media'].idxmin()]
    
    print(f"\nMELHOR PERFORMANCE:")
    print(f"   {vertical_analysis['Taxa_Media'].idxmax()}")
    print(f"   Taxa média: {melhor_vertical['Taxa_Media']:.1%}")
    print(f"   Esgotadas: {melhor_vertical['Esgotada']:,} ({melhor_vertical['%_Esgotada']:.1f}%)")
    
    print(f"\nPIOR PERFORMANCE:")
    print(f"   {vertical_analysis['Taxa_Media'].idxmin()}")
    print(f"   Taxa média: {pior_vertical['Taxa_Media']:.1%}")
    print(f"   Não vendidas: {pior_vertical['Não_Vendida']:,} ({pior_vertical['%_Não_Vendida']:.1f}%)")
    
    return vertical_analysis



def analise_temporal_por_vertical(df):
    """
    Análise temporal das vendas por vertical e categoria de performance
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    print("="*80)
    print("ANÁLISE TEMPORAL POR VERTICAL E CATEGORIA")
    print("="*80)
    
    # 1. Análise por dia
    df['DIA'] = df['OFFER_START_DATE'].dt.date
    df['DIA_SEMANA'] = df['OFFER_START_DATE'].dt.day_name()
    df['MES'] = df['OFFER_START_DATE'].dt.month
    
    # Estatísticas temporais
    print("\n ESTATÍSTICAS TEMPORAIS:")
    print(f"Período analisado: {df['OFFER_START_DATE'].min()} a {df['OFFER_START_DATE'].max()}")
    print(f"Total de dias: {df['DIA'].nunique()}")
    print(f"Dias da semana: {df['DIA_SEMANA'].value_counts().to_dict()}")
    
    # 2. Análise por dia da semana
    print("\n PERFORMANCE POR DIA DA SEMANA:")
    dia_semana_analysis = df.groupby(['DIA_SEMANA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    dia_semana_analysis['Total'] = dia_semana_analysis.sum(axis=1)
    
    # Calcular percentuais
    for col in dia_semana_analysis.columns[:-1]:
        dia_semana_analysis[f'%_{col}'] = (dia_semana_analysis[col] / dia_semana_analysis['Total'] * 100).round(1)
    
    print(dia_semana_analysis)
    
    # 3. Análise por vertical e tempo
    print("\n ANÁLISE POR VERTICAL E TEMPO:")
    vertical_temporal = df.groupby(['VERTICAL', 'PERFORMANCE_CATEGORY']).agg({
        'SALES_RATE': ['count', 'mean'],
        'SOLD_AMOUNT': 'sum',
        'DURATION_HOURS': 'mean'
    }).round(2)
    
    vertical_temporal.columns = ['Quantidade', 'Taxa_Media', 'Receita_Total', 'Duracao_Media']
    print(vertical_temporal)
    
    # 4. Gráficos temporais
    plt.figure(figsize=(20, 15))
    
    # Gráfico 1: Evolução temporal por categoria
    plt.subplot(3, 3, 1)
    evolucao_temporal = df.groupby(['DIA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_cat = [plt.cm.Set3(i/len(evolucao_temporal.columns)) for i in range(len(evolucao_temporal.columns))]
    evolucao_temporal.plot(kind='line', marker='o', color=colors_cat, ax=plt.gca())
    plt.title('Evolução Temporal por Categoria', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Número de Ofertas')
    plt.xticks(rotation=45)
    plt.legend(title='Categoria')
    plt.grid(alpha=0.3)
    
    # Gráfico 2: Performance por dia da semana
    plt.subplot(3, 3, 2)
    colors_perf = [plt.cm.Set3(i/3) for i in range(3)]
    dia_semana_analysis[['Não_Vendida', 'Venda_Parcial', 'Esgotada']].plot(kind='bar', color=colors_perf, ax=plt.gca())
    plt.title('Performance por Dia da Semana', fontsize=14, fontweight='bold')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Número de Ofertas')
    plt.xticks(rotation=45)
    plt.legend(title='Categoria')
    
    # Gráfico 3: Heatmap Vertical vs Dia da Semana
    plt.subplot(3, 3, 3)
    heatmap_data = df.groupby(['VERTICAL', 'DIA_SEMANA'])['PERFORMANCE_CATEGORY'].value_counts().unstack(fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=plt.gca())
    plt.title('Heatmap: Vertical vs Dia da Semana', fontsize=14, fontweight='bold')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Vertical')
    
    # Gráfico 4: Taxa de sucesso por dia da semana
    plt.subplot(3, 3, 4)
    taxa_sucesso_dia = df.groupby('DIA_SEMANA')['SALES_RATE'].agg(['mean', 'std']).round(3)
    colors_dias = [plt.cm.Set3(i/7) for i in range(len(taxa_sucesso_dia))]
    taxa_sucesso_dia['mean'].plot(kind='bar', color=colors_dias, alpha=0.8, ax=plt.gca())
    plt.title('Taxa Média de Venda por Dia', fontsize=14, fontweight='bold')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Taxa Média de Venda')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Gráfico 5: Duração vs Performance
    plt.subplot(3, 3, 5)
    df.boxplot(column='DURATION_HOURS', by='PERFORMANCE_CATEGORY', ax=plt.gca())
    plt.title('Duração vs Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Categoria de Performance')
    plt.ylabel('Duração (horas)')
    plt.suptitle('')  # Remove título automático
    
    # Gráfico 6: Receita por vertical e categoria
    plt.subplot(3, 3, 6)
    receita_vertical = df.groupby(['VERTICAL', 'PERFORMANCE_CATEGORY'])['SOLD_AMOUNT'].sum().unstack(fill_value=0)
    colors_receita = [plt.cm.Set3(i/len(receita_vertical.columns)) for i in range(len(receita_vertical.columns))]
    receita_vertical.plot(kind='bar', color=colors_receita, ax=plt.gca())
    plt.title('Receita por Vertical e Categoria', fontsize=14, fontweight='bold')
    plt.xlabel('Vertical')
    plt.ylabel('Receita Total')
    plt.xticks(rotation=45)
    plt.legend(title='Categoria')
    
    # Gráfico 7: Análise de tendência semanal
    plt.subplot(3, 3, 7)
    df['SEMANA'] = df['OFFER_START_DATE'].dt.isocalendar().week
    tendencia_semanal = df.groupby(['SEMANA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_tend = [plt.cm.Set3(i/len(tendencia_semanal.columns)) for i in range(len(tendencia_semanal.columns))]
    tendencia_semanal.plot(kind='line', marker='o', color=colors_tend, ax=plt.gca())
    plt.title('Tendência Semanal', fontsize=14, fontweight='bold')
    plt.xlabel('Semana do Ano')
    plt.ylabel('Número de Ofertas')
    plt.legend(title='Categoria')
    plt.grid(alpha=0.3)
    
    # Gráfico 8: Performance por mês
    plt.subplot(3, 3, 8)
    performance_mes = df.groupby(['MES', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_mes = [plt.cm.Set3(i/len(performance_mes.columns)) for i in range(len(performance_mes.columns))]
    performance_mes.plot(kind='bar', color=colors_mes, ax=plt.gca())
    plt.title('Performance por Mês', fontsize=14, fontweight='bold')
    plt.xlabel('Mês')
    plt.ylabel('Número de Ofertas')
    plt.xticks(rotation=0)
    plt.legend(title='Categoria')
    
    # Gráfico 9: Correlação temporal
    plt.subplot(3, 3, 9)
    # Criar variável dummy para categoria
    df_dummy = pd.get_dummies(df['PERFORMANCE_CATEGORY'])
    df_temporal = df[['DURATION_HOURS', 'SALES_RATE', 'SOLD_AMOUNT']].join(df_dummy)
    
    # Matriz de correlação
    correlacao = df_temporal.corr()
    sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=plt.gca())
    plt.title('Matriz de Correlação Temporal', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Insights temporais
    print("\n" + "="*80)
    print("INSIGHTS TEMPORAIS")
    print("="*80)
    
    # Melhor dia da semana
    melhor_dia = dia_semana_analysis['%_Esgotada'].idxmax()
    pior_dia = dia_semana_analysis['%_Não_Vendida'].idxmax()
    
    print(f"\n MELHOR DIA DA SEMANA (mais esgotadas):")
    print(f"   {melhor_dia}: {dia_semana_analysis.loc[melhor_dia, '%_Esgotada']:.1f}% esgotadas")
    
    print(f"\n PIOR DIA DA SEMANA (mais não vendidas):")
    print(f"   {pior_dia}: {dia_semana_analysis.loc[pior_dia, '%_Não_Vendida']:.1f}% não vendidas")
    
    # Melhor vertical por período
    melhor_vertical_temporal = df.groupby('VERTICAL')['SALES_RATE'].mean().idxmax()
    print(f"\n MELHOR VERTICAL TEMPORAL:")
    print(f"   {melhor_vertical_temporal}: {df.groupby('VERTICAL')['SALES_RATE'].mean().max():.1%} taxa média")
    
    # Análise de sazonalidade
    print(f"\n ANÁLISE DE SAZONALIDADE:")
    for mes in sorted(df['MES'].unique()):
        mes_data = df[df['MES'] == mes]
        taxa_media_mes = mes_data['SALES_RATE'].mean()
        esgotadas_mes = (mes_data['PERFORMANCE_CATEGORY'] == 'Esgotada').sum()
        print(f"   Mês {mes}: {taxa_media_mes:.1%} taxa média, {esgotadas_mes:,} esgotadas")
    
    return df



def analise_evolutiva_temporal(df):
    """
    Análise da evolução das vendas ao longo do tempo
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("="*60)
    print("ANALISE EVOLUTIVA TEMPORAL")
    print("="*60)
    
    # Preparar dados temporais
    df['DIA'] = df['OFFER_START_DATE'].dt.date
    df['DIA_SEMANA'] = df['OFFER_START_DATE'].dt.day_name()
    df['SEMANA'] = df['OFFER_START_DATE'].dt.isocalendar().week
    df['MES'] = df['OFFER_START_DATE'].dt.month
    
    # Estatísticas básicas
    periodo_inicio = df['OFFER_START_DATE'].min().strftime('%Y-%m-%d')
    periodo_fim = df['OFFER_START_DATE'].max().strftime('%Y-%m-%d')
    total_dias = df['DIA'].nunique()
    
    print(f"Periodo analisado: {periodo_inicio} a {periodo_fim}")
    print(f"Total de dias: {total_dias}")
    print(f"Total de semanas: {df['SEMANA'].nunique()}")
    
    # Gráficos evolutivos
    plt.figure(figsize=(18, 12))
    
    # Gráfico 1: Evolução diária por categoria
    plt.subplot(2, 3, 1)
    evolucao_diaria = df.groupby(['DIA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_evol = [plt.cm.Set3(i/3) for i in range(3)]
    evolucao_diaria[['Não_Vendida', 'Venda_Parcial', 'Esgotada']].plot(kind='line', marker='o', color=colors_evol, ax=plt.gca())
    plt.title('Evolucao Diaria por Categoria')
    plt.xlabel('Data')
    plt.ylabel('Numero de Ofertas')
    plt.xticks(rotation=45)
    plt.legend(title='Categoria')
    plt.grid(alpha=0.3)
    
    # Gráfico 2: Evolução semanal
    plt.subplot(2, 3, 2)
    evolucao_semanal = df.groupby(['SEMANA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_sem = [plt.cm.Set3(i/3) for i in range(3)]
    evolucao_semanal[['Não_Vendida', 'Venda_Parcial', 'Esgotada']].plot(kind='line', marker='s', color=colors_sem, ax=plt.gca())
    plt.title('Evolucao Semanal por Categoria')
    plt.xlabel('Semana do Ano')
    plt.ylabel('Numero de Ofertas')
    plt.legend(title='Categoria')
    plt.grid(alpha=0.3)
    
    # Gráfico 3: Taxa de sucesso ao longo do tempo
    plt.subplot(2, 3, 3)
    taxa_temporal = df.groupby('DIA')['SALES_RATE'].mean()
    taxa_temporal.plot(kind='line', marker='o', color=plt.cm.Set3(0.6), linewidth=2, markersize=4, ax=plt.gca())
    plt.title('Taxa de Sucesso ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Taxa Media de Venda')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Gráfico 4: Performance por dia da semana
    plt.subplot(2, 3, 4)
    dia_semana_analysis = df.groupby(['DIA_SEMANA', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    colors_dia = [plt.cm.Set3(i/3) for i in range(3)]
    dia_semana_analysis[['Não_Vendida', 'Venda_Parcial', 'Esgotada']].plot(kind='bar', color=colors_dia, ax=plt.gca())
    plt.title('Performance por Dia da Semana')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Numero de Ofertas')
    plt.xticks(rotation=45)
    plt.legend(title='Categoria')
    
    # Gráfico 5: Taxa média por dia da semana
    plt.subplot(2, 3, 5)
    taxa_dia_semana = df.groupby('DIA_SEMANA')['SALES_RATE'].mean().sort_values(ascending=False)
    colors_dia_semana = [plt.cm.Set3(i/len(taxa_dia_semana)) for i in range(len(taxa_dia_semana))]
    taxa_dia_semana.plot(kind='bar', color=colors_dia_semana, alpha=0.8, ax=plt.gca())
    plt.title('Taxa Media por Dia da Semana')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Taxa Media de Venda')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Gráfico 6: Evolução por vertical
    plt.subplot(2, 3, 6)
    evolucao_vertical = df.groupby(['DIA', 'VERTICAL'])['SALES_RATE'].mean().unstack(fill_value=0)
    colors_vert = [plt.cm.Set3(i/len(evolucao_vertical.columns)) for i in range(len(evolucao_vertical.columns))]
    evolucao_vertical.plot(kind='line', color=colors_vert, ax=plt.gca())
    plt.title('Evolucao da Taxa por Vertical')
    plt.xlabel('Data')
    plt.ylabel('Taxa Media de Venda')
    plt.xticks(rotation=45)
    plt.legend(title='Vertical',  loc='upper left') #bbox_to_anchor=(1.05, 1),
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Insights evolutivos
    print("\n" + "="*60)
    print("INSIGHTS EVOLUTIVOS")
    print("="*60)
    
    # Melhor e pior dia da semana
    melhor_dia = dia_semana_analysis['Esgotada'].idxmax()
    pior_dia = dia_semana_analysis['Não_Vendida'].idxmax()
    
    print(f"\nMelhor dia da semana (mais esgotadas):")
    print(f"   {melhor_dia}: {dia_semana_analysis.loc[melhor_dia, 'Esgotada']:,} esgotadas")
    
    print(f"\nPior dia da semana (mais nao vendidas):")
    print(f"   {pior_dia}: {dia_semana_analysis.loc[pior_dia, 'Não_Vendida']:,} nao vendidas")
    
    # Tendência temporal
    taxa_inicio = df[df['DIA'] == df['DIA'].min()]['SALES_RATE'].mean()
    taxa_fim = df[df['DIA'] == df['DIA'].max()]['SALES_RATE'].mean()
    tendencia = "melhorou" if taxa_fim > taxa_inicio else "piorou"
    
    print(f"\nTendencia temporal:")
    print(f"   Inicio do periodo: {taxa_inicio:.1%}")
    print(f"   Fim do periodo: {taxa_fim:.1%}")
    print(f"   Performance {tendencia} ao longo do tempo")
    
    return df


def analise_correlacoes_heatmaps(df):
    """
    Análise de correlações e heatmaps entre variáveis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    print("="*60)
    print("ANALISE DE CORRELACOES E HEATMAPS")
    print("="*60)
    
    # Preparar dados
    df['DIA_SEMANA'] = df['OFFER_START_DATE'].dt.day_name()
    df['MES'] = df['OFFER_START_DATE'].dt.month
    df['DIA'] = df['OFFER_START_DATE'].dt.day
    
    # Gráficos de correlação
    plt.figure(figsize=(18, 12))
    
    # Gráfico 1: Matriz de correlação numérica
    plt.subplot(2, 3, 1)
    colunas_numericas = ['SALES_RATE', 'SOLD_AMOUNT', 'DURATION_HOURS', 'INVOLVED_STOCK', 'STOCK_SOLD']
    correlacao_numerica = df[colunas_numericas].corr()
    sns.heatmap(correlacao_numerica, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=plt.gca())
    plt.title('Matriz de Correlacao Numerica')
    
    # Gráfico 2: Heatmap Vertical vs Dia da Semana
    plt.subplot(2, 3, 2)
    heatmap_vertical_dia = df.groupby(['VERTICAL', 'DIA_SEMANA'])['PERFORMANCE_CATEGORY'].value_counts().unstack(fill_value=0)
    sns.heatmap(heatmap_vertical_dia, annot=True, fmt='d', cmap='YlOrRd', ax=plt.gca())
    plt.title('Heatmap: Vertical vs Dia da Semana')
    plt.xlabel('Dia da Semana')
    plt.ylabel('Vertical')
    
    # Gráfico 3: Heatmap Performance por Mês
    plt.subplot(2, 3, 3)
    heatmap_mes = df.groupby(['MES', 'PERFORMANCE_CATEGORY']).size().unstack(fill_value=0)
    sns.heatmap(heatmap_mes, annot=True, fmt='d', cmap='RdYlGn', ax=plt.gca())
    plt.title('Heatmap: Performance por Mes')
    plt.xlabel('Categoria de Performance')
    plt.ylabel('Mes')
    
    # Gráfico 4: Correlação com variáveis dummy
    plt.subplot(2, 3, 4)
    df_dummy = pd.get_dummies(df['PERFORMANCE_CATEGORY'])
    df_correlacao = df[['SALES_RATE', 'SOLD_AMOUNT', 'DURATION_HOURS']].join(df_dummy)
    correlacao_dummy = df_correlacao.corr()
    sns.heatmap(correlacao_dummy, annot=True, fmt='.2f', cmap='viridis', ax=plt.gca())
    plt.title('Correlacao com Variaveis Dummy')
    
    # Gráfico 5: Heatmap por Vertical e Performance
    plt.subplot(2, 3, 5)
    heatmap_vertical_perf = df.groupby(['VERTICAL', 'PERFORMANCE_CATEGORY']).agg({
        'SALES_RATE': 'mean',
        'SOLD_AMOUNT': 'sum'
    }).round(2)
    sns.heatmap(heatmap_vertical_perf['SALES_RATE'].unstack(fill_value=0), 
                annot=True, fmt='.2f', cmap='Blues', ax=plt.gca())
    plt.title('Taxa Media por Vertical e Performance')
    plt.xlabel('Categoria de Performance')
    plt.ylabel('Vertical')
    
    # Gráfico 6: Correlação temporal (APENAS VARIÁVEIS NUMÉRICAS)
    plt.subplot(2, 3, 6)
    # Usar apenas variáveis numéricas para correlação
    correlacao_temporal = df[['DIA', 'MES', 'SALES_RATE', 'SOLD_AMOUNT', 'DURATION_HOURS']].corr()
    sns.heatmap(correlacao_temporal, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=plt.gca())
    plt.title('Correlacao Temporal (Variaveis Numericas)')
    
    plt.tight_layout()
    plt.show()
    
    # Insights de correlação
    print("\n" + "="*60)
    print("INSIGHTS DE CORRELACAO")
    print("="*60)
    
    # Correlações mais fortes
    correlacoes = df[colunas_numericas].corr()
    correlacoes_altas = []
    
    for i in range(len(correlacoes.columns)):
        for j in range(i+1, len(correlacoes.columns)):
            corr_val = correlacoes.iloc[i, j]
            if abs(corr_val) > 0.5:
                correlacoes_altas.append((correlacoes.columns[i], correlacoes.columns[j], corr_val))
    
    print(f"\nCorrelacoes mais fortes (|r| > 0.5):")
    for var1, var2, corr in sorted(correlacoes_altas, key=lambda x: abs(x[2]), reverse=True):
        print(f"   {var1} vs {var2}: {corr:.3f}")
    
    # Melhor combinação vertical + dia (CORRIGIDO)
    print(f"\nMelhor combinacao Vertical + Dia:")
    try:
        # Encontrar a combinação com mais ofertas
        combinacoes = df.groupby(['VERTICAL', 'DIA_SEMANA']).size().sort_values(ascending=False)
        melhor_vertical = combinacoes.index[0][0]
        melhor_dia = combinacoes.index[0][1]
        melhor_valor = combinacoes.iloc[0]
        
        print(f"   {melhor_vertical} + {melhor_dia}: {melhor_valor} ofertas")
        
        # Top 3 combinações
        print(f"\nTop 3 combinacoes Vertical + Dia:")
        for i in range(min(3, len(combinacoes))):
            vertical, dia = combinacoes.index[i]
            valor = combinacoes.iloc[i]
            print(f"   {i+1}. {vertical} + {dia}: {valor} ofertas")
            
    except Exception as e:
        print(f"   Erro ao calcular combinacoes: {e}")
    
    return df



def analise_data_quality(df):
    """
    Análise de qualidade dos dados: mismatch, oversell, etc.
    """
    print("\n" + "="*80)
    print("ANÁLISE DE QUALIDADE DOS DADOS")
    print("="*80)
    
    # % linhas com SOLD_UNITS_MISMATCH > 0
    if 'SOLD_UNITS_MISMATCH' in df.columns:
        mismatch_count = (df['SOLD_UNITS_MISMATCH'] > 0).sum()
        mismatch_pct = mismatch_count / len(df) * 100
        mismatch_mean = df[df['SOLD_UNITS_MISMATCH'] > 0]['SOLD_UNITS_MISMATCH'].mean()
        
        print(f"\nDivergência entre SOLD_QUANTITY e STOCK_SOLD:")
        print(f"   - Linhas com divergência: {mismatch_count:,} ({mismatch_pct:.1f}%)")
        print(f"   - Magnitude média da divergência: {mismatch_mean:.2f} unidades")
    
    # % oversell por vertical
    if 'OVERSELL_FLAG' in df.columns and 'OVERSELL_UNITS' in df.columns:
        print(f"\nOverselling por VERTICAL:")
        oversell_por_vertical = df.groupby('VERTICAL').agg(
            total_ofertas=('OVERSELL_FLAG', 'size'),
            oversell_ofertas=('OVERSELL_FLAG', 'sum'),
            oversell_units_total=('OVERSELL_UNITS', 'sum'),
        )
        oversell_por_vertical['pct_oversell'] = (
            oversell_por_vertical['oversell_ofertas'] / oversell_por_vertical['total_ofertas'] * 100
        ).round(1)
        oversell_por_vertical = oversell_por_vertical.sort_values('pct_oversell', ascending=False)
        print(oversell_por_vertical.to_string())
    
    # Impacto da ausência de ORIGIN
    if 'ORIGIN' in df.columns:
        origin_missing = df['ORIGIN'].isna().sum()
        origin_missing_pct = origin_missing / len(df) * 100
        print(f"\nImpacto da coluna ORIGIN:")
        print(f"   - Valores ausentes: {origin_missing:,} ({origin_missing_pct:.1f}%)")
        print(f"   - ORIGIN está quase vazia, não é útil para análise")
    
    return df

def analise_duracao_receita(df):
    """
    Análise de duração das ofertas e receita gerada
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("="*60)
    print("ANALISE DE DURACAO E RECEITA")
    print("="*60)
    
    # Estatísticas básicas
    duracao_media = df['DURATION_HOURS'].mean()
    duracao_mediana = df['DURATION_HOURS'].median()
    receita_total = df['SOLD_AMOUNT'].sum()
    receita_media = df['SOLD_AMOUNT'].mean()
    
    print(f"Duracao media das ofertas: {duracao_media:.1f} horas")
    print(f"Duracao mediana das ofertas: {duracao_mediana:.1f} horas")
    print(f"Receita total: R$ {receita_total:,.2f}")
    print(f"Receita media por oferta: R$ {receita_media:,.2f}")
    
    # Gráficos de duração e receita
    plt.figure(figsize=(18, 12))
    
    # Gráfico 1: Duração vs Performance
    plt.subplot(2, 3, 1)
    df.boxplot(column='DURATION_HOURS', by='PERFORMANCE_CATEGORY', ax=plt.gca())
    plt.title('Duracao vs Performance')
    plt.xlabel('Categoria de Performance')
    plt.ylabel('Duracao (horas)')
    plt.suptitle('')  # Remove título automático
    
    # Gráfico 2: Receita por categoria
    plt.subplot(2, 3, 2)
    receita_por_categoria = df.groupby('PERFORMANCE_CATEGORY')['SOLD_AMOUNT'].sum()
    colors_categoria = [plt.cm.Set3(i/len(receita_por_categoria)) for i in range(len(receita_por_categoria))]
    receita_por_categoria.plot(kind='bar', color=colors_categoria, alpha=0.8, ax=plt.gca())
    plt.title('Receita Total por Categoria')
    plt.xlabel('Categoria de Performance')
    plt.ylabel('Receita Total (R$)')
    plt.xticks(rotation=45)
    
    # Adicionar valores nas barras
    for i, v in enumerate(receita_por_categoria.values):
        plt.text(i, v + v*0.01, f'R$ {v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Duração vs Receita
    plt.subplot(2, 3, 3)
    plt.scatter(df['DURATION_HOURS'], df['SOLD_AMOUNT'], alpha=0.6, c=df['SALES_RATE'], cmap='viridis')
    plt.colorbar(label='Taxa de Venda')
    plt.title('Duracao vs Receita')
    plt.xlabel('Duracao (horas)')
    plt.ylabel('Receita (R$)')
    plt.grid(alpha=0.3)
    
    # Gráfico 4: Receita por vertical
    plt.subplot(2, 3, 4)
    receita_vertical = df.groupby('VERTICAL')['SOLD_AMOUNT'].sum().sort_values(ascending=True)
    colors_receita = [plt.cm.Set3(i/len(receita_vertical)) for i in range(len(receita_vertical))]
    receita_vertical.plot(kind='barh', color=colors_receita, alpha=0.8, ax=plt.gca())
    plt.title('Receita Total por Vertical')
    plt.xlabel('Receita Total (R$)')
    plt.ylabel('Vertical')
    
    # Gráfico 5: Duração por vertical
    plt.subplot(2, 3, 5)
    duracao_vertical = df.groupby('VERTICAL')['DURATION_HOURS'].mean().sort_values(ascending=True)
    colors_duracao = [plt.cm.Set3(i/len(duracao_vertical) + 0.3) for i in range(len(duracao_vertical))]
    duracao_vertical.plot(kind='barh', color=colors_duracao, alpha=0.8, ax=plt.gca())
    plt.title('Duracao Media por Vertical')
    plt.xlabel('Duracao Media (horas)')
    plt.ylabel('Vertical')
    
    # Gráfico 6: Receita vs Taxa de venda
    plt.subplot(2, 3, 6)
    plt.scatter(df['SALES_RATE'], df['SOLD_AMOUNT'], alpha=0.6, c=df['DURATION_HOURS'], cmap='plasma')
    plt.colorbar(label='Duracao (horas)')
    plt.title('Taxa de Venda vs Receita')
    plt.xlabel('Taxa de Venda')
    plt.ylabel('Receita (R$)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Insights de duração e receita
    print("\n" + "="*60)
    print("INSIGHTS DE DURACAO E RECEITA")
    print("="*60)
    
    # Melhor vertical por receita
    melhor_vertical_receita = df.groupby('VERTICAL')['SOLD_AMOUNT'].sum().idxmax()
    melhor_receita = df.groupby('VERTICAL')['SOLD_AMOUNT'].sum().max()
    
    print(f"\nMelhor vertical por receita:")
    print(f"   {melhor_vertical_receita}: R$ {melhor_receita:,.2f}")
    
    # Melhor vertical por duração
    melhor_vertical_duracao = df.groupby('VERTICAL')['DURATION_HOURS'].mean().idxmax()
    melhor_duracao = df.groupby('VERTICAL')['DURATION_HOURS'].mean().max()
    
    print(f"\nVertical com maior duracao media:")
    print(f"   {melhor_vertical_duracao}: {melhor_duracao:.1f} horas")
    
    # Análise de eficiência
    df['EFICIENCIA_RECEITA'] = df['SOLD_AMOUNT'] / df['DURATION_HOURS']
    melhor_eficiencia = df.groupby('VERTICAL')['EFICIENCIA_RECEITA'].mean().idxmax()
    melhor_eficiencia_valor = df.groupby('VERTICAL')['EFICIENCIA_RECEITA'].mean().max()
    
    print(f"\nVertical mais eficiente (receita/hora):")
    print(f"   {melhor_eficiencia}: R$ {melhor_eficiencia_valor:.2f}/hora")
    
    # Correlação duração vs receita
    correlacao_duracao_receita = df['DURATION_HOURS'].corr(df['SOLD_AMOUNT'])
    print(f"\nCorrelacao duracao vs receita: {correlacao_duracao_receita:.3f}")
    
    if correlacao_duracao_receita > 0.3:
        print("   Duracao e receita tem correlacao positiva moderada")
    elif correlacao_duracao_receita < -0.3:
        print("   Duracao e receita tem correlacao negativa moderada")
    else:
        print("   Duracao e receita tem correlacao fraca")
    
    return df


def analise_tempo_ate_esgotar(df):
    """
    Aproxima uma curva de 'sobrevivência' até esgotar.
    Usa DURATION_HOURS como censura quando não esgota.
    """
    print("\n" + "="*80)
    print("TEMPO ATÉ ESGOTAR (curva simplificada)")
    print("="*80)
    
    # tempo observado
    df_ = df.copy()
    df_['TIME_OBS'] = df_['DURATION_HOURS']
    # se esgotou, marcar evento
    df_['EVENT'] = (df_['STOCKOUT_FLAG'] == 1).astype(int)
    
    # Kaplan-Meier manual básico por grupo (ex.: SHIPPING_PAYMENT_TYPE)
    for grp_name, gdf in df_.groupby('SHIPPING_PAYMENT_TYPE'):
        gdf = gdf.dropna(subset=['TIME_OBS'])
        n = len(gdf)
        if n == 0: 
            continue
        # discretiza horas
        tbl = (
            gdf.groupby('TIME_OBS')
               .agg(events=('EVENT','sum'), total=('EVENT','size'))
               .sort_index()
        )
        at_risk = n
        surv = 1.0
        surv_curve = []
        for t, row in tbl.iterrows():
            # prob de sobreviver naquele intervalo
            if at_risk > 0:
                surv *= (1 - row['events']/at_risk)
                at_risk -= row['total']
                surv_curve.append((t, max(surv, 0)))
        if surv_curve:
            print(f"- {grp_name}: S(t) final ≈ {surv_curve[-1][1]:.2f}")
        else:
            print(f"- {grp_name}: sem pontos")
    
    return df_


def analise_pareto_gmv(df, nivel='VERTICAL', top_frac=0.1):
    """
    Pareto de GMV por oferta dentro do nível (VERTICAL/domínio).
    top_frac=0.1 => top 10% ofertas por GMV.
    """
    print("\n" + "="*80)
    print(f"PARETO GMV por {nivel}")
    print("="*80)
    
    g = df.groupby(nivel)
    out = []
    for k, gdf in g:
        gdf_ = gdf[['SOLD_AMOUNT']].fillna(0).sort_values('SOLD_AMOUNT', ascending=False)
        if len(gdf_) == 0:
            continue
        k_top = max(1, int(np.ceil(top_frac*len(gdf_))))
        share = gdf_.head(k_top)['SOLD_AMOUNT'].sum() / max(gdf_['SOLD_AMOUNT'].sum(), 1)
        out.append((k, len(gdf_), k_top, share))
    
    res = pd.DataFrame(out, columns=[nivel,'n_ofertas','n_top','gmv_share_top'])
    print(res.sort_values('gmv_share_top', ascending=False).to_string(index=False))
    
    return res


def comparacao_frete_controlada(df):
    """
    Compara SHIPPING_PAYMENT_TYPE dentro de estratos homogêneos de mix:
    (VERTICAL x DURATION_BUCKET). Evita viés de composição.
    """
    print("\n" + "="*80)
    print("FRETE GRÁTIS: COMPARAÇÃO ESTRATIFICADA (VERTICAL x DURATION_BUCKET)")
    print("="*80)
    
    keys = ['VERTICAL','DURATION_BUCKET','SHIPPING_PAYMENT_TYPE']
    agg = df.groupby(keys).agg(
        ofertas=('SOLD_AMOUNT','size'),
        sell_through=('SELL_THROUGH','mean'),
        gmv_per_stock=('GMV_PER_STOCK','mean'),
        stockout_rate=('STOCKOUT_FLAG','mean'),
    ).reset_index()
    
    # Pivot para comparar lado a lado
    piv = agg.pivot_table(index=['VERTICAL','DURATION_BUCKET'],
                          columns='SHIPPING_PAYMENT_TYPE',
                          values=['sell_through','gmv_per_stock','stockout_rate'])
    print(piv.round(3).to_string())
    
    return piv



def kpi_resumo(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Tabela-resumo de KPIs por grupo (VERTICAL ou DAYPART).
    
    KPIs:
      - n_ofertas
      - GMV (SOLD_AMOUNT) - soma
      - ASP (GMV / unidades vendidas)
      - sell_through (Σ STOCK_SOLD / Σ INVOLVED_STOCK)
      - stockout_rate (% de ofertas com STOCKOUT_FLAG=1)
      - pct_oversell (% de ofertas com OVERSELL_FLAG=1)
    """
    use_cols = [
        group_col, 'SOLD_AMOUNT', 'SOLD_QUANTITY', 'INVOLVED_STOCK',
        'STOCK_SOLD', 'STOCKOUT_FLAG', 'OVERSELL_FLAG'
    ]
    miss = [c for c in use_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Colunas ausentes para a tabela de KPIs: {miss}")
    
    g = df[use_cols].copy()
    
    # Somatórios por grupo
    agg_sum = g.groupby(group_col).agg(
        GMV=('SOLD_AMOUNT', 'sum'),
        SOLD_UNITS=('SOLD_QUANTITY', 'sum'),
        INV_STOCK=('INVOLVED_STOCK', 'sum'),
        STOCK_SOLD_SUM=('STOCK_SOLD', 'sum'),
        N=('SOLD_AMOUNT', 'size'),
    )
    
    # Taxas por grupo (médias simples de flags)
    agg_mean = g.groupby(group_col).agg(
        STOCKOUT_RATE=('STOCKOUT_FLAG', 'mean'),
        PCT_OVERSELL=('OVERSELL_FLAG', 'mean'),
    )
    
    out = agg_sum.join(agg_mean, how='left')
    
    # Métricas derivadas e robustez numérica
    out['ASP'] = np.where(out['SOLD_UNITS'] > 0, out['GMV'] / out['SOLD_UNITS'], np.nan)
    out['SELL_THROUGH'] = np.where(out['INV_STOCK'] > 0, out['STOCK_SOLD_SUM'] / out['INV_STOCK'], np.nan)
    
    # Ordenação padrão: maior GMV
    out = out.sort_values('GMV', ascending=False)
    
    # Seleção e formatação final
    out = out.rename(columns={
        'N': 'n_ofertas',
        'STOCKOUT_RATE': 'stockout_rate',
        'PCT_OVERSELL': 'pct_oversell'
    })[
        ['n_ofertas', 'GMV', 'ASP', 'SELL_THROUGH', 'stockout_rate', 'pct_oversell']
    ]
    
    # Opcional: arredondar e formatar (deixe bruto se for exportar)
    out_fmt = out.copy()
    out_fmt['GMV'] = out_fmt['GMV'].round(2)
    out_fmt['ASP'] = out_fmt['ASP'].round(2)
    out_fmt['SELL_THROUGH'] = (out_fmt['SELL_THROUGH'] * 100).round(1)
    out_fmt['stockout_rate'] = (out_fmt['stockout_rate'] * 100).round(1)
    out_fmt['pct_oversell'] = (out_fmt['pct_oversell'] * 100).round(1)
    
    # Renomear percentuais para deixar claro no display
    out_fmt = out_fmt.rename(columns={
        'SELL_THROUGH': 'sell_through_%',
        'stockout_rate': 'stockout_rate_%',
        'pct_oversell': 'pct_oversell_%'
    })
    
    return out, out_fmt


def kpi_resumo_2d(df: pd.DataFrame,
                  rows='VERTICAL',
                  cols='DAYPART'):
    """
    KPIs em duas dimensões (ex.: VERTICAL x DAYPART).
    
    Saídas:
      - dict de DataFrames pivotados (n_ofertas, GMV, ASP, sell_through_%, stockout_rate_%, pct_oversell_%)
      - além disso, plota 3 heatmaps (sell_through, stockout_rate, pct_oversell)
    """
    print("\n" + "="*80)
    print(f"KPIs 2D: {rows} x {cols}")
    print("="*80)
    
    # Base agregada
    base = df[[rows, cols, 'SOLD_AMOUNT', 'SOLD_QUANTITY',
               'INVOLVED_STOCK', 'STOCK_SOLD', 'STOCKOUT_FLAG', 'OVERSELL_FLAG']].copy()
    
    # Somatórios por célula
    agg_sum = base.groupby([rows, cols], dropna=False).agg(
        GMV=('SOLD_AMOUNT', 'sum'),
        SOLD_UNITS=('SOLD_QUANTITY', 'sum'),
        INV_STOCK=('INVOLVED_STOCK', 'sum'),
        STOCK_SOLD_SUM=('STOCK_SOLD', 'sum'),
        N=('SOLD_AMOUNT', 'size'),
    ).reset_index()
    
    # Taxas por célula (médias de flags)
    agg_mean = base.groupby([rows, cols], dropna=False).agg(
        STOCKOUT_RATE=('STOCKOUT_FLAG', 'mean'),
        PCT_OVERSELL=('OVERSELL_FLAG', 'mean'),
    ).reset_index()
    
    df2 = pd.merge(agg_sum, agg_mean, on=[rows, cols], how='left')
    
    # Métricas derivadas robustas
    df2['ASP'] = np.where(df2['SOLD_UNITS'] > 0, df2['GMV'] / df2['SOLD_UNITS'], np.nan)
    df2['SELL_THROUGH'] = np.where(df2['INV_STOCK'] > 0, df2['STOCK_SOLD_SUM'] / df2['INV_STOCK'], np.nan)
    
    # Pivôs
    piv = {}
    for metric, pretty in [
        ('N', 'n_ofertas'),
        ('GMV', 'GMV'),
        ('ASP', 'ASP'),
        ('SELL_THROUGH', 'sell_through_%'),
        ('STOCKOUT_RATE', 'stockout_rate_%'),
        ('PCT_OVERSELL', 'pct_oversell_%'),
    ]:
        tmp = df2.pivot(index=rows, columns=cols, values=metric)
        if metric in ('SELL_THROUGH','STOCKOUT_RATE','PCT_OVERSELL'):
            tmp = (tmp * 100).round(1)
        elif metric in ('GMV','ASP'):
            tmp = tmp.round(2)
        piv[pretty] = tmp
    
    print("\n== KPIs 2D: n_ofertas ==")
    print(piv['n_ofertas'].to_string())
    
    print("\n== KPIs 2D: GMV ==")
    print(piv['GMV'].to_string())
    
    print("\n== KPIs 2D: ASP ==")
    print(piv['ASP'].to_string())
    
    print("\n== KPIs 2D: sell_through_% ==")
    print(piv['sell_through_%'].to_string())
    
    print("\n== KPIs 2D: stockout_rate_% ==")
    print(piv['stockout_rate_%'].to_string())
    
    print("\n== KPIs 2D: pct_oversell_% ==")
    print(piv['pct_oversell_%'].to_string())
    
    # Heatmaps rápidos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(
        axes,
        ['sell_through_%','stockout_rate_%','pct_oversell_%'],
        ['Sell-through (%)','Stockout rate (%)','% Oversell']
    ):
        sns.heatmap(piv[key], annot=True, fmt='.1f', cmap='RdYlGn_r' if key=='sell_through_%' else 'YlOrRd', ax=ax,
                    cbar_kws={'label': title})
        ax.set_title(title)
        ax.set_xlabel(cols); ax.set_ylabel(rows)
    plt.tight_layout()
    plt.show()
    
    return piv
