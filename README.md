# Desafio Data Science - Meli

**Autor:** Lucas Porto  
**Data:** Outubro 2025  
**Objetivo:** Análise exploratória de ofertas e modelagem preditiva para otimização de operações.

---

## Visão Geral

Este projeto foi desenvolvido para responder às questões do case apresentado pelo time de Data Science da Meli, abordando dois desafios principais:

1. **Análise Exploratória de Ofertas Relâmpago** - Insights sobre performance e otimização
2. **Modelo de Previsão de Falhas** - Manutenção preditiva para dispositivos industriais

## Desafios Resolvidos

### Questão 1: Análise de Ofertas Relâmpago
- **Dataset:** `ofertas_relampago.csv` (48.746 ofertas)
- **Período:** Junho-Julho 2021
- **Objetivo:** EDA e insights sobre performance das ofertas
- **Resultados:** 
  - Receita total: R$ 1,26 milhão
  - Taxa média de venda: 20,7%
  - 49,6% das ofertas sem vendas
  - 7,5% com overselling

### Questão 2: Previsão de Falhas de Dispositivos
- **Dataset:** `full_devices.csv` (dados de sensores)
- **Objetivo:** Modelo preditivo para manutenção preventiva
- **Impacto:** Redução de custos (falha = 4x custo da manutenção preventiva)
- **Modelo:** XGBoost com métricas de negócio otimizadas

## Arquitetura do Projeto

```
desafio_meli/
├── data/                           # Datasets
│   ├── full_devices.csv           # Dados de sensores
│   └── ofertas_relampago.csv      # Ofertas relâmpago
├── models/                         # Modelos treinados
│   ├── device_failure_model_*.pkl # Modelo XGBoost
│   └── device_failure_model_*.json # Métricas e configurações
├── utils/                          # Módulos utilitários
│   ├── analytics_utils.py         # Análises e visualizações
│   ├── business_utils.py          # Regras de negócio
│   ├── data_utils.py              # Processamento de dados
│   ├── feature_utils.py           # Engenharia de features
│   ├── model_utils.py             # Treinamento e avaliação
│   └── pipeline_utils.py          # Pipeline de ML
├── questao_um.ipynb               # EDA Ofertas Relâmpago
├── questao_dois.ipynb             # Modelo de Previsão
└── main.py                        # Script principal
```

## Como Executar

### Pré-requisitos
- Python >= 3.11
- UV (gerenciador de dependências)

### Instalação
```bash
# Clone o repositório
git clone <repository-url>
cd desafio_meli

# Instale as dependências
uv sync
```

### Dependências Principais
- **XGBoost** - Modelo de machine learning
- **Pandas/NumPy** - Manipulação de dados
- **Scikit-learn** - Métricas e validação
- **Matplotlib/Seaborn** - Visualizações
- **LightGBM** - Modelo alternativo

## Principais Insights

### Ofertas Relâmpago
- **Beauty & Health** lidera em receita (43,5% do total)
- **Alta dispersão de performance** - oportunidades de otimização
- **Duração média de 5,7 horas** com GMV de R$ 51,21

### Previsão de Falhas
- **Modelo XGBoost** com alta precisão
- **Regras de negócio** otimizadas para custo-benefício
- **Métricas executivas** para tomada de decisão

## Funcionalidades

### Módulos Utilitários
- **`analytics_utils.py`** - Análises estatísticas e visualizações
- **`business_utils.py`** - Cálculos de impacto financeiro
- **`data_utils.py`** - Carregamento e limpeza de dados
- **`feature_utils.py`** - Engenharia de features
- **`model_utils.py`** - Treinamento e avaliação de modelos
- **`pipeline_utils.py`** - Pipeline completo de ML

## Resultados e Métricas

### Performance do Modelo
- **Precision/Recall** otimizados para regras de negócio
- **ROI calculado** baseado em custos de falha vs manutenção
- **Métricas executivas** para stakeholders

### Análise de Ofertas
- **Segmentação por vertical** e performance
- **Identificação de padrões** de sucesso
- **Recomendações** para otimização

## Notebooks

- **`questao_um.ipynb`** - Análise completa das ofertas relâmpago
- **`questao_dois.ipynb`** - Desenvolvimento e validação do modelo preditivo

## Contribuição

Este é um projeto de desafio técnico desenvolvido para o processo seletivo da Meli.

## Licença

Projeto desenvolvido para fins educacionais e de avaliação técnica.