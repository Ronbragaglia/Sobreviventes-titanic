"""
Módulo de Pré-processamento de Dados

Responsável por limpar e preparar os dados para treinamento do modelo.
"""

import pandas as pd
from typing import List, Tuple


class DataPreprocessor:
    """Gerencia o pré-processamento de dados do Titanic."""

    def __init__(self, features: List[str] = None):
        """Inicializa o pré-processador de dados.

        Args:
            features: Lista de features a serem utilizadas.
        """
        self.features = features or ['fare', 'pclass', 'sex', 'embarked']
        self.target = 'survived'

    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Pré-processa os dados.

        Args:
            data: DataFrame com os dados brutos.

        Returns:
            Tuple com X (features) e y (target).

        Raises:
            ValueError: Se as colunas necessárias não estiverem presentes.
        """
        # Verifica se as colunas necessárias existem
        required_columns = self.features + [self.target]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Colunas ausentes: {missing_columns}")

        # Remove linhas com valores ausentes nas colunas selecionadas
        data_clean = data.dropna(subset=required_columns)

        # Separa features e target
        X = data_clean[self.features]
        y = data_clean[self.target]

        # Aplica One-Hot Encoding nas variáveis categóricas
        X = self._apply_one_hot_encoding(X)

        return X, y

    def _apply_one_hot_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica One-Hot Encoding nas variáveis categóricas.

        Args:
            X: DataFrame com as features.

        Returns:
            DataFrame com One-Hot Encoding aplicado.
        """
        categorical_features = ['sex', 'embarked']
        existing_categorical = [col for col in categorical_features if col in X.columns]

        if not existing_categorical:
            return X

        return pd.get_dummies(X, columns=existing_categorical, drop_first=True)

    def get_feature_importance(self, data: pd.DataFrame) -> pd.Series:
        """Calcula a importância das features.

        Args:
            data: DataFrame com os dados.

        Returns:
            Series com a importância de cada feature.
        """
        # Correlação com a target
        correlations = data[self.features + [self.target]].corr()[self.target].abs()
        return correlations.sort_values(ascending=False)

    def display_preprocessing_info(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Exibe informações sobre o pré-processamento.

        Args:
            X: Features pré-processadas.
            y: Target pré-processado.
        """
        print("\n" + "="*60)
        print(" Informações de Pré-processamento ".center(60))
        print("="*60)
        print(f"\n📊 Shape de X: {X.shape}")
        print(f"📊 Shape de y: {y.shape}")
        print(f"\n📋 Features: {list(X.columns)}")
        print(f"🎯 Target: {y.name}")
        print(f"\n📈 Distribuição da Target:")
        print(y.value_counts())
        print("\n" + "="*60 + "\n")


def preprocess_data(data: pd.DataFrame, features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Função de conveniência para pré-processar dados.

    Args:
        data: DataFrame com os dados brutos.
        features: Lista de features a serem utilizadas.

    Returns:
        Tuple com X (features) e y (target).
    """
    preprocessor = DataPreprocessor(features)
    X, y = preprocessor.preprocess(data)
    preprocessor.display_preprocessing_info(X, y)
    return X, y
