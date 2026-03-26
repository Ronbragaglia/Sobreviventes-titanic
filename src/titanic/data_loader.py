"""
Módulo de Carregamento de Dados

Responsável por carregar e preparar os dados do Titanic.
"""

import seaborn as sns
import pandas as pd
from typing import Optional, Tuple


class DataLoader:
    """Gerencia o carregamento de dados do Titanic."""

    def __init__(self, dataset_name: str = 'titanic'):
        """Inicializa o carregador de dados.

        Args:
            dataset_name: Nome do dataset a ser carregado.
        """
        self.dataset_name = dataset_name
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Carrega o dataset do Titanic.

        Returns:
            DataFrame com os dados do Titanic.

        Raises:
            RuntimeError: Se houver erro ao carregar os dados.
        """
        try:
            self.data = sns.load_dataset(self.dataset_name)
            return self.data
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dataset: {e}")

    def get_data_info(self) -> dict:
        """Retorna informações sobre o dataset.

        Returns:
            Dicionário com informações do dataset.
        """
        if self.data is None:
            return {}

        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }

    def display_info(self) -> None:
        """Exibe informações sobre o dataset."""
        info = self.get_data_info()

        if not info:
            print("❌ Nenhum dado carregado.")
            return

        print("\n" + "="*60)
        print(" Informações do Dataset ".center(60))
        print("="*60)
        print(f"\n📊 Shape: {info['shape']}")
        print(f"📋 Colunas: {', '.join(info['columns'][:10])}")
        if len(info['columns']) > 10:
            print(f"   ... e mais {len(info['columns']) - 10} colunas")
        print(f"💾 Uso de Memória: {info['memory_usage']:.2f} MB")
        print(f"\n🔍 Valores Ausentes:")
        for col, count in info['missing_values'].items():
            if count > 0:
                print(f"   {col}: {count}")

        print("\n" + "="*60 + "\n")

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retorna os dados carregados.

        Returns:
            DataFrame com os dados ou None se não carregado.
        """
        return self.data

    def save_data(self, filepath: str) -> None:
        """Salva os dados em um arquivo CSV.

        Args:
            filepath: Caminho do arquivo para salvar.

        Raises:
            ValueError: Se não houver dados carregados.
        """
        if self.data is None:
            raise ValueError("Nenhum dado carregado para salvar.")

        self.data.to_csv(filepath, index=False)
        print(f"✅ Dados salvos em {filepath}")


def load_titanic_data() -> pd.DataFrame:
    """Função de conveniência para carregar dados do Titanic.

    Returns:
        DataFrame com os dados do Titanic.
    """
    loader = DataLoader('titanic')
    data = loader.load_data()
    loader.display_info()
    return data
