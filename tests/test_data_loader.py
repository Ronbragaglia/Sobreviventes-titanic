"""
Testes unitários para o módulo de carregamento de dados.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from titanic.data_loader import DataLoader


class TestDataLoader:
    """Testes para a classe DataLoader."""

    def test_initialization(self):
        """Testa se o DataLoader é inicializado corretamente."""
        loader = DataLoader('titanic')
        assert loader.dataset_name == 'titanic'
        assert loader.data is None

    def test_load_data(self):
        """Testa se os dados são carregados corretamente."""
        loader = DataLoader('titanic')
        data = loader.load_data()

        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_get_data_info(self):
        """Testa se as informações dos dados são retornadas corretamente."""
        loader = DataLoader('titanic')
        loader.load_data()
        info = loader.get_data_info()

        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'missing_values' in info
        assert 'memory_usage' in info

    def test_get_data(self):
        """Testa se os dados são retornados corretamente."""
        loader = DataLoader('titanic')
        loader.load_data()
        data = loader.get_data()

        assert data is not None
        assert isinstance(data, pd.DataFrame)

    def test_save_data(self, tmp_path):
        """Testa se os dados são salvos corretamente."""
        loader = DataLoader('titanic')
        loader.load_data()
        loader.save_data(tmp_path)

        assert os.path.exists(tmp_path)

        # Limpa o arquivo de teste
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_save_data_without_loading(self, tmp_path):
        """Testa se erro é lançado ao salvar sem carregar dados."""
        loader = DataLoader('titanic')

        with pytest.raises(ValueError, match="Nenhum dado carregado"):
            loader.save_data(tmp_path)

    def test_invalid_dataset_name(self):
        """Testa se erro é lançado com nome de dataset inválido."""
        loader = DataLoader('invalid_dataset')

        with pytest.raises(RuntimeError, match="Erro ao carregar dataset"):
            loader.load_data()
