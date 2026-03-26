"""
Módulo de Modelo de Machine Learning

Responsável por criar e treinar o modelo de predição.
"""

import os
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


class TitanicModel:
    """Modelo de predição de sobreviventes do Titanic."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        test_size: float = 0.2
    ):
        """Inicializa o modelo Titanic.

        Args:
            n_estimators: Número de estimadores no Random Forest.
            random_state: Semente aleatória para reprodutibilidade.
            test_size: Proporção dos dados de teste.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, X, y) -> None:
        """Treina o modelo Random Forest.

        Args:
            X: Features de treinamento.
            y: Labels de treinamento.
        """
        # Divide os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Cria e treina o modelo
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

        print(f"✅ Modelo treinado com sucesso!")
        print(f"📊 Dados de treino: {self.X_train.shape}")
        print(f"📊 Dados de teste: {self.X_test.shape}")

    def predict(self, X) -> list:
        """Faz previsões usando o modelo treinado.

        Args:
            X: Features para previsão.

        Returns:
            Lista com as previsões.

        Raises:
            RuntimeError: Se o modelo não estiver treinado.
        """
        if not self.is_trained:
            raise RuntimeError("O modelo não está treinado. Chame train() primeiro.")

        predictions = self.model.predict(X)
        return predictions.tolist()

    def predict_proba(self, X) -> list:
        """Faz previsões de probabilidade.

        Args:
            X: Features para previsão.

        Returns:
            Lista com as probabilidades.

        Raises:
            RuntimeError: Se o modelo não estiver treinado.
        """
        if not self.is_trained:
            raise RuntimeError("O modelo não está treinado. Chame train() primeiro.")

        probas = self.model.predict_proba(X)
        return probas.tolist()

    def get_feature_importance(self) -> dict:
        """Retorna a importância das features.

        Returns:
            Dicionário com a importância de cada feature.

        Raises:
            RuntimeError: Se o modelo não estiver treinado.
        """
        if not self.is_trained:
            raise RuntimeError("O modelo não está treinado. Chame train() primeiro.")

        importances = self.model.feature_importances_
        feature_names = self.X_train.columns

        return dict(zip(feature_names, importances))

    def save_model(self, filepath: str) -> None:
        """Salva o modelo treinado.

        Args:
            filepath: Caminho do arquivo para salvar.

        Raises:
            RuntimeError: Se o modelo não estiver treinado.
        """
        if not self.is_trained:
            raise RuntimeError("O modelo não está treinado. Chame train() primeiro.")

        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        joblib.dump(self.model, filepath)
        print(f"✅ Modelo salvo em {filepath}")

    def load_model(self, filepath: str) -> None:
        """Carrega um modelo treinado.

        Args:
            filepath: Caminho do arquivo do modelo.
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✅ Modelo carregado de {filepath}")

    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo.
        """
        if not self.is_trained:
            return {
                'is_trained': False,
                'model_type': None
            }

        return {
            'is_trained': True,
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'n_features': len(self.X_train.columns) if self.X_train is not None else 0,
            'n_train_samples': len(self.y_train) if self.y_train is not None else 0,
            'n_test_samples': len(self.y_test) if self.y_test is not None else 0
        }

    def display_model_info(self) -> None:
        """Exibe informações sobre o modelo."""
        info = self.get_model_info()

        print("\n" + "="*60)
        print(" Informações do Modelo ".center(60))
        print("="*60)
        print(f"\n🤖 Tipo: {info['model_type']}")
        print(f"🔢 Número de Estimadores: {info['n_estimators']}")
        print(f"🎲 Semente Aleatória: {info['random_state']}")
        print(f"📊 Tamanho do Teste: {info['test_size']}")
        print(f"📋 Número de Features: {info['n_features']}")
        print(f"📊 Amostras de Treino: {info['n_train_samples']}")
        print(f"📊 Amostras de Teste: {info['n_test_samples']}")
        print(f"\n🎯 Treinado: {'✅ Sim' if info['is_trained'] else '❌ Não'}")
        print("\n" + "="*60 + "\n")
