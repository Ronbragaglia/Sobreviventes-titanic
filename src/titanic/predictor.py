"""
Módulo de Predição

Responsável por orquestrar todo o pipeline de predição.
"""

import os
from typing import Optional, Dict
import pandas as pd

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .model import TitanicModel
from .evaluator import ModelEvaluator


class TitanicPredictor:
    """Preditor de sobreviventes do Titanic."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        test_size: float = 0.2,
        features: Optional[list] = None,
        save_plots: bool = True,
        save_model: bool = True,
        model_path: Optional[str] = None
    ):
        """Inicializa o preditor.

        Args:
            n_estimators: Número de estimadores no Random Forest.
            random_state: Semente aleatória para reprodutibilidade.
            test_size: Proporção dos dados de teste.
            features: Lista de features a serem utilizadas.
            save_plots: Se True, salva os gráficos de avaliação.
            save_model: Se True, salva o modelo treinado.
            model_path: Caminho para salvar/carregar o modelo.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.features = features
        self.save_plots = save_plots
        self.save_model = save_model
        self.model_path = model_path or 'models/titanic_model.pkl'

        # Inicializa componentes
        self.data_loader = DataLoader('titanic')
        self.preprocessor = DataPreprocessor(features)
        self.model = TitanicModel(
            n_estimators=n_estimators,
            random_state=random_state,
            test_size=test_size
        )
        self.evaluator = ModelEvaluator(save_plots=save_plots)

        # Dados
        self.data = None
        self.X = None
        self.y = None

    def load_data(self) -> None:
        """Carrega os dados do Titanic."""
        print("\n🔄 Carregando dados...")
        self.data = self.data_loader.load_data()
        self.data_loader.display_info()

    def preprocess_data(self) -> None:
        """Pré-processa os dados."""
        if self.data is None:
            raise RuntimeError("Dados não carregados. Chame load_data() primeiro.")

        print("\n🔄 Pré-processando dados...")
        self.X, self.y = self.preprocessor.preprocess(self.data)
        self.preprocessor.display_preprocessing_info(self.X, self.y)

    def train(self) -> None:
        """Treina o modelo."""
        if self.X is None or self.y is None:
            raise RuntimeError("Dados não pré-processados. Chame preprocess_data() primeiro.")

        print("\n🔄 Treinando modelo...")
        self.model.train(self.X, self.y)
        self.model.display_model_info()

    def evaluate(self) -> Dict:
        """Avalia o modelo."""
        if not self.model.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")

        print("\n🔄 Avaliando modelo...")
        metrics = self.evaluator.evaluate(
            self.model.model,
            self.model.X_test,
            self.model.y_test
        )
        self.evaluator.display_metrics()

        # Plota gráficos
        if self.save_plots:
            print("\n🔄 Gerando gráficos...")
            self.evaluator.plot_confusion_matrix(save=True)
            self.evaluator.plot_feature_importance(self.model.X_train, save=True)

        return metrics

    def predict(self, X: pd.DataFrame) -> list:
        """Faz previsões.

        Args:
            X: Features para previsão.

        Returns:
            Lista com as previsões.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> list:
        """Faz previsões de probabilidade.

        Args:
            X: Features para previsão.

        Returns:
            Lista com as probabilidades.
        """
        return self.model.predict_proba(X)

    def save_trained_model(self) -> None:
        """Salva o modelo treinado."""
        if not self.model.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")

        if self.save_model:
            print("\n🔄 Salvando modelo...")
            self.model.save_model(self.model_path)

    def load_trained_model(self) -> None:
        """Carrega um modelo treinado."""
        print("\n🔄 Carregando modelo...")
        self.model.load_model(self.model_path)
        self.model.display_model_info()

    def run_full_pipeline(self) -> Dict:
        """Executa o pipeline completo.

        Returns:
            Dicionário com as métricas de avaliação.
        """
        # Carrega dados
        self.load_data()

        # Pré-processa dados
        self.preprocess_data()

        # Treina modelo
        self.train()

        # Avalia modelo
        metrics = self.evaluate()

        # Salva modelo
        self.save_trained_model()

        return metrics

    def get_feature_importance(self) -> Dict:
        """Retorna a importância das features.

        Returns:
            Dicionário com a importância de cada feature.
        """
        if not self.model.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")

        return self.model.get_feature_importance()

    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo.
        """
        return self.model.get_model_info()

    def show_metrics(self) -> None:
        """Exibe as métricas de avaliação."""
        if not self.evaluator.metrics:
            print("❌ Nenhuma métrica disponível. Execute evaluate() primeiro.")
            return

        self.evaluator.display_metrics()


def main():
    """Função principal para execução do pipeline."""
    print("\n" + "="*60)
    print(" Predição de Sobreviventes do Titanic ".center(60, "🚢"))
    print("="*60)

    try:
        # Cria o preditor
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=True,
            save_model=True
        )

        # Executa o pipeline completo
        metrics = predictor.run_full_pipeline()

        # Exibe resumo final
        print("\n" + "="*60)
        print(" Resumo Final ".center(60))
        print("="*60)
        print(f"\n🎯 Acurácia Final: {metrics['accuracy']:.2%}")
        print(f"🎯 F1-Score Final: {metrics['f1']:.2%}")
        print(f"🎯 Precisão Final: {metrics['precision']:.2%}")
        print(f"🎯 Recall Final: {metrics['recall']:.2%}")
        print("\n✅ Pipeline concluído com sucesso!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


if __name__ == "__main__":
    main()
