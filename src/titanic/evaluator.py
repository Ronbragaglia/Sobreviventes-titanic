"""
Módulo de Avaliação de Modelo

Responsável por avaliar o desempenho do modelo de ML.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import List, Dict, Optional
import numpy as np


class ModelEvaluator:
    """Gerencia a avaliação do modelo."""

    def __init__(self, save_plots: bool = True, plots_dir: str = 'plots'):
        """Inicializa o avaliador do modelo.

        Args:
            save_plots: Se True, salva os gráficos de avaliação.
            plots_dir: Diretório para salvar os gráficos.
        """
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.metrics = {}

    def evaluate(self, model, X_test, y_test) -> Dict:
        """Avalia o modelo.

        Args:
            model: Modelo treinado.
            X_test: Features de teste.
            y_test: Labels de teste.

        Returns:
            Dicionário com as métricas de avaliação.
        """
        # Faz previsões
        y_pred = model.predict(X_test)

        # Calcula métricas
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return self.metrics

    def display_metrics(self) -> None:
        """Exibe as métricas de avaliação."""
        if not self.metrics:
            print("❌ Nenhuma métrica disponível.")
            return

        print("\n" + "="*60)
        print(" Avaliação do Modelo ".center(60))
        print("="*60)

        # Métricas principais
        print("\n📊 Métricas Principais:")
        print(f"   Acurácia: {self.metrics['accuracy']:.2%}")
        print(f"   Precisão: {self.metrics['precision']:.2%}")
        print(f"   Recall: {self.metrics['recall']:.2%}")
        print(f"   F1-Score: {self.metrics['f1']:.2%}")

        # Relatório de classificação
        print("\n📋 Relatório de Classificação:")
        report = self.metrics['classification_report']
        for label in ['0', '1']:
            if label in report:
                metrics = report[label]
                label_name = 'Não Sobreviveu' if label == '0' else 'Sobreviveu'
                print(f"\n   {label_name}:")
                print(f"      Precisão: {metrics['precision']:.2%}")
                print(f"      Recall: {metrics['recall']:.2%}")
                print(f"      F1-Score: {metrics['f1-score']:.2%}")
                print(f"      Suporte: {metrics['support']}")

        # Matriz de confusão
        print("\n📊 Matriz de Confusão:")
        cm = self.metrics['confusion_matrix']
        print(f"   {'Predito: Não Sobreviveu':^25} {'Predito: Sobreviveu':^25}")
        print(f"   {'Real: Não Sobreviveu':^25} {cm[0,0]:^10} {cm[0,1]:^10}")
        print(f"   {'Real: Sobreviveu':^25} {cm[1,0]:^10} {cm[1,1]:^10}")

        print("\n" + "="*60 + "\n")

    def plot_confusion_matrix(self, save: bool = None) -> None:
        """Plota a matriz de confusão.

        Args:
            save: Se True, salva o gráfico. Usa self.save_plots se None.
        """
        if not self.metrics:
            print("❌ Nenhuma métrica disponível para plotar.")
            return

        save_plot = save if save is not None else self.save_plots

        plt.figure(figsize=(10, 8))
        cm = self.metrics['confusion_matrix']

        # Cria heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Não Sobreviveu', 'Sobreviveu'],
            yticklabels=['Não Sobreviveu', 'Sobreviveu']
        )

        plt.title('Matriz de Confusão', fontsize=16, pad=20)
        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.tight_layout()

        if save_plot:
            import os
            os.makedirs(self.plots_dir, exist_ok=True)
            filepath = os.path.join(self.plots_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Matriz de confusão salva em {filepath}")

        plt.show()
        plt.close()

    def plot_feature_importance(self, model, X_train, save: bool = None) -> None:
        """Plota a importância das features.

        Args:
            model: Modelo treinado.
            X_train: Features de treinamento.
            save: Se True, salva o gráfico. Usa self.save_plots se None.
        """
        save_plot = save if save is not None else self.save_plots

        importances = model.feature_importances_
        feature_names = X_train.columns

        # Ordena por importância
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Importância das Features', fontsize=16, pad=20)
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importância', fontsize=12)
        plt.tight_layout()

        if save_plot:
            import os
            os.makedirs(self.plots_dir, exist_ok=True)
            filepath = os.path.join(self.plots_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Importância de features salva em {filepath}")

        plt.show()
        plt.close()

    def get_metrics_summary(self) -> Dict:
        """Retorna um resumo das métricas.

        Returns:
            Dicionário com o resumo das métricas.
        """
        if not self.metrics:
            return {}

        return {
            'accuracy': self.metrics['accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1_score': self.metrics['f1']
        }


def evaluate_model(model, X_test, y_test, save_plots: bool = True) -> Dict:
    """Função de conveniência para avaliar um modelo.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Labels de teste.
        save_plots: Se True, salva os gráficos.

    Returns:
        Dicionário com as métricas de avaliação.
    """
    evaluator = ModelEvaluator(save_plots=save_plots)
    metrics = evaluator.evaluate(model, X_test, y_test)
    evaluator.display_metrics()
    return metrics
