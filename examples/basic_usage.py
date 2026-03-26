"""
Exemplo Básico de Uso do Projeto Titanic

Este exemplo demonstra como usar o preditor de forma simples e direta.
"""

import sys
import os

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from titanic import TitanicPredictor


def exemplo_basico():
    """Demonstra o uso básico do preditor."""
    print("="*60)
    print(" Exemplo Básico de Uso ".center(60))
    print("="*60)

    try:
        # Inicializa o preditor
        print("\n🚀 Inicializando o preditor...")
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=False
        )

        # Carrega dados
        print("\n📊 Carregando dados...")
        predictor.load_data()

        # Pré-processa dados
        print("\n🔄 Pré-processando dados...")
        predictor.preprocess_data()

        # Treina o modelo
        print("\n🤖 Treinando o modelo...")
        predictor.train()

        # Avalia o modelo
        print("\n📈 Avaliando o modelo...")
        metrics = predictor.evaluate()

        # Exibe métricas
        print("\n📊 Métricas de Avaliação:")
        print(f"   Acurácia: {metrics['accuracy']:.2%}")
        print(f"   Precisão: {metrics['precision']:.2%}")
        print(f"   Recall: {metrics['recall']:.2%}")
        print(f"   F1-Score: {metrics['f1']:.2%}")

        # Exibe informações do modelo
        print("\n🤖 Informações do Modelo:")
        model_info = predictor.get_model_info()
        print(f"   Tipo: {model_info['model_type']}")
        print(f"   Estimadores: {model_info['n_estimators']}")
        print(f"   Features: {model_info['n_features']}")
        print(f"   Amostras de Treino: {model_info['n_train_samples']}")
        print(f"   Amostras de Teste: {model_info['n_test_samples']}")

        # Exibe importância das features
        print("\n📊 Importância das Features:")
        feature_importance = predictor.get_feature_importance()
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {importance:.4f}")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_previsao():
    """Demonstra como fazer previsões."""
    print("\n" + "="*60)
    print(" Exemplo de Previsão ".center(60))
    print("="*60)

    try:
        # Inicializa o preditor
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=False
        )

        # Carrega, pré-processa e treina
        predictor.load_data()
        predictor.preprocess_data()
        predictor.train()

        # Cria dados de exemplo para previsão
        import pandas as pd
        exemplo_dados = pd.DataFrame({
            'fare': [50.0, 30.0, 100.0],
            'pclass': [1, 2, 3],
            'sex_male': [1, 0, 1],
            'embarked_Q': [1, 0, 0],
            'embarked_S': [0, 1, 0]
        })

        # Faz previsões
        print("\n🔮 Fazendo previsões...")
        previsoes = predictor.predict(exemplo_dados)

        # Exibe resultados
        print("\n📊 Resultados das Previsões:")
        for i, previsao in enumerate(previsoes, 1):
            status = "Sobreviveu" if previsao == 1 else "Não Sobreviveu"
            print(f"   Passageiro {i}: {status}")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_importancia_features():
    """Demonstra como analisar a importância das features."""
    print("\n" + "="*60)
    print(" Exemplo de Importância de Features ".center(60))
    print("="*60)

    try:
        # Inicializa o preditor
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=False
        )

        # Carrega, pré-processa e treina
        predictor.load_data()
        predictor.preprocess_data()
        predictor.train()

        # Obtém importância das features
        print("\n📊 Analisando importância das features...")
        feature_importance = predictor.get_feature_importance()

        # Exibe em formato de tabela
        print("\n📋 Tabela de Importância:")
        print("-" * 60)
        print(f"{'Feature':<30} {'Importância':>20}")
        print("-" * 60)
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:>20.4f}")
        print("-" * 60)

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


if __name__ == "__main__":
    print("\n" + "🚢 Projeto Titanic - Exemplos Básicos ".center(60, "=") + "\n")

    # Executa os exemplos
    exemplo_basico()
    exemplo_previsao()
    exemplo_importancia_features()

    print("\n" + "✅ Exemplos básicos concluídos com sucesso! ".center(60, "=") + "\n")
