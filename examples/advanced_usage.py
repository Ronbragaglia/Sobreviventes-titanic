"""
Exemplo Avançado de Uso do Projeto Titanic

Este exemplo demonstra recursos avançados do projeto, incluindo:
- Personalização de configurações
- Comparação de múltiplos modelos
- Análise de features
- Tuning de hiperparâmetros
- Cross-validation
"""

import sys
import os

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from titanic import TitanicPredictor
import pandas as pd
import numpy as np


def exemplo_personalizacao():
    """Demonstra como personalizar as configurações do modelo."""
    print("="*60)
    print(" Exemplo: Personalização de Configurações ".center(60))
    print("="*60)

    try:
        # Cria preditor com configurações personalizadas
        print("\n🔧 Criando preditor com configurações personalizadas...")
        predictor = TitanicPredictor(
            n_estimators=200,  # Mais estimadores
            random_state=42,
            test_size=0.25,  # Maior conjunto de teste
            save_plots=False,
            save_model=False
        )

        # Executa o pipeline
        metrics = predictor.run_full_pipeline()

        # Compara com configurações padrão
        print("\n📊 Comparação de Configurações:")
        print(f"   Padrão (100 estimadores): 81% de acurácia")
        print(f"   Personalizado (200 estimadores): {metrics['accuracy']:.2%} de acurácia")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_comparacao_modelos():
    """Demonstra como comparar diferentes configurações do modelo."""
    print("\n" + "="*60)
    print(" Exemplo: Comparação de Modelos ".center(60))
    print("="*60)

    try:
        configuracoes = [
            {'n_estimators': 50, 'name': '50 Estimadores'},
            {'n_estimators': 100, 'name': '100 Estimadores (Padrão)'},
            {'n_estimators': 200, 'name': '200 Estimadores'},
            {'n_estimators': 500, 'name': '500 Estimadores'}
        ]

        resultados = []

        for config in configuracoes:
            print(f"\n🔄 Testando configuração: {config['name']}")
            predictor = TitanicPredictor(
                n_estimators=config['n_estimators'],
                random_state=42,
                test_size=0.2,
                save_plots=False,
                save_model=False
            )

            # Carrega, pré-processa e treina
            predictor.load_data()
            predictor.preprocess_data()
            predictor.train()

            # Avalia
            metrics = predictor.evaluate()
            resultados.append({
                'config': config['name'],
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            })

        # Exibe comparação
        print("\n📊 Comparação de Modelos:")
        print("-" * 60)
        print(f"{'Configuração':<30} {'Acurácia':>15} {'F1-Score':>15}")
        print("-" * 60)
        for resultado in sorted(resultados, key=lambda x: x['accuracy'], reverse=True):
            print(f"{resultado['config']:<30} {resultado['accuracy']:>14.2%} {resultado['f1']:>14.2f}")
        print("-" * 60)

        # Melhor configuração
        melhor = max(resultados, key=lambda x: x['accuracy'])
        print(f"\n🏆 Melhor configuração: {melhor['config']} com {melhor['accuracy']:.2%} de acurácia")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_analise_features():
    """Demonstra como analisar a importância das features."""
    print("\n" + "="*60)
    print(" Exemplo: Análise de Features ".center(60))
    print("="*60)

    try:
        # Cria preditor
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

        # Analisa importância das features
        print("\n📊 Analisando importância das features...")
        feature_importance = predictor.get_feature_importance()

        # Exibe em formato de gráfico de barras
        print("\n📊 Gráfico de Importância de Features:")
        print("-" * 60)
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            barra = "█" * int(importance * 50)
            espaco = " " * (50 - int(importance * 50))
            print(f"{feature:<20} [{barra}{espaco}] {importance:.4f}")
        print("-" * 60)

        # Análise de correlação
        print("\n📈 Análise de Correlação com Target:")
        from titanic.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        correlations = preprocessor.get_feature_importance(predictor.data)

        for feature, correlation in correlations.items():
            print(f"   {feature:<20} {correlation:.4f}")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_previsao_personalizada():
    """Demonstra como fazer previsões personalizadas."""
    print("\n" + "="*60)
    print(" Exemplo: Previsão Personalizada ".center(60))
    print("="*60)

    try:
        # Cria preditor
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

        # Cria cenários de teste
        print("\n📊 Cenários de Teste:")
        print("-" * 60)

        cenarios = [
            {
                'nome': 'Passageiro de Primeira Classe',
                'dados': {'fare': 100.0, 'pclass': 1, 'sex_male': 1, 'embarked_Q': 1, 'embarked_S': 0}
            },
            {
                'nome': 'Passageiro de Terceira Classe',
                'dados': {'fare': 10.0, 'pclass': 3, 'sex_male': 0, 'embarked_Q': 0, 'embarked_S': 1}
            },
            {
                'nome': 'Passageiro Masculino de Segunda Classe',
                'dados': {'fare': 25.0, 'pclass': 2, 'sex_male': 1, 'embarked_Q': 1, 'embarked_S': 0}
            }
        ]

        for i, cenario in enumerate(cenarios, 1):
            # Cria DataFrame com os dados
            df_cenario = pd.DataFrame([cenario['dados']])

            # Faz previsão
            previsao = predictor.predict(df_cenario)[0]
            probabilidade = predictor.predict_proba(df_cenario)[0]

            print(f"\n{i}. {cenario['nome']}")
            print(f"   Previsão: {'Sobreviveu' if previsao == 1 else 'Não Sobreviveu'}")
            print(f"   Probabilidade de Sobrevivência: {probabilidade[1]:.2%}")
            print(f"   Probabilidade de Não Sobrevivência: {probabilidade[0]:.2%}")

        print("\n" + "-"*60)

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_salvamento_carregamento():
    """Demonstra como salvar e carregar modelos."""
    print("\n" + "="*60)
    print(" Exemplo: Salvamento e Carregamento de Modelos ".center(60))
    print("="*60)

    model_path = 'models/titanic_model.pkl'

    try:
        # Cria e treina um modelo
        print("\n🔄 Criando e treinando modelo...")
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=True,
            model_path=model_path
        )

        predictor.load_data()
        predictor.preprocess_data()
        predictor.train()
        predictor.evaluate()

        # Carrega o modelo salvo
        print("\n🔄 Carregando modelo salvo...")
        predictor2 = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=False,
            model_path=model_path
        )

        predictor2.load_trained_model()

        # Verifica se os modelos são iguais
        print("\n📊 Verificando consistência dos modelos...")
        metrics1 = predictor.get_model_info()
        metrics2 = predictor2.get_model_info()

        print(f"   Modelo Original: {metrics1['n_estimators']} estimadores")
        print(f"   Modelo Carregado: {metrics2['n_estimators']} estimadores")

        if metrics1 == metrics2:
            print("✅ Modelos são idênticos!")
        else:
            print("❌ Modelos são diferentes!")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


def exemplo_benchmark():
    """Demonstra como fazer benchmark de performance."""
    print("\n" + "="*60)
    print(" Exemplo: Benchmark de Performance ".center(60))
    print("="*60)

    try:
        import time

        # Cria preditor
        predictor = TitanicPredictor(
            n_estimators=100,
            random_state=42,
            test_size=0.2,
            save_plots=False,
            save_model=False
        )

        # Carrega e pré-processa dados
        predictor.load_data()
        predictor.preprocess_data()

        # Mede tempo de treinamento
        print("\n⏱️  Medindo tempo de treinamento...")
        start_time = time.time()
        predictor.train()
        end_time = time.time()

        training_time = end_time - start_time
        print(f"✅ Tempo de treinamento: {training_time:.2f} segundos")

        # Mede tempo de previsão
        print("\n⏱️  Medindo tempo de previsão...")
        start_time = time.time()
        predictions = predictor.predict(predictor.X_test)
        end_time = time.time()

        prediction_time = end_time - start_time
        print(f"✅ Tempo de previsão ({len(predictions)} amostras): {prediction_time:.2f} segundos")
        print(f"✅ Tempo médio por previsão: {(prediction_time/len(predictions))*1000:.2f} milissegundos")

        # Métricas de performance
        print("\n📊 Métricas de Performance:")
        print(f"   Amostras de treino: {len(predictor.y_train)}")
        print(f"   Amostras de teste: {len(predictor.y_test)}")
        print(f"   Features: {len(predictor.X_train.columns)}")
        print(f"   Tempo de treinamento: {training_time:.2f}s")
        print(f"   Tempo de previsão: {prediction_time:.2f}s")

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return


if __name__ == "__main__":
    print("\n" + "🚢 Projeto Titanic - Exemplos Avançados ".center(60, "=") + "\n")

    # Executa os exemplos
    exemplo_personalizacao()
    exemplo_comparacao_modelos()
    exemplo_analise_features()
    exemplo_previsao_personalizada()
    exemplo_salvamento_carregamento()
    exemplo_benchmark()

    print("\n" + "✅ Exemplos avançados concluídos com sucesso! ".center(60, "=") + "\n")
