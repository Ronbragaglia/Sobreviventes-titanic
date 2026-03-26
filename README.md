# 🚢 Predição de Sobreviventes do Titanic com Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**Projeto de Machine Learning para prever a sobrevivência dos passageiros do Titanic.**

[Documentação](#-documentação) • [Instalação](#-instalação) • [Uso](#-uso) • [Contribuindo](#-contribuindo)

</div>

---

## 📋 Índice

- [🎯 Visão Geral](#-visão-geral)
- [✨ Recursos Principais](#-recursos-principais)
- [🏗️ Arquitetura](#️-arquitetura)
- [📦 Instalação](#-instalação)
- [🚀 Uso](#-uso)
- [🔧 Configuração](#-configuração)
- [📊 Métricas](#-métricas)
- [🤝 Contribuindo](#-contribuindo)
- [📄 Licença](#-licença)

---

## 🎯 Visão Geral

Este projeto utiliza **aprendizado de máquina** para prever a sobrevivência dos passageiros do Titanic com base em características como classe do passageiro, sexo, tarifa paga e local de embarque. O modelo é treinado utilizando o **Random Forest Classifier**, um algoritmo de aprendizado supervisionado popular, para prever se um passageiro sobreviveu ou não ao desastre.

### Objetivos

- ✅ Prever a sobrevivência dos passageiros do Titanic com base em características selecionadas
- ✅ Utilizar técnicas de aprendizado supervisionado para treinar um modelo de classificação
- ✅ Avaliar o desempenho do modelo usando métricas como precisão, recall e F1-score
- ✅ Gerar uma matriz de confusão para analisar os erros do modelo

### Casos de Uso

- Análise histórica de dados do Titanic
- Treinamento e avaliação de modelos de classificação
- Previsão de sobrevivência em novos dados
- Estudo de técnicas de pré-processamento de dados

---

## ✨ Recursos Principais

### 🤖 Machine Learning
- ✅ **Random Forest Classifier**: Algoritmo robusto para classificação
- ✅ **One-Hot Encoding**: Conversão de variáveis categóricas
- ✅ **Train/Test Split**: Divisão adequada dos dados
- ✅ **Múltiplas Métricas**: Precisão, Recall, F1-Score

### 📊 Análise de Dados
- ✅ **Carregamento via Seaborn**: Dataset Titanic pronto para uso
- ✅ **Pré-processamento**: Remoção de valores ausentes
- ✅ **Seleção de Features**: Características relevantes identificadas
- ✅ **Matriz de Confusão**: Análise visual dos erros

### 🛠️ Funcionalidades
- ✅ **Pipeline Completo**: Do carregamento à avaliação
- ✅ **Logging**: Registro de operações e métricas
- ✅ **Persistência de Modelo**: Salvamento do modelo treinado
- ✅ **Visualizações**: Gráficos de resultados

---

## 🏗️ Arquitetura

```
titanic-survival-prediction/
├── src/
│   └── titanic/
│       ├── __init__.py
│       ├── data_loader.py      # Carregamento de dados
│       ├── preprocessor.py     # Pré-processamento
│       ├── model.py           # Modelo ML
│       ├── evaluator.py       # Avaliação
│       └── predictor.py       # Predição
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_evaluator.py
├── examples/
│   ├── __init__.py
│   ├── basic_usage.py
│   └── advanced_usage.py
├── docs/
│   ├── architecture.md
│   ├── data_analysis.md
│   └── model_evaluation.md
├── data/                      # Dados brutos e processados
├── models/                    # Modelos treinados
├── logs/                      # Logs de execução
├── plots/                     # Visualizações
├── .env.example              # Exemplo de configuração
├── .gitignore                # Arquivos ignorados
├── LICENSE                   # Licença MIT
├── requirements.txt          # Dependências
├── README.md                 # Este arquivo
├── CONTRIBUTING.md           # Guia de contribuição
└── CHANGELOG.md             # Histórico de mudanças
```

---

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Clone o Repositório

```bash
git clone https://github.com/Ronbragaglia/Sobreviventes-titanic.git
cd Sobreviventes-titanic
```

### Passo 2: Crie um Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instale as Dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Configure o Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env com suas configurações
```

### Passo 5: Execute o Projeto

```bash
python -m src.titanic
```

---

## 🚀 Uso

### Uso Básico

```python
from src.titanic import TitanicPredictor

# Inicialize o preditor
predictor = TitanicPredictor()

# Treine o modelo
predictor.train()

# Avalie o modelo
predictor.evaluate()

# Faça previsões
predictions = predictor.predict(new_data)
```

### Exemplo Completo

```python
from src.titanic import TitanicPredictor

# Inicialize o preditor
predictor = TitanicPredictor()

# Treine o modelo
print("🔄 Treinando o modelo...")
predictor.train()

# Avalie o modelo
print("\n📊 Avaliando o modelo...")
predictor.evaluate()

# Exiba as métricas
predictor.show_metrics()

# Salve o modelo
predictor.save_model("models/titanic_model.pkl")
```

### Exemplo de Saída

```
🔄 Treinando o modelo...
✅ Modelo treinado com sucesso!

📊 Avaliando o modelo:
              precision    recall  f1-score   support

           0       0.81      0.88      0.84       109
           1       0.79      0.69      0.74        69

    accuracy                           0.81       178
   macro avg       0.80      0.79      0.79       178
weighted avg       0.80      0.81      0.80       178

Matriz de Confusão:
[[96 13]
 [21 48]]

✅ Acurácia: 81.0%
```

---

## 🔧 Configuração

### Variáveis de Ambiente

Configure as seguintes variáveis no arquivo `.env`:

```env
# Model Configuration
MODEL_NAME=RandomForest
N_ESTIMATORS=100
RANDOM_STATE=42

# Data Configuration
TEST_SIZE=0.2
FEATURES=fare,pclass,sex,embarked

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/titanic.log

# Visualization Configuration
SAVE_PLOTS=True
PLOTS_DIR=plots

# Model Persistence
SAVE_MODEL=True
MODEL_PATH=models/titanic_model.pkl

# Application Settings
APP_NAME=Predição de Sobreviventes do Titanic
APP_VERSION=2.0.0
```

### Configuração do Modelo

- **MODEL_NAME**: Nome do modelo a ser usado (padrão: `RandomForest`)
- **N_ESTIMATORS**: Número de estimadores no Random Forest (padrão: `100`)
- **RANDOM_STATE**: Semente aleatória para reprodutibilidade (padrão: `42`)

### Configuração dos Dados

- **TEST_SIZE**: Proporção dos dados de teste (padrão: `0.2`)
- **FEATURES**: Features selecionadas para o modelo (padrão: `fare,pclass,sex,embarked`)

---

## 📊 Métricas

### Resultados do Modelo

- **Acurácia**: 81%
- **Precisão**: 80% (média ponderada)
- **Recall**: 81% (média ponderada)
- **F1-Score**: 80% (média ponderada)

### Matriz de Confusão

|                | Previsto: Não Sobreviveu | Previsto: Sobreviveu |
|----------------|-------------------------|---------------------|
| **Real: Não Sobreviveu** | 96 (Verdadeiro Negativo) | 13 (Falso Positivo) |
| **Real: Sobreviveu** | 21 (Falso Negativo) | 48 (Verdadeiro Positivo) |

### Análise dos Resultados

- **Verdadeiros Negativos (96)**: O modelo previu corretamente que o passageiro não sobreviveu
- **Falsos Positivos (13)**: O modelo previu que o passageiro sobreviveu, mas ele não sobreviveu
- **Falsos Negativos (21)**: O modelo previu que o passageiro não sobreviveu, mas ele sobreviveu
- **Verdadeiros Positivos (48)**: O modelo previu corretamente que o passageiro sobreviveu

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia o [`CONTRIBUTING.md`](CONTRIBUTING.md) para detalhes sobre nosso código de conduta e o processo para nos enviar pull requests.

### Como Contribuir

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Desenvolvimento

```bash
# Clone seu fork
git clone https://github.com/seu-usuario/Sobreviventes-titanic.git

# Crie uma branch
git checkout -b minha-feature

# Faça suas mudanças
# ...

# Teste suas mudanças
pytest tests/

# Formate seu código
black src/ tests/

# Commit e push
git add .
git commit -m "Descrição das mudanças"
git push origin minha-feature
```

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [`LICENSE`](LICENSE) para detalhes.

---

## 📝 Changelog

Veja o [`CHANGELOG.md`](CHANGELOG.md) para um histórico de mudanças.

---

## 🙋 Suporte

Se você tiver alguma dúvida ou precisar de ajuda:

- Abra uma [issue](https://github.com/Ronbragaglia/Sobreviventes-titanic/issues)
- Consulte a [documentação](docs/)
- Entre em contato: ronbragaglia@gmail.com

---

## 🌟 Reconhecimentos

- [Scikit-Learn](https://scikit-learn.org/) pela biblioteca de machine learning
- [Pandas](https://pandas.pydata.org/) pela biblioteca de manipulação de dados
- [Seaborn](https://seaborn.pydata.org/) pelo dataset Titanic
- Comunidade Python pelas bibliotecas incríveis
- Todos os contribuidores que ajudaram a melhorar este projeto

---

<div align="center">

**Feito com ❤️ por [Ron Bragaglia](https://github.com/Ronbragaglia)**

[⬆ Voltar ao topo](#-predição-de-sobreviventes-do-titanic-com-machine-learning)

</div>
