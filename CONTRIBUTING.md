# 🤝 Contribuindo para o Projeto Titanic

Obrigado pelo seu interesse em contribuir para o **Projeto de Predição de Sobreviventes do Titanic**! Este documento fornece diretrizes e instruções para contribuir com o projeto.

---

## 📋 Índice

- [🎯 Como Contribuir](#-como-contribuir)
- [📝 Código de Conduta](#-código-de-conduta)
- [🔧 Configuração do Ambiente de Desenvolvimento](#-configuração-do-ambiente-de-desenvolvimento)
- [🐛 Processo de Desenvolvimento](#-processo-de-desenvolvimento)
- [📦 Estilo de Código](#-estilo-de-código)
- [🧪 Testes](#-testes)
- [📖 Documentação](#-documentação)
- [🚀 Enviando um Pull Request](#-enviando-um-pull-request)
- [❓ Perguntas Frequentes](#-perguntas-frequentes)

---

## 🎯 Como Contribuir

Existem várias maneiras de contribuir para o projeto:

### 🐛 Reportar Bugs

Se você encontrou um bug, por favor:

1. Verifique se o bug já foi relatado [buscando issues existentes](https://github.com/Ronbragaglia/Sobreviventes-titanic/issues)
2. Se não encontrado, crie uma nova issue com:
   - Título claro e descritivo
   - Descrição detalhada do problema
   - Passos para reproduzir
   - Comportamento esperado vs. comportamento atual
   - Ambiente (Python version, OS, etc.)
   - Logs relevantes (se aplicável)

### 💡 Sugerir Melhorias

Tem uma ideia para melhorar o projeto? Ótimo! Por favor:

1. Verifique se a ideia já foi sugerida
2. Crie uma issue com a tag `enhancement`
3. Descreva a melhoria proposta
4. Explique por que ela seria útil
5. Forneça exemplos, se possível

### 📝 Contribuir com Código

Quer contribuir com código? Excelente! Siga o processo descrito abaixo.

### 📖 Melhorar a Documentação

A documentação é crucial. Você pode:

- Corrigir erros gramaticais ou de ortografia
- Adicionar exemplos de uso
- Melhorar a clareza das explicações
- Traduzir para outros idiomas
- Adicionar diagramas ou imagens

---

## 📝 Código de Conduta

### Nossos Princípios

- **Respeito**: Trate todos com respeito e dignidade
- **Inclusão**: Seja acolhedor com novos contribuidores
- **Colaboração**: Trabalhe juntos para resolver problemas
- **Construtividade**: Forneça feedback construtivo e útil
- **Profissionalismo**: Mantenha um tom profissional em todas as interações

### Comportamento Inaceitável

- Linguagem ofensiva ou discriminatória
- Ataques pessoais ou humilhação
- Assédio de qualquer tipo
- Publicação de informações privadas
- Outras condutas não profissionais

Se você testemunhar ou sofrer qualquer comportamento inaceitável, entre em contato com os mantenedores do projeto.

---

## 🔧 Configuração do Ambiente de Desenvolvimento

### Pré-requisitos

- Python 3.8 ou superior
- Git
- pip

### Passo 1: Fork e Clone

```bash
# Fork o repositório no GitHub
# Clone seu fork
git clone https://github.com/seu-usuario/Sobreviventes-titanic.git
cd Sobreviventes-titanic
```

### Passo 2: Configure o Remote Upstream

```bash
# Adicione o repositório original como upstream
git remote add upstream https://github.com/Ronbragaglia/Sobreviventes-titanic.git
```

### Passo 3: Crie um Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 4: Instale as Dependências

```bash
# Instale as dependências de desenvolvimento
pip install -r requirements.txt
```

### Passo 5: Configure o Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env com suas configurações
```

### Passo 6: Verifique a Instalação

```bash
# Execute os testes
pytest tests/

# Execute o projeto
python -m src.titanic
```

---

## 🐛 Processo de Desenvolvimento

### 1. Escolha uma Issue

- Procure issues com as tags `good first issue` ou `help wanted`
- Comente na issue para indicar que você vai trabalhar nela
- Aguarde aprovação dos mantenedores

### 2. Crie uma Branch

```bash
# Atualize seu branch main
git checkout main
git pull upstream main

# Crie uma branch para sua feature
git checkout -b feature/nome-da-feature

# Ou para um bugfix
git checkout -b fix/nome-do-bug
```

### 3. Faça suas Mudanças

- Escreva código limpo e bem documentado
- Siga o estilo de código do projeto
- Adicione testes para novas funcionalidades
- Atualize a documentação

### 4. Teste suas Mudanças

```bash
# Execute todos os testes
pytest tests/

# Execute testes com cobertura
pytest --cov=src --cov-report=html

# Execute testes específicos
pytest tests/test_model.py
```

### 5. Commit suas Mudanças

```bash
# Adicione os arquivos modificados
git add .

# Commit com uma mensagem clara
git commit -m "feat: adicionar nova funcionalidade de X"

# ou
git commit -m "fix: corrigir bug de Y"
```

### 6. Push e Crie um Pull Request

```bash
# Push para o seu fork
git push origin feature/nome-da-feature

# Abra um Pull Request no GitHub
```

---

## 📦 Estilo de Código

### Formatação

Usamos **Black** para formatação de código:

```bash
# Formatar todo o código
black src/ tests/

# Verificar formatação sem alterar
black --check src/ tests/
```

### Linting

Usamos **Flake8** para linting:

```bash
# Executar linting
flake8 src/ tests/
```

### Type Checking

Usamos **MyPy** para verificação de tipos:

```bash
# Executar verificação de tipos
mypy src/
```

### Convenções de Nomenclatura

- **Classes**: `PascalCase` (ex: `TitanicPredictor`)
- **Funções e Variáveis**: `snake_case` (ex: `carregar_dados`)
- **Constantes**: `UPPER_SNAKE_CASE` (ex: `N_ESTIMATORS`)
- **Módulos**: `snake_case` (ex: `data_loader.py`)

### Docstrings

Use docstrings no estilo Google:

```python
def treinar_modelo(X_treino, y_treino):
    """Treina o modelo Random Forest.

    Args:
        X_treino: Features de treinamento.
        y_treino: Labels de treinamento.

    Returns:
        Modelo treinado.

    Raises:
        ValueError: Se os dados forem inválidos.
    """
    pass
```

---

## 🧪 Testes

### Escrevendo Testes

Use **pytest** para escrever testes:

```python
import pytest
from src.titanic.model import TitanicModel

def test_treinar_modelo():
    """Testa o treinamento do modelo."""
    model = TitanicModel()
    model.train(X_treino, y_treino)

    assert model.is_trained is True
```

### Executando Testes

```bash
# Executar todos os testes
pytest

# Executar com verbose
pytest -v

# Executar com cobertura
pytest --cov=src --cov-report=html

# Executar testes específicos
pytest tests/test_model.py::test_treinar_modelo
```

### Cobertura de Código

A meta é manter pelo menos 80% de cobertura de código:

```bash
# Verificar cobertura
pytest --cov=src --cov-report=html
```

---

## 📖 Documentação

### Atualizando a Documentação

- Mantenha a documentação atualizada com as mudanças
- Use linguagem clara e concisa
- Adicione exemplos de uso
- Inclua diagramas quando apropriado

### Formato

- Use Markdown para documentação
- Siga o estilo do README.md existente
- Adicione links internos quando possível

---

## 🚀 Enviando um Pull Request

### Checklist Antes de Enviar

- [ ] Seu código segue o estilo do projeto
- [ ] Você adicionou testes para novas funcionalidades
- [ ] Todos os testes passam
- [ ] Você atualizou a documentação
- [ ] Você adicionou entradas no CHANGELOG.md
- [ ] Seu PR tem uma descrição clara

### Título do PR

Use o formato conventional commits:

- `feat:` nova funcionalidade
- `fix:` correção de bug
- `docs:` mudanças na documentação
- `style:` formatação, ponto e vírgula, etc.
- `refactor:` refatoração de código
- `test:` adição ou correção de testes
- `chore:` mudanças no processo de build

Exemplo: `feat: adicionar suporte a novos modelos`

### Descrição do PR

Inclua:

- Descrição das mudanças
- Motivo das mudanças
- Issues relacionadas
- Capturas de tela (se aplicável)
- Instruções de teste

---

## ❓ Perguntas Frequentes

### Como faço para testar minha mudança localmente?

```bash
# Ative o ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Execute o projeto
python -m src.titanic
```

### O que fazer se meus testes falharem?

1. Verifique se você instalou todas as dependências
2. Certifique-se de que seu ambiente virtual está ativado
3. Verifique se há conflitos de versão
4. Peça ajuda em uma issue ou no PR

### Posso contribuir sem escrever código?

Sim! Você pode:
- Reportar bugs
- Sugerir melhorias
- Melhorar a documentação
- Traduzir para outros idiomas
- Ajudar outros usuários

### Quanto tempo leva para meu PR ser revisado?

Geralmente revisamos PRs dentro de 2-3 dias úteis. Se você não receber feedback após uma semana, sinta-se à vontade para nos lembrar.

---

## 📚 Recursos Adicionais

- [Documentação do Python](https://docs.python.org/3/)
- [Documentação do Scikit-Learn](https://scikit-learn.org/stable/)
- [Documentação do Pandas](https://pandas.pydata.org/docs/)
- [Guia de Estilo PEP 8](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## 🎉 Agradecimentos

Obrigado por dedicar seu tempo para contribuir com este projeto! Cada contribuição, não importa o tamanho, é valiosa.

---

<div align="center">

**Feito com ❤️ pela comunidade**

[⬆ Voltar ao topo](#-contribuindo-para-o-projeto-titanic)

</div>
