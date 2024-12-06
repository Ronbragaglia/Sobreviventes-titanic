Descrição do Projeto
Este projeto utiliza aprendizado de máquina para prever a sobrevivência dos passageiros do Titanic com base em características como classe do passageiro, sexo, tarifa paga e local de embarque. O modelo é treinado utilizando o Random Forest Classifier, um algoritmo de aprendizado supervisionado popular, para prever se um passageiro sobreviveu ou não ao desastre.

Objetivos
Prever a sobrevivência dos passageiros do Titanic com base em características selecionadas.
Utilizar técnicas de aprendizado supervisionado para treinar um modelo de classificação.
Avaliar o desempenho do modelo usando métricas como precisão, recall e F1-score.
Gerar uma matriz de confusão para analisar os erros do modelo.

Tecnologias Usadas
Python: Linguagem principal utilizada para o desenvolvimento do projeto.
Pandas: Biblioteca para manipulação e análise de dados.
Seaborn: Biblioteca para visualização e carregamento de datasets. O dataset Titanic é carregado diretamente do Seaborn.
Scikit-learn: Biblioteca para machine learning que inclui ferramentas para modelagem, avaliação e pré-processamento de dados.

Estrutura do Código
Carregamento dos Dados: O dataset Titanic é carregado diretamente da biblioteca Seaborn.
Pré-processamento:
Remoção de valores ausentes nas colunas selecionadas (fare, pclass, sex, embarked e survived).
Conversão das variáveis categóricas (sex e embarked) para variáveis numéricas utilizando One-Hot Encoding.
Divisão dos Dados: O dataset é dividido em conjuntos de treinamento (80%) e teste (20%) usando a função train_test_split.
Treinamento do Modelo: O modelo Random Forest Classifier é treinado utilizando o conjunto de dados de treinamento.
Avaliação do Modelo: O desempenho do modelo é avaliado utilizando as métricas de precisão, recall, F1-score e a matriz de confusão.

Exemplo de Saída do Modelo

![image](https://github.com/user-attachments/assets/4044d39a-b8d1-43d5-b415-b7bc81e9fbdf)

Conclusão
Este projeto demonstra como aplicar aprendizado de máquina para prever a sobrevivência dos passageiros do Titanic. Utilizando o Random Forest Classifier, foi possível criar um modelo capaz de classificar corretamente os passageiros, com uma precisão de 81%. O modelo pode ser aprimorado com mais variáveis ou técnicas de pré-processamento, mas já serve como uma introdução prática ao uso de Scikit-learn e Pandas em problemas de classificação.


Contribuições
Contribuições são bem-vindas! Se você deseja melhorar o projeto ou adicionar novos recursos, fique à vontade para enviar uma pull request.
