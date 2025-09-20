# MVP Machine Learning - Customer Churn Prediction

O notebook presente no repositório documenta o processo de construção de um modelo de Machine Learning para prever o churn (abandono) de clientes, utilizando o [Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset).

## 1. Introdução

O churn de clientes é um problema crítico para negócios baseados em assinatura. O objetivo é identificar clientes com alta probabilidade de churn para aplicar estratégias de retenção. Este é um problema de classificação supervisionada.

## 2. Visão geral do dataset

O dataset contém mais de 500.000 entradas com 12 colunas, incluindo `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, e `Churn`.

Uma análise inicial revelou:
- Não há valores nulos no dataset.
- A coluna `Churn` apresenta uma proporção maior de clientes que abandonaram o serviço.
- `Support Calls`, `Payment Delay`, e `Total Spend` mostram correlações notáveis com o `Churn`.
- Clientes com planos mensais (`Contract Length_Monthly`) parecem ter menor taxa de abandono.
- Uma redução na concentração de usuários ativos onde a última interação ocorreu a mais de 15 dias (`Last Interaction`) pode indicar uma relação com o comportamento de abandono.

Hipóteses iniciais sobre a importância das features: `Last Interaction`, `Contract Lenght`, `Total Spend`, `Tenure`.

## 3. Preparação dos dados

- O dataset foi separado em conjuntos de treino e teste (80/20).
- Foi utilizada a técnica One-Hot Encoding para as variáveis categóricas (`Gender`, `Subscription Type`, `Contract Length`).
- A normalização dos dados foi realizada utilizando `MinMaxScaler`.
- A seleção de features, utilizando `ExtraTreesClassifier` e `SelectKBest`, indicou as seguintes features como mais importantes: 'Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Gender', e 'Contract Length'.

As features selecionadas para o modelo foram:
'Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Gender_Female', 'Gender_Male', 'Contract Length_Annual', 'Contract Length_Monthly', 'Contract Length_Quarterly'.

## 4. Modelagem e Treinamento

Foram avaliados os seguintes modelos: KNN, Decision Tree, Random Forest, AdaBoost e Logistic Regression.

A avaliação utilizando validação cruzada (KFold com 5 partições) indicou o **Random Forest Classifier** como o modelo com melhor performance em termos de acurácia.

## 5. Avaliação dos resultados

O modelo Random Forest foi treinado com o conjunto completo de treino e avaliado no conjunto de teste usando as métricas: Acurácia, Precisão, Recall e F1-Score.

Resultados iniciais no conjunto de teste:
- Acurácia: 0.9225
- Precisão: 0.8977
- Recall: 0.9710
- F1-Score: 0.9329

Uma otimização de hiperparâmetros utilizando `RandomizedSearchCV` foi realizada para o modelo Random Forest. Os melhores hiperparâmetros encontrados foram:
- 'max_depth': 21
- 'max_features': 'log2'
- 'min_samples_leaf': 3
- 'min_samples_split': 13
- 'n_estimators': 104

Com os hiperparâmetros otimizados, os resultados no conjunto de teste foram:
- Acurácia: 0.9282
- Precisão: 0.8989
- Recall: 0.9810
- F1-Score: 0.9382

A comparação da acurácia entre os conjuntos de treino e teste (0.9281 vs 0.9282) indica que não há overfitting significativo.

## Solução final proposta

A solução final proposta é um modelo **Random Forest Classifier** treinado com as features selecionadas ('Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Gender', 'Contract Length') e os hiperparâmetros otimizados. Este modelo apresenta bom desempenho na identificação de clientes com risco de churn, com destaque para o alto valor de Recall, o que é crucial para ações de prevenção.
