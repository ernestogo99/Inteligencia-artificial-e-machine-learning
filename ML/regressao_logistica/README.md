# Machine Learning

## 📚 Sumário

- [O que é Machine Learning](#-o-que-é-machine-learning)
- [Tarefa de Classificação](#-tarefa-de-classificação)
- [Normalização](#-normalização)
- [Parâmetros e Hiperparâmetros](#-parâmetros-e-hiperparâmetros)
- [Grid Search e Random Search](#-grid-search-e-random-search)
- [Métricas de Avaliação](#-métricas-de-avaliação)
- [Regressão Logística](#-regressão-logística)
- [Train/Test Split](#-traintest-split)
- [K-Fold Cross Validation](#-k-fold-cross-validation)
- [Fluxo de um Problema de Classificação](#-fluxo-de-um-problema-de-classificação)

---

# 🧠 O que é Machine Learning

Machine Learning é uma área da inteligência artificial que busca desenvolver algoritmos capazes de aprender padrões a partir de dados, permitindo automatizar tarefas normalmente associadas ao raciocínio humano.

Seu objetivo é construir modelos capazes de:

- aprender com exemplos;
- identificar padrões;
- realizar previsões;
- tomar decisões automaticamente.

---

# 🎯 Tarefa de Classificação

A classificação consiste em relacionar vetores de entrada a um conjunto finito de rótulos, categorias ou classes de saída.

## Tipos de Classificação

### Classificação Binária

Problemas que possuem apenas duas classes.

Exemplos:

- sim/não
- positivo/negativo
- fraude/não fraude
- gato/cachorro

---

### Classificação Multiclasse

Problemas com mais de duas classes.

Exemplos:

- reconhecimento de dígitos;
- classificação de letras;
- raças de cachorro;
- marcas de veículos.

---

# ⚙️ Normalização

A normalização é uma técnica de pré-processamento utilizada para colocar os dados em uma mesma escala.

Ela é importante porque muitos algoritmos de Machine Learning são sensíveis à magnitude das variáveis.

## Objetivos da Normalização

- equilibrar escalas;
- evitar dominância de atributos;
- melhorar o treinamento;
- acelerar convergência;
- melhorar desempenho do modelo.

---

# 🧩 Parâmetros e Hiperparâmetros

## Parâmetros

São valores aprendidos automaticamente pelo modelo durante o treinamento.

Exemplos:

- pesos da regressão logística;
- pesos de redes neurais;
- médias e variâncias do Naive Bayes.

---

## Hiperparâmetros

São configurações definidas antes do treinamento que controlam:

- comportamento do algoritmo;
- capacidade do modelo;
- processo de aprendizado.

Os hiperparâmetros não são aprendidos diretamente dos dados.

---

# 🔍 Grid Search e Random Search

Técnicas utilizadas para encontrar os melhores hiperparâmetros.

## Grid Search

O Grid Search testa todas as combinações possíveis de hiperparâmetros definidas em uma grade (_grid_).

### Vantagens

- busca exaustiva;
- simples de implementar;
- garante avaliação de todas as combinações.

### Desvantagens

- alto custo computacional;
- pouco escalável.

---

## Random Search

O Random Search seleciona combinações aleatórias de hiperparâmetros.

### Vantagens

- menor custo computacional;
- mais eficiente em grandes espaços de busca;
- boa escalabilidade.

### Desvantagens

- não garante explorar todas as combinações.

---

# 📊 Métricas de Avaliação

## Matriz de Confusão

Tabela utilizada para resumir os acertos e erros de um classificador.

|                   | Predito Positivo | Predito Negativo |
| ----------------- | ---------------- | ---------------- |
| **Real Positivo** | TP               | FN               |
| **Real Negativo** | FP               | TN               |

Onde:

- **TP** → True Positive
- **TN** → True Negative
- **FP** → False Positive
- **FN** → False Negative

---

## Acurácia

Mede o percentual total de acertos do modelo.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

## Precisão

Mede quantos positivos previstos realmente eram positivos.

```text
Precision = TP / (TP + FP)
```

---

## Revocação (Recall)

Mede quantos positivos reais foram encontrados pelo modelo.

```text
Recall = TP / (TP + FN)
```

---

## F1-score

Média harmônica entre precisão e recall.

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

# 📌 Exemplo de Matriz de Confusão

|                   | Predito Positivo | Predito Negativo |
| ----------------- | ---------------- | ---------------- |
| **Real Positivo** | TP = 90          | FN = 10          |
| **Real Negativo** | FP = 20          | TN = 880         |

---

## Interpretação

- **TP**: o modelo previu positivo e acertou;
- **TN**: o modelo previu negativo e acertou;
- **FP**: o modelo previu positivo, mas errou;
- **FN**: o modelo previu negativo, mas errou.

---

## Cálculo das Métricas

### Acurácia

```text
Accuracy =
(90 + 880) / (90 + 880 + 20 + 10)

Accuracy =
970 / 1000

Accuracy = 0.97
```

**Acurácia = 97%**

---

### Precisão

```text
Precision =
90 / (90 + 20)

Precision =
90 / 110

Precision ≈ 0.818
```

**Precisão ≈ 81.8%**

---

### Recall

```text
Recall =
90 / (90 + 10)

Recall =
90 / 100

Recall = 0.90
```

**Recall = 90%**

---

### F1-score

```text
F1 =
2 * (0.818 * 0.90) / (0.818 + 0.90)

F1 ≈ 0.857
```

## **F1-score ≈ 85.7%**

# 📈 Regressão Logística

A Regressão Logística é um algoritmo de aprendizado supervisionado utilizado principalmente em problemas de classificação, especialmente classificação binária.

Apesar do nome “regressão”, o modelo não prevê valores contínuos. Seu objetivo é estimar a probabilidade de uma instância pertencer a determinada classe.

---

---

# Train/Test Split

O Train/Test Split é uma técnica utilizada para separar os dados em dois conjuntos:

- conjunto de treino;
- conjunto de teste.

O objetivo é avaliar a capacidade de generalização do modelo em dados nunca vistos durante o treinamento.

---

## Funcionamento

Normalmente os dados são divididos em proporções como:

- 80% treino / 20% teste;
- 70% treino / 30% teste.

O modelo aprende padrões utilizando o conjunto de treino e, posteriormente, é avaliado no conjunto de teste(medir desempenho real, verificar generalização, simular novos dados).

---

## Importância

O Train/Test Split ajuda a:

- evitar overfitting(ir bem em dados de treino e mal nos dados novos);
- medir generalização(capacidade do modelo funcionar bem em dados nunca vistos);
- simular dados reais;
- avaliar desempenho de forma mais confiável.

---

## Exemplo com Scikit-Learn

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=357,
    stratify=y
)
```

---

## Principais Parâmetros

| Parâmetro      | Descrição                     |
| -------------- | ----------------------------- |
| `test_size`    | percentual destinado ao teste |
| `random_state` | garante reprodutibilidade     |
| `stratify=y`   | mantém proporção das classes  |

---

## Estratificação

O parâmetro `stratify=y` é importante em problemas de classificação porque mantém a proporção original das classes nos conjuntos de treino e teste.

Isso é especialmente importante em datasets desbalanceados.

---

# 🔄 K-Fold Cross Validation

O K-Fold Cross Validation é uma técnica de validação cruzada utilizada para avaliar modelos de Machine Learning de forma mais robusta.

O conjunto de dados é dividido em \(K\) partes (_folds_). O modelo é treinado múltiplas vezes, utilizando diferentes subconjuntos para treino e teste.

## Benefícios

- reduz variância da avaliação;
- melhora uso dos dados;
- produz estimativas mais confiáveis;
- reduz dependência de um único split.

---

# 🚀 Fluxo de um Problema de Classificação

```text
Entendimento do problema
        ↓
EDA
        ↓
Pré-processamento
        ↓
Train/Test Split
        ↓
Pipeline
        ↓
Baseline Models
        ↓
Grid Search + Cross Validation
        ↓
Nested Cross Validation
        ↓
Avaliação com múltiplas métricas
        ↓
Comparação de modelos
        ↓
Modelo final
```
