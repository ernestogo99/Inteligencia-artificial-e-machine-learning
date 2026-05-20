# Machine learning

- Busca automatizar tarefas intelectuais normalmente associadas
  aos humanos.

- Busca algoritmos que permitam aprender a resolver uma tarefa
  (não necessariamente como humanos a resolvem) a partir de dados
  disponíveis.

## Tarefa de classificação

Relaciona vetores de entrada a um número finito de
rótulos/categorias/classes de sa´ıda.

- Classificação binária: Somente duas classes (sim/não,
  positivo/negativo, gato/cachorro, etc.)
- Classificação multiclasse: Mais de duas classes (dígitos,
  letras, raças de cachorro, marcas de carro, etc.)

## Normalização

- A normalização é uma técnica de pré-processamento usada para colocar os dados em uma mesma escala.

Ela é muito importante porque muitos algoritmos de Machine Learning são sensíveis à magnitude das variáveis.

A normalização busca:

- equilibrar as escalas
- evitar dominância de atributos
- melhorar treinamento
- acelerar convergência

## Parâmetro e hiperparâmetro

Parâmetro → aprendido automaticamente pelo modelo durante o treinamento

Hiperparâmetros

São configurações escolhidas antes do treinamento que controlam:

comportamento do algoritmo
capacidade do modelo
processo de aprendizado

Eles não são aprendidos diretamente dos dados.

Selecionamos os melhores hiperparâmetros por meio do Grid Search ou Random Search.

Eles automatizam o processo de testar diferentes combinações de hiperparâmetros e escolher a que produz o melhor desempenho.

- O Grid Search testa todas as combinações possíveis de hiperparâmetros definidos em uma grade (grid).

- O Random Search escolhe combinações de hiperparâmetros aleatoriamente, em vez de testar tudo, ele amostra apenas algumas combinações.

## Métricas

- Matriz de confusão: Tabela que sumariza os erros e acertos de um classificador
- Acurácia: Percentual de acertos gerais do modelo, considerando todos os
  verdadeiros positivos, verdadeiros negativos, falsos positivos e
  falsos negativos.
- Acurácia = TP+TN/+TP+TN FP+FN​
- Precisão: Proporção de exemplos corretamente classificados como positivos em relação ao total de exemplos classificados como positivos.
- Precisão = TP/TP + FP
- Revocação: dentre todos os positivos reais, quantos o modelo conseguiu encontrar.
- Revocação = TP/TP + FN
- f1_score:Média harmônica entre precisão e revocação, usada quando há um balanço entre as duas métricas.

exemplo:

```
# Exemplo de Matriz de Confusão

|                       | Predito Positivo | Predito Negativo |
|-----------------------|------------------|------------------|
| **Real Positivo**     | TP = 90          | FN = 10          |
| **Real Negativo**     | FP = 20          | TN = 880         |

---

# Interpretação

- **TP (True Positive)**: o modelo previu positivo e acertou.
- **TN (True Negative)**: o modelo previu negativo e acertou.
- **FP (False Positive)**: o modelo previu positivo, mas errou.
- **FN (False Negative)**: o modelo previu negativo, mas errou.

---

# Cálculo das Métricas

## Acurácia

Acurácia mede o desempenho geral do modelo.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

\[
Accuracy = \frac{90 + 880}{90 + 880 + 20 + 10}
\]

\[
Accuracy = \frac{970}{1000} = 0.97
\]

**Acurácia = 97%**

---

## Precisão

Precisão mede quantos positivos previstos realmente eram positivos.

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Precision = \frac{90}{90 + 20}
\]

\[
Precision = \frac{90}{110} \approx 0.818
\]

**Precisão ≈ 81.8%**

---

## Revocação (Recall)

Recall mede quantos positivos reais foram encontrados.

\[
Recall = \frac{TP}{TP + FN}
\]

\[
Recall = \frac{90}{90 + 10}
\]

\[
Recall = \frac{90}{100} = 0.90
\]

**Recall = 90%**

---

## F1-score

F1-score é a média harmônica entre precisão e recall.

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

\[
F1 = 2 \cdot \frac{0.818 \cdot 0.90}{0.818 + 0.90}
\]

\[
F1 \approx 0.857
\]

**F1-score ≈ 85.7%**
```

## Regressão logística

A Regressão Logística é um algoritmo de aprendizado supervisionado usado principalmente para problemas de classificação, especialmente classificação binária.

Apesar do nome “regressão”, ela não prevê valores contínuos; ela prevê a probabilidade de um exemplo pertencer a uma determinada classe.

## K-fold

O K-Fold Cross Validation é uma técnica de validação cruzada usada para avaliar modelos de Machine Learning de forma mais robusta e confiável.

A ideia principal é:

dividir os dados em várias partes e treinar/testar o modelo múltiplas vezes.

O K-Fold:

- reduz variância da avaliação
- usa melhor os dados
- produz estimativas mais confiáveis
- reduz dependência de um único split

## Passo a passo para resolver um problema de classificação

```
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
