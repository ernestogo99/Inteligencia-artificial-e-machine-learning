## O que são Redes Bayesianas?

Redes Bayesianas são modelos probabilísticos gráficos que representam um conjunto de variáveis aleatórias e suas dependências condicionais através de um grafo acíclico direcionado (DAG).

## Componentes principais:

- Nós: representam variáveis aleatórias
- Arestas: representam dependências probabilísticas diretas
- Tabelas de Probabilidade Condicional (CPTs): quantificam as relações entre variáveis

## Por que usar Redes Bayesianas?

- Modelam incerteza de forma natural
- Permitem inferência probabilística
- Combinam conhecimento de domínio com dados
- São interpretáveis e explicáveis

### Tipos de Conexões em Redes Bayesianas

Existem três padrões básicos de conexão:

1. **Cadeia (Chain)**: A → B → C

   - A e C são independentes dado B

2. **Divergência (Fork)**: A ← B → C

   - A e C são independentes dado B

3. **Convergência (Collider)**: A → B ← C
   - A e C são **dependentes** dado B (caso especial!)

No nosso exemplo do alarme, **Alarme** é um _collider_ (convergência).

## Exemplo 1: Sistema de Diagnóstico Médico Simplificado (diagnostico.py)

### Cenário

Vamos modelar um sistema simples de diagnóstico para **gripe**.

**Variáveis:**

- **Gripe**: pessoa tem gripe? (Sim/Não)
- **Febre**: pessoa tem febre? (Sim/Não)
- **DorCabeca**: pessoa tem dor de cabeça? (Sim/Não)
- **Tosse**: pessoa tem tosse? (Sim/Não)

**Estrutura causal:**

```
      Gripe
     /  |  \
Febre Tosse DorCabeca
```

A gripe **causa** os sintomas (febre, tosse, dor de cabeça).

## Exemplo 2: Sistema de Alarme Residencial

### Cenário

Um sistema de alarme residencial que pode ser ativado por:

- Roubo
- Terremoto

Além disso, o alarme pode levar dois vizinhos (João e Maria) a ligarem para você.

**Variáveis:**

- **Roubo**: ocorreu um roubo? (Sim/Não)
- **Terremoto**: ocorreu um terremoto? (Sim/Não)
- **Alarme**: o alarme disparou? (Sim/Não)
- **LigacaoJoao**: João ligou? (Sim/Não)
- **LigacaoMaria**: Maria ligou? (Sim/Não)

**Estrutura causal:**

```
Roubo    Terremoto
    \    /
     Alarme
     /    \
LigacaoJoao  LigacaoMaria
```

## Exercício Prático - Sistema de Recomendação de Filmes

### Descrição do Problema

Você deve implementar uma Rede Bayesiana para um sistema simples de recomendação de filmes.

### Cenário

Um serviço de streaming quer recomendar se um usuário vai gostar de um filme de ficção científica.

**Variáveis:**

1. **IdadeUsuario**: (Jovem=0, Adulto=1)
2. **GostaCiencia**: usuário gosta de conteúdo científico? (Não=0, Sim=1)
3. **GostaAcao**: usuário gosta de filmes de ação? (Não=0, Sim=1)
4. **AssistiuFilme**: usuário assistiu ao filme? (Não=0, Sim=1)
5. **AvaliacaoPositiva**: usuário deu avaliação positiva? (Não=0, Sim=1)

### Estrutura Causal (Sugerida)

```
      IdadeUsuario
           |
      GostaCiencia ----
           |           \
      GostaAcao -----> AssistiuFilme
                            |
                    AvaliacaoPositiva
```

### Tarefas

#### Tarefa 1: Criar a Estrutura

- Crie um objeto `DiscreteBayesianNetwork` com a estrutura acima
- Visualize a rede usando a função `visualizar_rede()`

#### Tarefa 2: Definir as CPTs

Use as seguintes probabilidades:

**P(IdadeUsuario):**

- P(Jovem) = 0.4
- P(Adulto) = 0.6

**P(GostaCiencia | IdadeUsuario):**

- P(GostaCiencia=Sim | Jovem) = 0.6
- P(GostaCiencia=Sim | Adulto) = 0.7

**P(GostaAcao | IdadeUsuario):**

- P(GostaAcao=Sim | Jovem) = 0.8
- P(GostaAcao=Sim | Adulto) = 0.5

**P(AssistiuFilme | GostaCiencia, GostaAcao):**

- P(Assistiu=Sim | GostaCiencia=Não, GostaAcao=Não) = 0.1
- P(Assistiu=Sim | GostaCiencia=Não, GostaAcao=Sim) = 0.3
- P(Assistiu=Sim | GostaCiencia=Sim, GostaAcao=Não) = 0.4
- P(Assistiu=Sim | GostaCiencia=Sim, GostaAcao=Sim) = 0.8

**P(AvaliacaoPositiva | AssistiuFilme):**

- P(Avaliacao=Positiva | Assistiu=Não) = 0.0
- P(Avaliacao=Positiva | Assistiu=Sim) = 0.75

#### Tarefa 3: Realizar Inferências

Responda às seguintes perguntas usando inferência:

1. Qual a probabilidade de um usuário assistir ao filme sem nenhuma informação adicional?

2. Se sabemos que um usuário é jovem, qual a probabilidade dele assistir ao filme?

3. Se um usuário gosta de ciência E gosta de ação, qual a probabilidade dele dar avaliação positiva?

4. Se um usuário deu avaliação positiva, qual a probabilidade dele gostar de ciência? (inferência reversa!)

#### Tarefa 4: Análise e Discussão

- Quais padrões você observou nas probabilidades?
- Como a rede poderia ser melhorada?
- Que outras variáveis poderiam ser úteis?

---
