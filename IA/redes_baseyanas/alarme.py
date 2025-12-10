import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from utils import visualizar_rede

# Passo 1: Definir a estrutura
modelo_alarme = DiscreteBayesianNetwork([
    ('Roubo', 'Alarme'),
    ('Terremoto', 'Alarme'),
    ('Alarme', 'LigacaoJoao'),
    ('Alarme', 'LigacaoMaria')
])

visualizar_rede(modelo_alarme, "Rede Bayesiana: Sistema de Alarme")


# Passo 2: Definir as CPTs

# P(Roubo) - probabilidade a priori de roubo
cpd_roubo = TabularCPD(
    variable='Roubo',
    variable_card=2,
    values=[[0.999],  # P(Roubo=Não) = 99.9%
            [0.001]]  # P(Roubo=Sim) = 0.1%
)

# P(Terremoto) - probabilidade a priori de terremoto
cpd_terremoto = TabularCPD(
    variable='Terremoto',
    variable_card=2,
    values=[[0.998],  # P(Terremoto=Não) = 99.8%
            [0.002]]  # P(Terremoto=Sim) = 0.2%
)

# P(Alarme | Roubo, Terremoto)
# Esta é uma CPT com 2 pais, então tem 4 colunas (2^2)
# Colunas: [R=0,T=0], [R=0,T=1], [R=1,T=0], [R=1,T=1]
cpd_alarme = TabularCPD(
    variable='Alarme',
    variable_card=2,
    values=[
        [0.999, 0.71, 0.06, 0.05],  # P(Alarme=Não | ...)
        [0.001, 0.29, 0.94, 0.95]   # P(Alarme=Sim | ...)
    ],
    evidence=['Roubo', 'Terremoto'],
    evidence_card=[2, 2]
)

# P(LigacaoJoao | Alarme)
cpd_joao = TabularCPD(
    variable='LigacaoJoao',
    variable_card=2,
    values=[
        [0.95, 0.10],  # P(LigacaoJoao=Não | Alarme)
        [0.05, 0.90]   # P(LigacaoJoao=Sim | Alarme)
    ],
    evidence=['Alarme'],
    evidence_card=[2]
)

# P(LigacaoMaria | Alarme)
cpd_maria = TabularCPD(
    variable='LigacaoMaria',
    variable_card=2,
    values=[
        [0.99, 0.30],  # P(LigacaoMaria=Não | Alarme)
        [0.01, 0.70]   # P(LigacaoMaria=Sim | Alarme)
    ],
    evidence=['Alarme'],
    evidence_card=[2]
)

# Adicionar CPTs ao modelo
modelo_alarme.add_cpds(cpd_roubo, cpd_terremoto, cpd_alarme, cpd_joao, cpd_maria)

# Validar
print("Modelo válido:", modelo_alarme.check_model())

print("\nCPT do Alarme (mais complexa - 2 pais):")
print(cpd_alarme)


# Criar objeto de inferência
inferencia_alarme = VariableElimination(modelo_alarme)

# Cenário 1: João e Maria ligaram. Qual a probabilidade de roubo?
resultado = inferencia_alarme.query(
    variables=['Roubo'],
    evidence={'LigacaoJoao': 1, 'LigacaoMaria': 1}
)
print("Cenário 1: João E Maria ligaram")
print("P(Roubo | João ligou, Maria ligou):")
print(resultado)
print()



# Cenário 2: Apenas João ligou. Qual a probabilidade de roubo?
resultado = inferencia_alarme.query(
    variables=['Roubo'],
    evidence={'LigacaoJoao': 1}
)
print("Cenário 2: Apenas João ligou")
print("P(Roubo | João ligou):")
print(resultado)
print()


# Cenário 3: João ligou e sabemos que NÃO houve terremoto
resultado = inferencia_alarme.query(
    variables=['Roubo'],
    evidence={'LigacaoJoao': 1, 'Terremoto': 0}
)
print("Cenário 3: João ligou E não houve terremoto")
print("P(Roubo | João ligou, Sem terremoto):")
print(resultado)
print()

print("\n💡 Insight:")
print("Eliminar a possibilidade de terremoto aumenta significativamente")
print("a probabilidade de que o alarme foi causado por roubo!")



# Cenário 4: Probabilidade conjunta - qual a chance de roubo E terremoto?
resultado = inferencia_alarme.query(
    variables=['Roubo', 'Terremoto'],
    evidence={'LigacaoJoao': 1, 'LigacaoMaria': 1}
)
print("Cenário 4: Análise conjunta (roubo E terremoto)")
print("P(Roubo, Terremoto | João ligou, Maria ligou):")
print(resultado)