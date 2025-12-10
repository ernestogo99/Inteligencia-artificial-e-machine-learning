# Importações necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from utils import visualizar_rede

# Configuração para visualização
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Passo 1: Definir a estrutura da rede (grafo)
modelo_gripe = DiscreteBayesianNetwork([
    ('Gripe', 'Febre'),
    ('Gripe', 'Tosse'),
    ('Gripe', 'DorCabeca')
])

print("Estrutura da rede criada!")
print(f"Nós: {modelo_gripe.nodes()}")
print(f"Arestas: {modelo_gripe.edges()}")

# Passo 2: Definir as Tabelas de Probabilidade Condicional (CPTs)

# CPT para Gripe (probabilidade a priori)
# P(Gripe=Sim) = 0.1 (10% da população tem gripe)
cpd_gripe = TabularCPD(
    variable='Gripe',
    variable_card=2,
    values=[[0.9],  # P(Gripe=Não)
            [0.1]]  # P(Gripe=Sim)
)

# CPT para Febre dado Gripe
# P(Febre=Sim | Gripe=Sim) = 0.8
# P(Febre=Sim | Gripe=Não) = 0.1
cpd_febre = TabularCPD(
    variable='Febre',
    variable_card=2,
    values=[[0.9, 0.2],   # P(Febre=Não | Gripe)
            [0.1, 0.8]],  # P(Febre=Sim | Gripe)
    evidence=['Gripe'],
    evidence_card=[2]
)

# CPT para Tosse dado Gripe
# P(Tosse=Sim | Gripe=Sim) = 0.7
# P(Tosse=Sim | Gripe=Não) = 0.15
cpd_tosse = TabularCPD(
    variable='Tosse',
    variable_card=2,
    values=[[0.85, 0.3],   # P(Tosse=Não | Gripe)
            [0.15, 0.7]],  # P(Tosse=Sim | Gripe)
    evidence=['Gripe'],
    evidence_card=[2]
)

# CPT para DorCabeca dado Gripe
# P(DorCabeca=Sim | Gripe=Sim) = 0.6
# P(DorCabeca=Sim | Gripe=Não) = 0.2
cpd_dorcabeca = TabularCPD(
    variable='DorCabeca',
    variable_card=2,
    values=[[0.8, 0.4],   # P(DorCabeca=Não | Gripe)
            [0.2, 0.6]],  # P(DorCabeca=Sim | Gripe)
    evidence=['Gripe'],
    evidence_card=[2]
)

print("CPTs definidas!")


# Passo 3: Adicionar as CPTs ao modelo
modelo_gripe.add_cpds(cpd_gripe, cpd_febre, cpd_tosse, cpd_dorcabeca)

# Verificar se o modelo está correto
print("Validação do modelo:", modelo_gripe.check_model())

# Visualizar uma CPT
print("\nCPT da Gripe:")
print(cpd_gripe)
print("\nCPT da Febre:")
print(cpd_febre)
print("\nCPT da Tosse:")
print(cpd_tosse)
print("\nCPT da Dor de Cabeça:")
print(cpd_dorcabeca)


visualizar_rede(modelo_gripe, "Rede Bayesiana: Diagnóstico de Gripe")


## inferência probabilistica
# Criar objeto de inferência
inferencia_gripe = VariableElimination(modelo_gripe)

# Consulta 1: Qual a probabilidade de ter gripe SEM nenhuma evidência?
resultado = inferencia_gripe.query(variables=['Gripe'])
print("P(Gripe):")
print(resultado)
print()


# Consulta 2: Uma pessoa tem febre. Qual a probabilidade de ter gripe?
resultado = inferencia_gripe.query(
    variables=['Gripe'],
    evidence={'Febre': 1}  # 1 = Sim, 0 = Não
)
print("P(Gripe | Febre=Sim):")
print(resultado)
print()


# Consulta 3: Uma pessoa tem febre E tosse. Qual a probabilidade de ter gripe?
resultado = inferencia_gripe.query(
    variables=['Gripe'],
    evidence={'Febre': 1, 'Tosse': 1}
)
print("P(Gripe | Febre=Sim, Tosse=Sim):")
print(resultado)
print()


# Consulta 4: Uma pessoa tem TODOS os sintomas. Qual a probabilidade de ter gripe?
resultado = inferencia_gripe.query(
    variables=['Gripe'],
    evidence={'Febre': 1, 'Tosse': 1, 'DorCabeca': 1}
)
print("P(Gripe | Febre=Sim, Tosse=Sim, DorCabeca=Sim):")
print(resultado)
print()

print("\n📊 Análise:")
print("Observe como a probabilidade de gripe aumenta à medida que")
print("observamos mais sintomas (evidências)!")