import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from utils import visualizar_rede


plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


modelo_filmes=DiscreteBayesianNetwork([
    ('IdadeUsuario','GostaCiencia'),
    ('GostaCiencia','GostaAcao'),
    ('GostaCiencia','AssistiuFilme'),
    ('GostaAcao','AssistiuFilme'),
    ('AssistiuFilme', 'AvaliacaoPositiva')
])

#Definir as cpts

#P(Jovem) = 0.4
#P(Adulto) = 0.6
cpd_idade = TabularCPD(
    variable='IdadeUsuario',
    variable_card=2,
    values=[
        [0.4],  
        [0.6]   
    ]
)
# P(GostaCiencia=Sim | Jovem) = 0.6  => P(Não | Jovem)=0.4
# P(GostaCiencia=Sim | Adulto) = 0.7 => P(Não | Adulto)=0.3
cpd_gosta_ciencia=TabularCPD(
    variable='GostaCiencia',
    variable_card=2,
    values=[[0.4,0.3],
            [0.6,0.7]]
    ,evidence=['IdadeUsuario'],
    evidence_card=[2]

)

# P(GostaAcao=Sim | Jovem) = 0.8  => P(Não | Jovem)=0.2
# P(GostaAcao=Sim | Adulto) = 0.5 => P(Não | Adulto)=0.5
cpd_gosta_acao=TabularCPD(
    variable='GostaAcao',
    variable_card=2,
    values=[[0.2,0.5],[0.8,0.5]],
    evidence=['GostaCiencia'],
    evidence_card=[2]
)

# AvaliacaoPositiva | AssistiuFilme
# P(Avaliacao=Positiva | Assistiu=Não) = 0.0  => P(NãoPositiva|Não)=1.0
# P(Avaliacao=Positiva | Assistiu=Sim) = 0.75 => P(NãoPositiva|Sim)=0.25
cpd_avaliacao = TabularCPD(
    variable='AvaliacaoPositiva',
    variable_card=2,
    values=[
        [1.0, 0.25],  
        [0.0, 0.75]   
    ],
    evidence=['AssistiuFilme'],
    evidence_card=[2]
)


#P(Assistiu=Sim | GostaCiencia=Não, GostaAcao=Não) = 0.1
#P(Assistiu=Sim | GostaCiencia=Não, GostaAcao=Sim) = 0.3
#P(Assistiu=Sim | GostaCiencia=Sim, GostaAcao=Não) = 0.4
#P(Assistiu=Sim | GostaCiencia=Sim, GostaAcao=Sim) = 0.8
cpd_assistiu_filme=TabularCPD(
    variable='AssistiuFilme',
    variable_card=2,
    values=[[0.9,0.7,0.6,0.2],[0.1,0.3,0.4,0.8]],
    evidence=['GostaCiencia', 'GostaAcao'],
    evidence_card=[2, 2]
)

modelo_filmes.add_cpds(cpd_assistiu_filme,cpd_avaliacao,cpd_gosta_acao,cpd_gosta_ciencia,cpd_idade)

# Validar
print("Modelo válido:", modelo_filmes.check_model())

# Visualizar uma CPT
print("\nCPT da Idade:")
print(cpd_idade)
print("\nCPT do Assistiu Filme:")
print(cpd_assistiu_filme)
print("\nCPT da Avaliação:")
print(cpd_avaliacao)
print("\nCPT do Gosta Ação:")
print(cpd_gosta_acao)
print("\nCPT do Gosta Ciência:")
print(cpd_gosta_ciencia)

visualizar_rede(modelo_filmes, "Sistema de Recomendação de Filmes")


inferencia_filmes = VariableElimination(modelo_filmes)

# Pergunta 1: P(AssistiuFilme)
resultado =inferencia_filmes.query(variables=['AssistiuFilme'])
print('Resultado da pergunta 1')
print(resultado)
# Pergunta 2: P(AssistiuFilme | IdadeUsuario=Jovem)
resultado=inferencia_filmes.query(variables=['AssistiuFilme'],evidence={'IdadeUsuario':0})
print('P(AssistiuFilme | IdadeUsuario=Jovem)')
print(resultado)



# Pergunta 3: Se um usuário gosta de ciência E gosta de ação, qual a probabilidade dele dar avaliação positiva?
#Pergunta 3: P(AvaliacaoPositiva | GostaCiencia=Sim, GostaAcao=Sim)
resultado=inferencia_filmes.query(variables=['AvaliacaoPositiva'],evidence={'GostaCiencia':1,'GostaAcao':1})
print('P(AvaliacaoPositiva | GostaCiencia=Sim, GostaAcao=Sim)')
print(resultado)
# Pergunta 4: Se um usuário deu avaliação positiva, qual a probabilidade dele gostar de ciência? (inferência reversa!)

# Pergunta 4: P(GostaCiencia | AvaliacaoPositiva=Sim)
resultado = inferencia_filmes.query(
    variables=['GostaCiencia'],
    evidence={'AvaliacaoPositiva': 1}
)
print('P(GostaCiencia | AvaliacaoPositiva=Sim):')
print(resultado)