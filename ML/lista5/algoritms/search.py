from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score,precision_score, f1_score ,precision_recall_curve, roc_curve, auc, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def test_hyperparameter_rec(hyperparameter_list, model, x_train, x_teste, y_train, y_teste, scaler_y = None, hyperparameter=0, hyperparameters_atuais=[]):
    """
    essa funcao calcula recursivamente as metricas de avaliacao para todas as combinacoes de hiperparametros possiveis e retorna elas em formato de lista
    
    """
    
    if len(hyperparameters_atuais) == len(hyperparameter_list):
        if isinstance(model, RandomForestClassifier):
            modelo = RandomForestClassifier(**dict(hyperparameters_atuais))
        else:
            modelo = SVC(**dict(hyperparameters_atuais))

        modelo.fit(x_train, y_train)
        predictions = modelo.predict(x_teste)

        accuracy = np.mean(predictions == y_teste)
        return [[accuracy, hyperparameters_atuais]]  

    resultados = []
    for element in hyperparameter_list[hyperparameter][1]:
        hyperparameters_temp = hyperparameters_atuais + [(hyperparameter_list[hyperparameter][0], element)]
        res = test_hyperparameter_rec(hyperparameter_list, model, x_train, x_teste, y_train, y_teste, scaler_y, hyperparameter=hyperparameter + 1, hyperparameters_atuais=hyperparameters_temp)
        resultados.extend(res)  
    return resultados


def kfold_com_grid_search(model, features_data, label_data, hyperparameters_list, num_folds):
    """
    essa funcao calcula as avaliacoes para todas as combinacoes possiveis, variando os conjuntos de treino e validacao
    ao final ela retorna o modelo com o melhor conjunto de hiperparametros e o valor da metrica de avaliacao obtida
    """
    tamanho_particoes = int(np.ceil(len(features_data) / num_folds))

    lista_avaliacoes = []

    for i in range(num_folds):
        inicio = i * tamanho_particoes
        fim = min((i + 1) * tamanho_particoes, len(features_data))

        x_teste = features_data.iloc[inicio:fim]
        y_teste = label_data.iloc[inicio:fim]

        x_train_fold = features_data.drop(features_data.index[inicio:fim])
        y_train_fold = label_data.drop(label_data.index[inicio:fim])

        scaler_x = StandardScaler()
        x_train_fold_normalizado = pd.DataFrame(scaler_x.fit_transform(x_train_fold), columns=x_train_fold.columns, index=x_train_fold.index)
        x_train_fold = x_train_fold_normalizado.copy()
        x_teste_normalizado = pd.DataFrame(scaler_x.transform(x_teste), columns=x_teste.columns, index=x_teste.index)
        x_teste = x_teste_normalizado.copy()

        avaliacoes = test_hyperparameter_rec(hyperparameter_list=hyperparameters_list, model=model, x_train=x_train_fold, x_teste=x_teste, y_train=y_train_fold, y_teste=y_teste)

        lista_avaliacoes.extend(avaliacoes)

    combinacoes = {}

    for resultado in lista_avaliacoes:
        score, params = resultado
        params_tuple = tuple(params)  
        if params_tuple not in combinacoes:
            combinacoes[params_tuple] = []
        combinacoes[params_tuple].append(score)

    medias_comb = {param: np.mean(scores) for param, scores in combinacoes.items()}

    melhor_comb = max(medias_comb, key=medias_comb.get)  

    return melhor_comb, medias_comb[melhor_comb]



def plotar_roc_precrecall(y_teste, y_score):
    fpr, tpr, _ = roc_curve(y_teste, y_score)
    roc_auc = roc_auc_score(y_teste, y_score)
    precision, recall, _ = precision_recall_curve(y_teste, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()