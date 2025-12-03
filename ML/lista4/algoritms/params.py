from models.neural_network import MLPClassifier,MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def test_hyperparameter_rec(
    hyperparameter_list, model, x_train, x_teste, y_train, y_teste,
    scaler_y=None, hyperparameter=0, hyperparameters_atuais=[]
):
    """
    Calcula recursivamente as métricas para todas as combinações de hiperparâmetros,
    """



    if len(hyperparameters_atuais) == len(hyperparameter_list):

     

        try:
            if isinstance(model, MLPRegressor):
                modelo = MLPRegressor(**dict(hyperparameters_atuais))
            else:
                modelo = MLPClassifier(**dict(hyperparameters_atuais))

            modelo.fit(x_train, y_train)

            predictions = modelo.predict(x_teste)

        except OverflowError:
            print("   ⚠️  overflow encontrado! pulando combinação.")
            return [[np.inf, hyperparameters_atuais]]

        except Exception as e:
            print(f"   ⚠️  erro inesperado: {e}")
            return [[np.inf, hyperparameters_atuais]]


        if isinstance(model, MLPRegressor):
            predictions_desnormalized = scaler_y.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).ravel()
            rmse = np.sqrt(np.mean((y_teste - predictions_desnormalized) ** 2))

           
            return [[rmse, hyperparameters_atuais]]

        else:
            accuracy = np.mean(predictions == y_teste)

            return [[accuracy, hyperparameters_atuais]]

   
    results = []

    nome_param, valores = hyperparameter_list[hyperparameter]

    for valor in valores:

        hyperparameters_temp = hyperparameters_atuais + [(nome_param, valor)]

        res = test_hyperparameter_rec(
            hyperparameter_list, model,
            x_train, x_teste, y_train, y_teste,
            scaler_y,
            hyperparameter=hyperparameter + 1,
            hyperparameters_atuais=hyperparameters_temp
        )

        results.extend(res)

    return results


def kfold_com_grid_search(model, features_data, label_data, hyperparameters_list, num_folds):
    """
    essa funcao calcula as avaliacoes para todas as combinacoes possiveis, variando os conjuntos de treino e validacao
    ao final ela retorna o modelo com o melhor conjunto de hiperparametros e o valor da metrica de avaliacao obtida
    """
    tamanho_particoes = int(np.ceil(len(features_data) / num_folds))

    lista_avaliacoes = []

    for i in range(num_folds):
        init = i * tamanho_particoes
        fim = min((i + 1) * tamanho_particoes, len(features_data))

        x_teste = features_data.iloc[init:fim]
        y_teste = label_data.iloc[init:fim]

        x_train_fold = features_data.drop(features_data.index[init:fim])
        y_train_fold = label_data.drop(label_data.index[init:fim])

        scaler_x = StandardScaler()
        x_train_fold_normalizado = pd.DataFrame(scaler_x.fit_transform(x_train_fold), columns=x_train_fold.columns, index=x_train_fold.index)
        x_train_fold = x_train_fold_normalizado.copy()
        x_teste_normalizado = pd.DataFrame(scaler_x.transform(x_teste), columns=x_teste.columns, index=x_teste.index)
        x_teste = x_teste_normalizado.copy()

        if isinstance(model, MLPRegressor):
            scaler_y = StandardScaler()
            y_train_fold_normalizado = scaler_y.fit_transform(y_train_fold.values.reshape(-1, 1))
            y_train_fold = y_train_fold_normalizado.ravel()
            avaliacoes = test_hyperparameter_rec(hyperparameter_list=hyperparameters_list, model=model, x_train=x_train_fold, x_teste=x_teste, y_train=y_train_fold, y_teste=y_teste, scaler_y= scaler_y)
        else:
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

    if isinstance(model, MLPRegressor):
        melhor_comb = min(medias_comb, key=medias_comb.get)  
    else:
        melhor_comb = max(medias_comb, key=medias_comb.get)  

    return melhor_comb, medias_comb[melhor_comb]


def kfold_padrao(model, hyperparameters, features_data, label_data, num_folds):
    """
    essa funcao faz o kfold simples e ao final plota o grafico das diferentes particoes
    """
    tamanho_particoes = int(np.ceil(len(features_data) / num_folds))
    curvas = []
    rmses = []
    for i in range(num_folds):
        if isinstance(model, MLPRegressor):
            model = MLPRegressor(**dict(hyperparameters))
        else:
            model = MLPClassifier(**dict(hyperparameters))
        init = i * tamanho_particoes
        end = min((i + 1) * tamanho_particoes, len(features_data))

        x_teste = features_data.iloc[init:end]
        y_teste = label_data.iloc[init:end]

        x_train_fold = features_data.drop(features_data.index[init:end])
        y_train_fold = label_data.drop(label_data.index[init:end])

        scaler_x = StandardScaler()
        x_train_fold_normalized = pd.DataFrame(scaler_x.fit_transform(x_train_fold), columns=x_train_fold.columns, index=x_train_fold.index)
        x_train_fold = x_train_fold_normalized.copy()
        x_teste_normalized = pd.DataFrame(scaler_x.transform(x_teste), columns=x_teste.columns, index=x_teste.index)
        x_teste = x_teste_normalized.copy()

        if isinstance(model, MLPRegressor):
            scaler_y = StandardScaler()
            y_train_fold_normalized = scaler_y.fit_transform(y_train_fold.values.reshape(-1, 1))
            y_train_fold = y_train_fold_normalized.ravel()
            y_teste_normalized = scaler_y.transform(y_teste.values.reshape(-1, 1))
            y_teste = y_teste_normalized.ravel()
        
        model.fit(x_train_fold_normalized, y_train_fold)
        curvas.append(model.loss_curve_)

    fig, axes  = plt.subplots(1, 5, figsize=(20,3))
    axes = axes.ravel()
    for i, curva in enumerate(curvas):
        axes[i].plot(curva)
        axes[i].set_title(f'Fold {i+1}')
        axes[i].set_xlabel('Iterações')
        axes[i].set_ylabel('funcao custo')
    plt.tight_layout()

    plt.show()    