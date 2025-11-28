import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from algoritms.params import kfold_com_grid_search, kfold_padrao
from models.neural_network import MLPClassifier, MLPRegressor


def main():

    data_1 = pd.read_csv('data/concrete.csv')
    data_1 = data_1.sample(frac =1).reset_index(drop= True)

    train, test = train_test_split(data_1, test_size= 0.2)

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:,-1].astype('int')
    X_teste = test.iloc[:, :-1]
    y_teste = test.iloc[:, -1].astype('int')
    y_teste = np.array(y_teste)



    hyperparameter_list = [
    ("learning_rate", [0.01, 0.1]),
    ('momentum', [0.8, 0.9]),
    ('batch_size', [10, 20, 40]),
    ('hidden_layer_sizes', [(20,), (50,), (100,)]),
    ('max_iter', [1000]),
    ("activation", ['relu', 'tanh']),
    ("solver", ['sgd'])
]

    bests_hyperparameters, rmse_best_model = kfold_com_grid_search(model = MLPRegressor(weight_decay=1e-4), features_data= X_train, label_data= y_train , hyperparameters_list= hyperparameter_list, num_folds= 5)
    print(rmse_best_model)
    print(bests_hyperparameters)
    model = MLPRegressor()
    kfold_padrao(model=model, hyperparameters= bests_hyperparameters, features_data= X_train, label_data= y_train, num_folds= 5)
    y_teste = np.array(y_teste)

    model = MLPRegressor(**dict(bests_hyperparameters))
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_normalizado = pd.DataFrame(scaler_x.fit_transform(X_train), columns= X_train.columns)
    X_teste_normalizado = pd.DataFrame(scaler_x.transform(X_teste), columns= X_teste.columns)
    y_train_normalizado = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    model.fit(X_train_normalizado, y_train_normalizado)
    predictions = model.predict(X_teste_normalizado)
    predictions_desnormalizadas = scaler_y.inverse_transform(predictions.reshape(-1,1)).ravel()
    rmse = np.sqrt(np.mean((predictions_desnormalizadas - y_teste)**2 )) 
    print(f'rmse do modelo eh igual a {rmse}')
    mae = np.mean(np.abs(predictions_desnormalizadas - y_teste))
    print(f'mean absolute error do modelo eh {mae}')
    mre = np.mean(np.abs( (predictions_desnormalizadas -y_teste)/y_teste))
    print(f'mean relative error do modelo eh {mre}')


def q2():
    data_2 = pd.read_csv('data/vowel.csv')
    data_2 = data_2.sample(frac =1).reset_index(drop= True)
    train, test = train_test_split(data_2, test_size= 0.2)

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:,-1].astype('int')
    X_teste = test.iloc[:, :-1]
    y_teste = test.iloc[:, -1].astype('int')
    hyperparameter_list = [
    ("learning_rate", [0.01, 0.1]),
    ('momentum', [0.8, 0.9]),
    ('batch_size', [10, 20, 40]),
    ('hidden_layer_sizes', [(20,), (50,), (100,)]),
    ('max_iter', [1000]),
    ("activation", ['relu', 'tanh']), 
    ("solver", ['sgd'])
]

    bests_hyperparams, acuracia_melhor_modelo = kfold_com_grid_search(model = MLPClassifier(weight_decay=1e-4), features_data= X_train, label_data= y_train , hyperparameters_list= hyperparameter_list, num_folds= 5)
    print(bests_hyperparams)
    print(acuracia_melhor_modelo)
    kfold_padrao(model = MLPClassifier(), hyperparameters= bests_hyperparams, features_data= X_train, label_data= y_train, num_folds= 5)
    model = MLPClassifier(**dict(bests_hyperparams))

    scaler_x = StandardScaler()
    X_train_normalizado = pd.DataFrame(scaler_x.fit_transform(X_train), columns= X_train.columns)
    X_teste_normalizado = pd.DataFrame(scaler_x.transform(X_teste), columns= X_teste.columns)
    model.fit(X_train_normalizado, y_train)
    predictions = model.predict(X_teste_normalizado)
    acuracia = np.mean(y_teste == predictions)
    print(f'a acuracia do modelo eh {acuracia}')

if __name__ =="__main__":
    q2()