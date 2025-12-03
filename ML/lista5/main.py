import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algoritms.search import kfold_com_grid_search, plotar_roc_precrecall
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main():
    data = pd.read_csv('data/californiabin.csv')
    data = data.sample(frac= 1).reset_index(drop= True)

    train, test = train_test_split(data, test_size= 0.2)

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:,-1].astype('int')
    X_teste = test.iloc[:, :-1]
    y_teste = test.iloc[:, -1].astype('int')
    y_teste = np.array(y_teste)

    scaler_x = StandardScaler()
    X_train_normalizado = pd.DataFrame(scaler_x.fit_transform(X_train), columns= X_train.columns)
    X_teste_normalizado = pd.DataFrame(scaler_x.transform(X_teste), columns= X_teste.columns)

    cs = []
    for i in range(-5, 16):
        cs.append(2**i)
    h1 = ("C", cs)

    gammas = []
    for i in range(-15,4):
        gammas.append(2**i)

    h2 = ("gamma", gammas)

    svc_hyperparameter_list = [h1, h2]

    svm_best_hyperparameters, svm_best_accuracy = kfold_com_grid_search(model= SVC(), features_data= X_train, label_data= y_train, hyperparameters_list= svc_hyperparameter_list, num_folds= 10)
    print(f'melhor precisão do svm: {svm_best_accuracy}')
    print(f'melhores hiperparâmetros do svm: {svm_best_hyperparameters}')

    svm_best_hyperparameters = dict(svm_best_hyperparameters)
    svm_best_hyperparameters['probability'] = True

    best_svm = SVC(**svm_best_hyperparameters)
    best_svm.fit(X_train_normalizado, y_train)

    predictions_svm = best_svm.predict(X_teste_normalizado)
    svm_predicions_probabilities = best_svm.predict_proba(X_teste)
    svm_class_1 = svm_predicions_probabilities[:,1]

    acuracia_svm = np.mean(predictions_svm == y_teste)
    print(f'acurácia do svm: {acuracia_svm}')
    revocacao_svm = recall_score(y_teste, predictions_svm)
    print(f'revocação do svm: {revocacao_svm}')
    precisao_svm = precision_score(y_teste, predictions_svm)
    print(f'precisão do svm: {precisao_svm}')
    f1_svm = f1_score(y_teste, predictions_svm)
    print(f'f1 score do svm {f1_svm}')
    y_scores_svm = best_svm.predict_proba(X_teste_normalizado)[:,1]
    plotar_roc_precrecall(y_teste= y_teste, y_score= y_scores_svm)

    number_classifiers = []
    for i in range(20):
        number_classifiers.append((i+1)*10)
    h1_rf = ('n_estimators', number_classifiers)
    h2_rf = ('max_depth', [4,6,8,10, None])

    rf_hyperparameter_list = [h1_rf, h2_rf]

    rf_best_hyperparameters, rf_best_acuracia = kfold_com_grid_search(model= RandomForestClassifier(), features_data= X_train, label_data= y_train, hyperparameters_list= rf_hyperparameter_list, num_folds= 10)
    print(f'Melhores parâmetros do random forest {rf_best_hyperparameters}')
    print(f'Melhor acurácia do random forest {rf_best_acuracia} ')

    rf_foda = RandomForestClassifier(**dict(rf_best_hyperparameters))
    rf_foda.fit(X_train_normalizado, y_train)

    predictions_rf = rf_foda.predict(X_teste_normalizado)
    predicions_probabilities_rf = rf_foda.predict_proba(X_teste)
    rf_class_1 = predicions_probabilities_rf[:, 1]

    acuracia_rf = np.mean(predictions_rf == y_teste)
    print(f'acurácia do random forest: {acuracia_rf}')
    revocacao_rf = recall_score(y_teste, predictions_rf)
    print(f'Revocação do random forest: {revocacao_rf}')
    precisao_rf = precision_score(y_teste, predictions_rf)
    print(f'Precisão do random forest:{precisao_rf}')
    f1_rf = f1_score(y_teste, predictions_rf)
    print(f'f1 score do random forest: {f1_rf}')
    y_scores_rf = rf_foda.predict_proba(X_teste_normalizado)[:, -1]
    plotar_roc_precrecall(y_teste= y_teste, y_score= y_scores_rf)


if __name__ =='__main__':
    main()