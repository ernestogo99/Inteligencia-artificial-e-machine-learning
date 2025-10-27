import numpy as np
from .normalize import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from .logisticregressiongd import LogisticRegressionGD,LogisticRegressionSoftmaxGD


def global_accuracy_function(true_values: np.array,predicted: np.array):

  if len(true_values) != len(predicted):
    return 'erro de tamanho'

  num_erros = 0
  num_acertos = 0

  for i, prediction in enumerate(predicted):
    if true_values[i] == prediction:
      num_acertos +=1
    else:
      num_erros +=1
  return num_acertos/(num_acertos + num_erros)


def kfold(model ,features_data, label_data, num_folds = 10):
    """
    model: modelo de ML
    features data: X dos dados que serao usados no modelo
    label data: y dos dados que serao usados no modelo
    num_folds: numero de folds
    """

 
    tamanho_particoes =  int(np.ceil(len(features_data)/num_folds))


    if tamanho_particoes*num_folds > len(features_data):
        ultima = len(features_data) - tamanho_particoes*(num_folds-1)
    
    inicio = 0
    fim= tamanho_particoes

    acuracias_globais = np.array([])
    acuracias_por_classe = {}
    curvas = []
    for i in range(num_folds):
      x_teste = features_data.iloc[inicio:fim]
      y_teste = label_data[inicio:fim]

      x_train_fold  = features_data.drop(features_data.index[inicio:fim])
      y_train_fold = label_data.drop(label_data.index[inicio:fim])
      scaler = StandardScaler()
      x_train_fold = pd.DataFrame(scaler.fit_transform(x_train_fold), columns=x_train_fold.columns)
      x_teste = pd.DataFrame(scaler.transform(x_teste), columns=x_teste.columns)


      if hasattr(model, 'loss_history'):  
        model.loss_history = []


      model.fit(x_train_fold.values, y_train_fold.values)
      predictions = model.predict(x_teste)

      if isinstance(model, LogisticRegressionGD) or isinstance(model, LogisticRegressionSoftmaxGD ):
        curvas.append(model.loss_history)

   
      
      y_teste = np.array(y_teste)

      for classe in np.unique(y_teste):
        acerto_classe = 0   
        total_classe = 0
        for i, elemento in enumerate(list(y_teste)):
          if elemento == classe:
            total_classe +=1
            if predictions[i] == classe:
               acerto_classe +=1

        acc = acerto_classe/total_classe
        if classe not in acuracias_por_classe:
                acuracias_por_classe[classe] = []
        acuracias_por_classe[classe].append(acc)
      

      acuracia_global = global_accuracy_function(y_teste, predictions)
      acuracias_globais = np.append(acuracias_globais, acuracia_global)


      if i+1 == num_folds -1:
          fim += ultima
      
      else: 
          fim += tamanho_particoes
      inicio +=tamanho_particoes


    if isinstance(model, LogisticRegressionGD) or isinstance(model, LogisticRegressionSoftmaxGD) :
      fig, axes = plt.subplots(2, 5, figsize=(20, 10))
      axes = axes.ravel()

      for i, curva in enumerate(curvas):
          norma = np.linalg.norm(curva)
          erros_normalizados = curva / norma
          axes[i].plot(erros_normalizados)
          axes[i].set_title(f'Fold {i+1}')
          axes[i].set_xlabel('Iteração')
          axes[i].set_ylabel('Log Loss')

      plt.tight_layout()
      plt.show()

    for classe, acuracias in acuracias_por_classe.items():
        media_classe = np.mean(acuracias)
        print(f'A acurácia para a classe {classe} foi igual a {media_classe}')

    media = acuracias_globais.mean()    
    print(f'a acuracia global media do modelo foi igual a: {media}')