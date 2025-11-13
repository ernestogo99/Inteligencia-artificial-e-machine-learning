import numpy as np
from .normalize import StandardScaler
import pandas as pd





def accuracy_function(true_values: np.array,predicted: np.array):

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

def revocacao_function(true_values:np.array, predicted: np.array):
  num_true_positives = 0
  num_false_negatives = 0

  for i, prediction in enumerate(predicted):
    if true_values[i] == 1 and prediction == 1:
      num_true_positives +=1
    elif true_values[i] == 1 and prediction == 0:
      num_false_negatives +=1
  
  if num_true_positives+num_false_negatives == 0:
    return 0
  
  return (num_true_positives/(num_true_positives+num_false_negatives))

def precision_function(true_values: np.array, predicted: np.array):
  num_true_positives = 0
  num_false_positives = 0
  
  for i, prediction in enumerate(predicted):
    if true_values[i] == 1 and prediction == 1:
      num_true_positives +=1
    elif true_values[i] == 0 and prediction == 1:
      num_false_positives +=1
  
  if num_true_positives+num_false_positives == 0:
    return 0 
  return (num_true_positives/(num_true_positives + num_false_positives))

def f1_score_function(true_values:np.array, predicted: np.array):
  revocacao = revocacao_function(true_values, predicted)
  precision = precision_function(true_values, predicted)
  if revocacao + precision == 0:
    return 0
  return 2*((revocacao*precision)/(revocacao+precision))


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

      
    f1_score=np.array([])
    acuracia=np.array([])
    revocacao=np.array([])
    precision=np.array([])
    inicio = 0
    fim= tamanho_particoes

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

      y_teste = np.array(y_teste)
      acuracia=np.append(acuracia,accuracy_function(y_teste,predictions))
      revocacao=np.append(revocacao,revocacao_function(y_teste,predictions))
      f1_score=np.append(f1_score,f1_score_function(y_teste,predictions))
      precision=np.append(precision,precision_function(y_teste,predictions))

      if i+1 == num_folds-1:
        fim += ultima
      else:
        fim += tamanho_particoes
      inicio += tamanho_particoes

    acuracia_media = acuracia.mean()
    acuracia_desvio_padrao = np.std(acuracia)
    print(f'Valor medio da acuracia: {acuracia_media} \nDesvio padrao da acuracia: {acuracia_desvio_padrao}')

    revocacao_media = revocacao.mean()
    revocacao_desvio_padrao = np.std(revocacao)
    print(f'\nValor medio da revocacao: {revocacao_media} \nDesvio padrao da revocacao: {revocacao_desvio_padrao}')

    precision_media = precision.mean()
    precision_desvio_padrao = np.std(precision)
    print(f'\nValor medio da precisao: {precision_media} \nDesvio padrao da precisao: {precision_desvio_padrao}')

    f1_score_media = f1_score.mean()
    f1_score_desvio_padrao = np.std(f1_score)

    print(f'\nValor medio do f1_score: {f1_score_media} \nDesvio padrao do f1_score: {f1_score_desvio_padrao}')

  

   
      
    

   