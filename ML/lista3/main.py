import pandas as pd
from algoritms.kfold import kfold
from algoritms.knn import KNNClassifier
from sklearn.tree import   DecisionTreeClassifier

def main():
    dataset_kc2=pd.read_csv('data/kc2.csv')
    dataset_kc2=dataset_kc2.sample(frac=1).reset_index(drop=True)

    x_kc2=dataset_kc2.iloc[:,:-1]
    y_kc2=dataset_kc2.iloc[:,-1]

    knn_1=KNNClassifier(k=1)
    knn_2=KNNClassifier(k=1, metric='mahalanobis')
    knn_3=KNNClassifier(k=5)
    knn_4=KNNClassifier(k=5,metric='mahalanobis')

    tree_1=DecisionTreeClassifier(criterion='gini')
    tree_2=DecisionTreeClassifier(criterion='entropy')
    

    knn_k_1=[knn_1,knn_2]
    knn_k_5=[knn_3,knn_4]

    trees=[tree_1,tree_2]

    for model in knn_k_1:
        print('----='*10)
        print(f'\n Metricas para KNN com K={model.k} e metrica = {model.metric}')
        kfold(model,x_kc2,y_kc2)

    for model in knn_k_5:
        print('----='*10)
        print(f'\n Metricas para KNN com K={model.k} e metrica = {model.metric}')
        kfold(model,x_kc2,y_kc2)

    for model in trees:
        print('----='*10)
        print(f'\n Métricas para árvores com critério de {model.criterion}')
        kfold(model,x_kc2,y_kc2)


if __name__ =='__main__':
    main()

