import pandas as pd
from algorithms.logisticregressiongd import LogisticRegressionGD,LogisticRegressionSoftmaxGD
from algorithms.kfold import kfold
from algorithms.gaussian import GaussianDiscriminantAnalysis,GaussianNaiveBayes
def main():
    dataset_cancer=pd.read_csv('data/breastcancer.csv')
    dataset_cancer=dataset_cancer.sample(frac=1).reset_index(drop=True)
    x_breast_cancer = dataset_cancer.iloc[:,:-1]
    y_breast_cancer = dataset_cancer.iloc[:,-1]

    regressao_logistica=LogisticRegressionGD(alpha=0.1,iterations=500)
    #kfold(model=regressao_logistica,features_data=x_breast_cancer,label_data=y_breast_cancer)
    discriminante=GaussianDiscriminantAnalysis()
    kfold(model=discriminante,features_data=x_breast_cancer,label_data=y_breast_cancer)    

    naive_bayes=GaussianNaiveBayes()
    kfold(model=naive_bayes,features_data=x_breast_cancer,label_data=y_breast_cancer)


def q2():
    dataset_vehicles=pd.read_csv('data/vehicle.csv')
    dataset_vehicles=dataset_vehicles.sample(frac=1).reset_index(drop=True)
    x_vehicles=dataset_vehicles.iloc[:,:-1]
    y_vehicles=dataset_vehicles.iloc[:,-1]
    softmax=LogisticRegressionSoftmaxGD(alpha=0.9)
    kfold(model=softmax,features_data=x_vehicles,label_data=y_vehicles)
    discriminante=GaussianDiscriminantAnalysis()
    kfold(model=discriminante,features_data=x_vehicles,label_data=y_vehicles)    

    naive_bayes=GaussianNaiveBayes()
    kfold(model=naive_bayes,features_data=x_vehicles,label_data=y_vehicles)



if __name__ =='__main__':
    q2()