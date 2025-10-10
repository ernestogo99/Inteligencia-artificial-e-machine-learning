import numpy as np 
import matplotlib.pyplot as plt
from algoritms.ordinary_ls import OrdinaryLS
from algoritms.metrics import Metrics
from algoritms.linearregressiongd import LinearRegressionGD
from algoritms.linearregressionsgd import LinearRegressionSGD

def main():
#questão1 a)
    artificial_data=np.genfromtxt('data/artificial1d.csv',delimiter=',')
    x=artificial_data[:,[0]]
    y=artificial_data[:,[1]]
    ordinary_ls=OrdinaryLS()
    metrics=Metrics()
    ordinary_ls.fit(x,y)
    print('parâmetros')
    params=ordinary_ls.get_params()
    print(params)
    print('MSE')
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    predicts=ordinary_ls.predict(x_line)
    print(metrics.mse(y_true=y,y_pred=predicts))
    print('predict')
    print(predicts)


    plt.figure(figsize=(10,5))
    plt.title('Gráfico do dataset artificial1d')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.scatter(x,y)
    plt.plot(x_line,predicts)
    plt.show()

def item_b():
    artificial_data=np.genfromtxt('data/artificial1d.csv',delimiter=',')
    x=artificial_data[:,[0]]
    y=artificial_data[:,[1]]
    linear_regression_gd=LinearRegressionGD()
    metrics=Metrics()
    linear_regression_gd.fit(x,y)
    print("parâmetros")
    print(linear_regression_gd.get_params())
    print('MSE')
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    predicts=linear_regression_gd.predict(x_line)
    print(metrics.mse(y_true=y,y_pred=predicts))
    print('predict')
    print(predicts)

    plt.figure(figsize=(10,5))
    plt.title('Gráfico do dataset artificial1d')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.scatter(x,y)
    plt.plot(x_line,predicts)
    plt.show()

def item_c():
    artificial_data=np.genfromtxt('data/artificial1d.csv',delimiter=',')
    x=artificial_data[:,[0]]
    y=artificial_data[:,[1]]
    linear_regression_sgd=LinearRegressionSGD()
    metrics=Metrics()
    linear_regression_sgd.fit_stochastic_descent(x,y)
    print("parâmetros")
    print(linear_regression_sgd.get_params())
    print('MSE')
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    predicts=linear_regression_sgd.predict(x_line)
    print(metrics.mse(y_true=y,y_pred=predicts))
    print('predict')
    print(predicts)

    plt.figure(figsize=(10,5))
    plt.title('Gráfico do dataset artificial1d')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.scatter(x,y)
    plt.plot(x_line,predicts)
    plt.show()








if __name__ == "__main__":
    item_c()