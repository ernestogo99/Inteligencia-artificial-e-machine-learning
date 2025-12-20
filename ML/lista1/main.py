import numpy as np 
import matplotlib.pyplot as plt
from algoritms.ordinary_ls import OrdinaryLS
from algoritms.metrics import Metrics
from algoritms.linearregressiongd import LinearRegressionGD
from algoritms.linearregressionsgd import LinearRegressionSGD
from algoritms.train import train_test_split
from algoritms.normalize import StandardScaler,MinMaxScaler
from algoritms.ridgeregression import RidgeRegression

def main():
#questão1 a)
    artificial_data=np.genfromtxt('data/artificial1d.csv',delimiter=',')
    x=artificial_data[:,[0]]
    y=artificial_data[:,[1]]
    print(len(x))
    print(len(y))
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


def questao2():
    #item a
    california_data=np.genfromtxt('data/california.csv',delimiter=',')
    x=california_data[:,[0]]
    y=california_data[:,[1]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
    print(f"Tamanho treino: {len(x_train)} | teste: {len(x_test)}")

    x_scaler=MinMaxScaler()
    y_scaler=StandardScaler()

    x_train_norm=x_scaler.fit_transform(x_train)
    x_test_norm=x_scaler.transform(x_test)

    y_train_norm=y_scaler.fit_transform(y_train)
    y_test_norm=y_scaler.transform(y_test)


    metrics=Metrics()
    degrees, train_rmse, test_rmse = [], [], []


    for degree in range(1,14):
        x_train_poly=metrics.polynomial_features(x_train_norm,degree=degree)
        x_test_poly=metrics.polynomial_features(x_test_norm,degree=degree)


        model=OrdinaryLS()
        model.fit(x_train_poly,y_train_norm)

        y_train_pred_norm = model.predict(x_train_poly)
        y_test_pred_norm = model.predict(x_test_poly)

        y_train_pred = y_scaler.inverse_transform(y_train_pred_norm)
        y_test_pred = y_scaler.inverse_transform(y_test_pred_norm)

        rmse_train = metrics.rmse(y_train, y_train_pred)
        rmse_test = metrics.rmse(y_test, y_test_pred)

        degrees.append(degree)
        train_rmse.append(rmse_train)
        test_rmse.append(rmse_test)

        print(f"Grau {degree}: RMSE treino={rmse_train:.4f}, RMSE teste={rmse_test:.4f}")


    plt.plot(degrees, train_rmse, label="Treino")
    plt.plot(degrees, test_rmse, label="Teste")
    plt.xlabel("Grau do polinômio")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Regressão Polinomial")
    plt.show()


def questao_2_d():
    california_data=np.genfromtxt('data/california.csv',delimiter=',')
    x=california_data[:,[0]]
    y=california_data[:,[1]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
    print(f"Tamanho treino: {len(x_train)} | teste: {len(x_test)}")

    x_scaler=MinMaxScaler()
    y_scaler=StandardScaler()

    x_train_norm=x_scaler.fit_transform(x_train)
    x_test_norm=x_scaler.transform(x_test)

    y_train_norm=y_scaler.fit_transform(y_train)
    y_test_norm=y_scaler.transform(y_test)


    metrics=Metrics()
    degrees, train_rmse, test_rmse = [], [], []


    for degree in range(1,14):
        x_train_poly=metrics.polynomial_features(x_train_norm,degree=degree)
        x_test_poly=metrics.polynomial_features(x_test_norm,degree=degree)


        model=RidgeRegression()
        model.fit(x_train_poly,y_train_norm)

        y_train_pred_norm = model.predict(x_train_poly)
        y_test_pred_norm = model.predict(x_test_poly)

        y_train_pred = y_scaler.inverse_transform(y_train_pred_norm)
        y_test_pred = y_scaler.inverse_transform(y_test_pred_norm)

        rmse_train = metrics.rmse(y_train, y_train_pred)
        rmse_test = metrics.rmse(y_test, y_test_pred)

        degrees.append(degree)
        train_rmse.append(rmse_train)
        test_rmse.append(rmse_test)

        print(f"Grau {degree}: RMSE treino={rmse_train:.4f}, RMSE teste={rmse_test:.4f}")


    plt.plot(degrees, train_rmse, label="Treino")
    plt.plot(degrees, test_rmse, label="Teste")
    plt.xlabel("Grau do polinômio")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Regressão Polinomial")
    plt.show()





if __name__ == "__main__":
    main()