from sklearn import datasets  # 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics



def dataGet():
    # 获取波士顿房价数据集
    bostonHousing = datasets.load_boston()
    # 获取特征指标数据集
    bostonData = bostonHousing.data
    # 获取标签集
    bostonTar = bostonHousing.target
    return bostonData, bostonTar


def dataSplit(bostonData, bostonTar):
    X_train, X_test, y_train, y_test = train_test_split(bostonData, bostonTar, test_size=5, random_state=0)
    return X_train, X_test, y_train, y_test


def linearReMain(X_train, X_test, y_train, y_test):
    liRe = LinearRegression()
    liRe.fit(X_train, y_train)
    BostonPre = liRe.predict(X_test)
    # 获取回归系数和阈值
    coef = liRe.coef_
    intercept = liRe.intercept_
    BostonMSE0 = metrics.mean_squared_error(y_test, BostonPre)
    print("多项式线性回归模型的MSE值为：%f" % BostonMSE0)
    BostonRMSE0 = np.sqrt(BostonMSE0)
    print("多项式线性回归模型的RMSE值为： %f" % BostonRMSE0)
    return BostonPre,coef,intercept


def MinMaxMain(X_train, X_test, y_train, y_test):
    mm = MinMaxScaler()
    X_train0 = mm.fit_transform(X_train)
    X_test0 = mm.transform(X_test)
    liRe = LinearRegression()
    liRe.fit(X_train0, y_train)
    BostonPre1 = liRe.predict(X_test0)

    BostonMSE1 = metrics.mean_squared_error(y_test, BostonPre1)
    print("归一化后，线性回归模型的MSE值为：%f" % BostonMSE1)
    BostonRMSE1 = np.sqrt(BostonMSE1)
    print("归一化后，线性回归模型的RMSE值为： %f" % BostonRMSE1)


if __name__ == '__main__':
    # 获取数据集
    bostonData, bostonTar = dataGet()
    # 数据集划分
    X_train, X_test, y_train, y_test = dataSplit(bostonData, bostonTar)
    # 调用线性回归接口进行预测
    BostonPre, coef, intercept = linearReMain(X_train, X_test, y_train, y_test)
    # print(BostonPre)
    # print(coef)
    # print(intercept)
    MinMaxMain(X_train, X_test, y_train, y_test)