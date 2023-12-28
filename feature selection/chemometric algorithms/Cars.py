"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm, rcParams
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import copy

# ref: https://blog.csdn.net/qq2512446791

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # PLS-DA判别分析中转换y
            y_train = pd.get_dummies(y_train)
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            # 将预测结果即类别矩阵转换为数值标签
            y_predict = np.array([np.argmax(i) for i in y_predict])
            RMSE.append(accuracy_score(y_test, y_predict))

        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmax(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # PLS-DA判别分析中转换y
        y_train = pd.get_dummies(y_train)
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        # 将预测结果即类别矩阵转换为数值标签
        y_predict = np.array([np.argmax(i) for i in y_predict])
        RMSE.append(accuracy_score(y_test, y_predict))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

def CARS_Cloud(X, y, N=50, f=20, cv=10):
    p = 0.8
    m, n = X.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    Coeff = []
    WaveNum =[]
    RMSECV = []
    r = []
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice\
            (np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        # xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1,-1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        cb = coeff[:wave_num]
        if wnum > 0:
            cb = np.vstack((cb, np.full((wnum, 1), -1)))
        if len(Coeff) == 0:
            Coeff = copy.deepcopy(cb)
        else:
            Coeff = np.hstack((Coeff, cb))
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV.append(Cross_Validation(xcal, ycal, rindex+1, cv))
        # print('RMSECV', RMSECV)
        CoeffData = Coeff.T

    WAVE = []
    COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        cd = CoeffData[i, :]
        WD = np.ones((len(wd)))
        CO = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                CO[j] = 0
            else:
                WD[j] = wd[ind[0]]
                CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        if len(COEFF) == 0:
            COEFF = copy.deepcopy(CO)
        else:
            COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

    # MinIndex = np.argmin(RMSECV)
    MinIndex = np.argmax(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]
    print("Cars的筛选变量索引：", OptWave)

    # 准备数据
    data_df = pd.DataFrame(WaveNum)  # 关键1，将ndarray格式转换为DataFrame
    # 更改表的索引
    data_df.columns = ['A']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
    # 将文件写入excel表格中
    writer = pd.ExcelWriter('方法CARS_WaveNum.xlsx')  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1',
                     float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4

    # 准备数据
    data_df = pd.DataFrame(RMSECV)  # 关键1，将ndarray格式转换为DataFrame
    # 更改表的索引
    data_df.columns = ['A']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
    # 将文件写入excel表格中
    writer = pd.ExcelWriter('方法CARS_RMSECV.xlsx')  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1',
                     float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    plt.xlabel('Number of sampling runs', fontsize=fonts)
    plt.ylabel('Number of variables', fontsize=fonts)
    # plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
    print('最佳迭代次数：', str(MinIndex))
    print('np.arange(N)', np.arange(N))
    print('WaveNum', WaveNum)
    plt.plot(np.arange(N), WaveNum)
    plt.show()

    plt.subplot(212)
    plt.xlabel('Number of sampling runs', fontsize=fonts)
    plt.ylabel('Accuracy', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)
    print('np.arange(N)', np.arange(N))
    print('RMSECV', RMSECV)
    plt.show()

    plt.subplot(313)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # y轴是各变量索引，显示在
    plt.ylabel('各变量系数值', fontsize=fonts)
    plt.plot(COEFF)
    plt.vlines(MinIndex, 0, 1e2, colors='r')
    plt.show()

    return OptWave