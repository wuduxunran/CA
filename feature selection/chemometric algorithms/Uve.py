"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from numpy.linalg import matrix_rank as rank
import numpy as np

class UVE:
    def __init__(self, x, y, ncomp=1, nrep=500, testSize=0.2):

        '''
        X : 预测变量矩阵
        y ：标签
        ncomp : 结果包含的变量个数
        testSize: PLS中划分的数据集
        return ：波长筛选后的光谱数据
        '''

        self.x = x
        self.y = y
        # The number of latent components should not be larger than any dimension size of independent matrix
        self.ncomp = min([ncomp, rank(x)])
        self.nrep = nrep
        self.testSize = testSize
        self.criteria = None

        self.featureIndex = None
        self.featureR2 = np.full(self.x.shape[1], np.nan)
        self.selFeature = None

    def calcCriteria(self):
        PLSCoef = np.zeros((self.nrep, self.x.shape[1]))
        ss = ShuffleSplit(n_splits=self.nrep, test_size=self.testSize)
        step = 0
        for train, test in ss.split(self.x, self.y):
            xtrain = self.x[train, :]
            ytrain = self.y[train]
            plsModel = PLSRegression(min([self.ncomp, rank(xtrain)]))
            plsModel.fit(xtrain, ytrain)
            PLSCoef[step, :] = plsModel.coef_.T
            step += 1
        meanCoef = np.mean(PLSCoef, axis=0)
        stdCoef = np.std(PLSCoef, axis=0)
        self.criteria = meanCoef / stdCoef
        print('meanCoef / stdCoef', meanCoef / stdCoef)

        # 准备数据
        data_df = pd.DataFrame(self.criteria)  # 关键1，将ndarray格式转换为DataFrame
        # 更改表的索引
        data_df.columns = ['A']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
        # 将文件写入excel表格中
        writer = pd.ExcelWriter('方法UVE_stability.xlsx')  # 关键2，创建名称为hhh的excel表格
        data_df.to_excel(writer, 'page_1',
                         float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
        writer.save()  # 关键4

    def evalCriteria(self, cv=10):
        accuracy = []
        self.featureIndex = np.argsort(-np.abs(self.criteria))
        for i in range(self.x.shape[1]):
            xi = self.x[:, self.featureIndex[:i + 1]]
            # PLS-DA判别分析中转换y
            yi = pd.get_dummies(self.y)
            if i < self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))

            y_predict = cross_val_predict(regModel, xi, yi, cv=cv)
            # 将预测结果即类别矩阵转换为数值标签
            y_predict = np.array([np.argmax(i) for i in y_predict])
            RMSE_mean = accuracy_score(self.y, y_predict)
            print('特征个数', i + 1)
            print('accuracy_score(self.y, y_predict)', accuracy_score(self.y, y_predict))
            accuracy.append(accuracy_score(self.y, y_predict))
            print('np.arange(80)', np.arange(80))
            print('accuracy', accuracy)

            # self.featureR2[i] = np.mean(cvScore)
            self.featureR2[i] = RMSE_mean

            # 准备数据
            data_df = pd.DataFrame(accuracy)  # 关键1，将ndarray格式转换为DataFrame
            # 更改表的索引
            data_df.columns = ['A']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
            # 将文件写入excel表格中
            writer = pd.ExcelWriter('方法UVE_accuracy.xlsx')  # 关键2，创建名称为hhh的excel表格
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
            writer.save()  # 关键4

    def cutFeature(self, *args):
        # cuti+1是特征累积后分类精确度最大的特征个数，出图
        cuti = np.argmax(self.featureR2)
        print('cuti', cuti+1)
        self.selFeature = self.featureIndex[:cuti+1]
        if len(args) != 0:
            returnx = list(args)
            i = 0
            for argi in args:
                if argi.shape[1] == self.x.shape[1]:
                    returnx[i] = argi[:, self.selFeature]
                i += 1
        print("Uve的筛选变量索引：", self.selFeature)

        return returnx

    def getFeatureid(self, *args):
        cuti = np.argmax(self.featureR2)
        self.selFeature = self.featureIndex[:cuti+1]
        print("Uve的筛选变量索引：", self.selFeature)

        return self.selFeature