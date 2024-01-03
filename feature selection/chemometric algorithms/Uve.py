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
        return ：筛选后的数据
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

    def evalCriteria(self, cv=10):
        accuracy = []
        self.featureIndex = np.argsort(-np.abs(self.criteria))
        for i in range(self.x.shape[1]):
            xi = self.x[:, self.featureIndex[:i + 1]]
            yi = pd.get_dummies(self.y)
            if i < self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))

            y_predict = cross_val_predict(regModel, xi, yi, cv=cv)
            y_predict = np.array([np.argmax(i) for i in y_predict])
            RMSE_mean = accuracy_score(self.y, y_predict)
            print('accuracy_score(self.y, y_predict)', accuracy_score(self.y, y_predict))
            accuracy.append(accuracy_score(self.y, y_predict))
            print('accuracy', accuracy)

            # self.featureR2[i] = np.mean(cvScore)
            self.featureR2[i] = RMSE_mean

    def cutFeature(self, *args):
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
        print("index：", self.selFeature)

        return returnx

    def getFeatureid(self, *args):
        cuti = np.argmax(self.featureR2)
        self.selFeature = self.featureIndex[:cuti+1]
        print("index_important：", self.selFeature)

        return self.selFeature
