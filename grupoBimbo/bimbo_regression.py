import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import xgboost as xgb


class BimboRegression:

    def init(self, filename):
        """

        :param filename:
        :param has_header
        :return:
        """
        data = pd.read_csv(filename, header=0)
        shape = data.shape
        # rows = shape[0]
        cols = shape[1]
        self.target = data.iloc[:, cols-1]
        self.x = data.iloc[:, 0:cols-5]
        print('target shape:', self.target.shape)
        print('x shape:', self.x.shape)

        # return (self.x, self.target)

    def dataset_split(self, test_percent):
        """

        :return:
        """
        return train_test_split(self.x, self.target, test_size=test_percent, random_state=41)

    def train_xgb(self):
        """

        :return:
        """
        x_train, x_test, y_train, y_test = self.dataset_split(0.2)
        # xgb_params = {'max_depth':[4, 6, 7], 'n_estimators':[20, 50, 100]}
        xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=100, nthread=40, learning_rate=0.1)
        xgb_model.fit(x_train, y_train)
        p_test = xgb_model.predict(x_test)

        rmsle = self.evalMetric(p_test, y_test)
        print('Root Mean Squared Logarithmic Error : %.4f' % rmsle)

        # The mean square error
        # print('Residual sum of squares: %.2f' % np.mean((p_test - y_test)**2))
        # Explained variance score: 1 is perfect prediction
        # print('Variance score: %.2f'
        #       % xgb_model.score(x_test, y_test))

        # gscv = GridSearchCV(xgb_model, xgb_params, verbose=1, n_jobs=4)
        # gscv.fit(x_train, y_train)
        # print(gscv.best_score_)
        # print(gscv.best_params_)



    def train(self):
        """

        :return:
        """

        x_train, x_test, y_train, y_test = self.dataset_split(0.2)

        regressor = lm.LinearRegression()
        regressor.fit(x_train, y_train)

        p_test = regressor.predict(x_test)

        rmsle = self.evalMetric(p_test, y_test)
        print('Root Mean Squared Logarithmic Error : %.4f' % rmsle)

        print('Coefficients: \n', regressor.coef_)
        # The mean square error
        print('Residual sum of squares: %.2f' % np.mean((p_test - y_test)**2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f'
              % regressor.score(x_test, y_test))

        # print('x_test shape:', x_test.shape)
        # print('y_test shape:', y_test.shape)
        # print('p_test shape:', p_test.shape)

        # Plot outputs: 10 dimensions, no way to plot
        # plt.scatter(x_test.iloc[:, 0], y_test, color='blue')
        # plt.plot(x_test.iloc[:, 0], p_test, color='green', linewidth=3)
        # plt.xticks()
        # plt.yticks()
        # plt.show()

    def evalMetric(self, p_set, a_set):
        n = p_set.shape[0]
        sqsum = np.sum(np.power(np.log(p_set+1)-np.log(a_set+1), 2))
        rmsle = math.sqrt(sqsum/n)
        return rmsle

    def main(self, filename):
        self.init(filename)
        # self.train()
        self.train_xgb()


if __name__ == "__main__":
    BimboRegression().main("dataset/train_test.csv")
