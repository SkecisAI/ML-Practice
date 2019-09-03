import numpy as np
import pandas as pd


def load_data():
    """
    load the dataset
    :return: the features and result matrix of the dataset
    """
    watermelon_data = pd.read_csv('watermelon.csv', index_col='Id')
    data_mat = watermelon_data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iter_num=10000, fit_intercept=True):
        """
        constructor function
        :param learning_rate: the rate of learning in gradient descent
        :param iter_num: the maximum counts of iterations
        :param fit_intercept: Specifies if a constant should be added to the decision function
        """
        self.learning_rate = learning_rate
        self.iter_num = iter_num
        self.fit_intercept = fit_intercept
        self.weight = 0

    def __add_intercept(self, samples):
        """
        the vectorization of x (add 1-vector)
        :param samples:
        :return:
        """
        intercept = np.ones((samples.shape[0], 1))
        return np.concatenate((samples, intercept), axis=1)

    def __sigmod(self, z):
        """
        the sigmod function
        :param z: linear function
        :return: sigmod function value
        """
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        """
        the loss function deduced from maximum likehood function
        :param h: the sigmod function
        :param y: the target values
        :return: loss
        """
        return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

    def fit(self, samples, y):
        """
        the process of training data
        :param samples: x
        :param y: y
        :return: none (the weights after training)
        """
        if self.fit_intercept:
            samples = self.__add_intercept(samples)
        x = samples
        self.weight = np.zeros(samples.shape[1])
        # the gradient descent
        for i in range(self.iter_num):
            z = np.dot(x, self.weight)
            h = self.__sigmod(z)
            gradient = np.dot(x.T, h - y)
            self.weight -= self.learning_rate*gradient

    def predict_pro(self, samples):
        """
        the value with trained weight of sigmod function is between 0 and 1.
        :param samples: x
        :return: value
        """
        if self.fit_intercept:
            samples = self.__add_intercept(samples)
        return self.__sigmod(np.dot(samples, self.weight))

    def predict(self, x, threshold=0.5):
        """
        predict the result
        :param x: x
        :param threshold: the threashold of the sigmod function 0.5
        :return:a boolean value vector
        """
        return self.predict_pro(x) >= threshold


if __name__ == '__main__':
    train_X, train_y = load_data()
    model = LogisticRegression(0.1, iter_num=20000)
    model.fit(train_X, train_y)
    preds = model.predict(train_X)
    print((preds == train_y).mean())