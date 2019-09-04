import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """
    load the dataset
    :return: the features and result matrix of the dataset
    """
    watermelon_data = pd.read_csv('watermelon.csv', index_col='Id')
    data_mat = watermelon_data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.weight = [0, 0]
        self.threshold = 0

    def __cal_mean(self, x):
        """
        calculate the mean value of every feature
        :param x:
        :return: the mean
        """
        mean = np.mean(x, axis=0)
        return mean

    def __cal_conv(self, x, mean):
        """
        calculate the covariance
        :param x:
        :param mean: the mean of feature
        :return: the covariance
        """
        return np.dot((x - mean).T, (x - mean))

    def __split_data(self, x, y):
        """
        split the data according to the numbers of categories
        :param x: the features vector
        :param y: the value of class
        :return: features after splitting
        """
        length = len(y)
        index_one = [i for i in range(length) if y[i] == 1]
        index_two = [i for i in range(length) if y[i] == 0]
        return x[index_one, :], x[index_two, :]

    def plot_data(self, x, y):
        """
        plot the data
        :param x: features matrix
        :param y: the class label
        :return: nothing
        """
        row, col = self.__split_data(x, y)
        plt.scatter(row[:, 0], row[:, 1], c='r', marker='*')
        plt.scatter(col[:, 0], col[:, 1], c='b', marker='.')
        t = np.linspace(0, 1, 20)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        y = (self.weight[1] / self.weight[0]) * t
        plt.title("Linear Discriminant Analysis")
        plt.plot(t, y)
        plt.show()

    def fit(self, x, y):
        """
        training data to get the weight
        :param x: the features matrix
        :param y: the class label
        :return: nothing
        """
        x_one, x_two = self.__split_data(x, y)
        mean_one = self.__cal_mean(x_one)
        mean_two = self.__cal_mean(x_two)
        conv_one = self.__cal_conv(x_one, mean_one)
        conv_two = self.__cal_conv(x_two, mean_two)
        s_w = conv_one + conv_two
        if np.linalg.det(s_w) == 0:
            print("The matrix is uninvertible. Break off.")
            return 0
        self.weight = np.dot(np.linalg.inv(s_w), (mean_one - mean_two).T)
        self.threshold = np.dot(self.weight, 0.5*(mean_one + mean_two))

    def __predict_prob(self, x):
        """
        calculate the mapping on the line
        :param x:
        :return: the result of mapping
        """
        return np.dot(x, self.weight)

    def predict(self, x):
        """
        compare to the threshold
        :param x: the validate
        :return: a bool vector
        """
        return self.__predict_prob(x) >= self.threshold


if __name__ == "__main__":
    train_X, train_y = load_data()
    model = LinearDiscriminantAnalysis()
    model.fit(train_X, train_y)
    preds = model.predict(train_X)
    print((preds == train_y).mean())
    model.plot_data(train_X, train_y)