import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


def load_data():
    """
    load the dataset
    :return a ndarray: the features and result matrix of the dataset
    """
    watermelon_data = pd.read_csv('watermelon3.csv', index_col='编号')
    watermelon_data = hash_encode(watermelon_data)
    data_mat = watermelon_data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]


def hash_encode(df):
    """
    factorize the data frame
    :param df: a dataframe
    :return:
    """
    df['好瓜'] = pd.factorize(df['好瓜'])[0]
    df['好瓜'] = df['好瓜'].apply(lambda x: 1 if x == 0 else 0)
    df = pd.get_dummies(df)
    # put the label at last
    cols = list(df)
    cols.pop(cols.index('好瓜'))
    cols.append('好瓜')
    df = df.loc[:, cols]
    return df


class NeuralNetwork:
    def __init__(self, learning_rate, layers=1, iter_nums=5000):
        self.lr = learning_rate
        self.layers = layers
        self.hide_node = 3
        self.iters = iter_nums
        self.weight = []
        self.bias = []

    def __active_fun(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def __init_para(self, x_train):
        rows, cols = x_train.shape
        w_1 = np.random.randn(cols, self.hide_node)
        b_1 = np.zeros((1, self.hide_node))
        w_2 = np.random.randn(self.hide_node, 1)
        b_2 = np.zeros((1, 1))
        return w_1, w_2, b_1, b_2

    def __mean_square_error(self, predict_y, y):
        differ = predict_y - y
        error = 0.5*sum(differ**2)
        return error

    def fit(self, x_train, y_train):
        self.hide_node = int(np.round(np.sqrt(x_train.shape[0])))
        # initialize the network parameter
        w_1, w_2, b_1, b_2 = self.__init_para(x_train)
        for i in range(self.iters):
            accum_error = 0
            for s in range(x_train.shape[0]):
                hidein_1 = np.dot(x_train[s, :], w_1) + b_1
                hideout_1 = self.__active_fun(hidein_1)

                hidein_2 = np.dot(hideout_1, w_2) + b_2
                hideout_2 = self.__active_fun(hidein_2)
                predict_y = hideout_2
                accum_error += self.__mean_square_error(predict_y, y_train[s])
            if accum_error <= 0.001:
                break
            else:  # update the parameter
                for s in range(x_train.shape[0]):
                    in_nums = x_train.shape[1]
                    # layer 1
                    hidein_1 = np.dot(x_train[s, :], w_1) + b_1
                    hideout_1 = self.__active_fun(hidein_1)
                    # layer 2
                    hidein_2 = np.dot(hideout_1, w_2) + b_2
                    hideout_2 = self.__active_fun(hidein_2)
                    predict_y = hideout_2

                    g = predict_y*(1 - predict_y)*(y_train[s] - predict_y)
                    e = g*w_2.T*(hideout_1*(1 - hideout_1))
                    w_2 = w_2 + self.lr*hideout_1.T*g
                    b_2 = b_2 - self.lr*g
                    w_1 = w_1 + self.lr*x_train[s, :].reshape(in_nums, 1)*e
                    b_1 = b_1 - self.lr*e
        self.weight.append(w_1)
        self.weight.append(w_2)
        self.bias.append(b_1)
        self.bias.append(b_2)

    def predict(self, x_train):
        hidein_1 = np.dot(x_train, self.weight[0]) + self.bias[0]
        hideout_1 = self.__active_fun(hidein_1)

        hidein_2 = np.dot(hideout_1, self.weight[1]) + self.bias[1]
        hideout_2 = self.__active_fun(hidein_2)
        predict_y = hideout_2
        predict_y = [1 if e > 0.5 else 0 for e in predict_y]
        print(predict_y)


if __name__ == "__main__":
    train_X, train_y = load_data()
    model = NeuralNetwork(0.1)
    model.fit(train_X, train_y)
    model.predict(train_X)