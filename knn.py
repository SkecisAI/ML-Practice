import numpy as np


class KNN:
    def __init__(self, k):
        """
        the k of KNN
        :param k:
        """
        self.k = k
        self.samples_nums = 0

    def predict(self, x_train, y_train, test):
        """
        predict the categorie of test sample
        :param x_train: the set of train sample
        :param y_train: the label set of train sample
        :param test:
        :return:
        """
        x_train = np.array(x_train)
        self.samples_nums = x_train.shape[0]
        y_train = np.array(y_train)
        sorted_k_v = self.__cal_dist(x_train, test)
        lab_dict = {}
        for ind in range(self.k):
            s = sorted_k_v[ind]
            if y_train[s[0]] not in lab_dict.keys():
                lab_dict[y_train[s[0]]] = 1
            else:
                lab_dict[y_train[s[0]]] += 1
        return max(lab_dict, key=lab_dict.get)

    def __cal_dist(self, x_train, x_test):
        """
        calculate the distance between train set and test sample
        :param x_train:
        :param x_test:
        :return:
        """
        d = {}
        for ind in range(self.samples_nums):
            d[ind] = np.sqrt(sum(np.power(x_train[ind, :] - x_test, 2)))
        sort_list = sorted(d.items(), key=lambda itm: itm[1])
        return sort_list


if __name__ == "__main__":
    my_knn = KNN(3)
    x = [[0.1, 0.2], [-1.5, -1.1], [0.7, 1.9], [-0.3, 1.0]]
    y = ['a', 'a', 'b', 'b']
    print(my_knn.predict(x, y, [1, 1]))
