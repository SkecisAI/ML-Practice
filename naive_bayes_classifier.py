import numpy as np
import pandas as pd


def load_data():
    """
    load the dataset
    :return a ndarray: the features and result matrix of the dataset
    """
    watermelon_data = pd.read_csv('watermelon3.csv', index_col='编号')
    data_mat = watermelon_data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]


class NaiveBayes:
    def __init__(self):
        self.labels_dict = {}
        self.attributes_dict = {}
        self.train_samples = None
        self.train_labels = None

    def __create_labels_dict(self, labels):
        for label in set(labels):
            self.labels_dict[label] = list(labels).count(label)

    def __create_attrib_dict(self, x, y):
        feat_nums = x.shape[1]
        for ind in range(feat_nums):
            if isinstance(x[0, ind], str):
                for val in set(x[:, ind]):
                    self.attributes_dict[val] = self.__split_data(x, y, ind, val)

    def __split_data(self, x, y, i, val):
        samples_num = x.shape[0]
        val_label = []
        val_dict = {}
        for label in set(y):
            val_dict[label] = 0
        for s in range(samples_num):
            if x[s, i] == val:
                val_label.append(y[s])
        for label in set(val_label):
            val_dict[label] = list(val_label).count(label)
        return val_dict

    def __cal_continuous_feat(self, feat_val, feat_id, spec_label):
        sample_nums = self.train_samples.shape[0]
        target_vet = []
        for i in range(sample_nums):
            if self.train_labels[i] == spec_label:
                target_vet.append(self.train_samples[i, feat_id])
        ave = np.mean(target_vet)
        var = np.var(target_vet)
        p = (1 / np.sqrt(2*np.pi*var)) * np.exp(-(feat_val - ave)**2 / (2*var))
        return p

    def fit(self, x_train, y_train):
        self.__create_labels_dict(y_train)
        self.__create_attrib_dict(x_train, y_train)
        self.train_samples = x_train
        self.train_labels = y_train

    def predict(self, sample):
        predict_prob = {}
        sample_nums = self.train_samples.shape[0]
        n = len(self.labels_dict.keys())
        for label in self.labels_dict.keys():
            predict_prob[label] = (self.labels_dict[label] + 1) / (sample_nums + n)
            for feat_nums in range(len(sample)):
                if isinstance(sample[feat_nums], str):
                    # if the attribute is scatter
                    ni = len(set(self.train_samples[:, feat_nums]))
                    dcx = self.attributes_dict[sample[feat_nums]][label]
                    predict_prob[label] *= (dcx + 1) / (self.labels_dict[label] + ni)
                else:
                    # if the attribute is continous
                    predict_prob[label] *= self.__cal_continuous_feat(sample[feat_nums], feat_nums, label)
        print(predict_prob)


if __name__ == "__main__":
    train_X, train_y = load_data()
    model = NaiveBayes()
    smp = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
    model.fit(train_X, train_y)
    model.predict(smp)