import numpy as np
import pandas as pd
import copy


def load_data():
    """
    load the dataset
    :return a ndarray: the features and result matrix of the dataset
    """
    watermelon_data = pd.read_csv('watermelon4.csv', index_col='编号')
    data_mat = watermelon_data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]


def get_attributes(get_vals=False):
    """
    get the attributes list
    :param get_vals:
    :return a list: the attributes
    """
    watermelon_data = pd.read_csv('watermelon4.csv', index_col='编号')
    if get_vals:
        # return a dataframe without label column
        attr_df = watermelon_data.drop(['好瓜'], axis=1)
        return attr_df
    attributes = watermelon_data.columns[:watermelon_data.shape[1]-1].values
    return list(attributes)


class DecisionTree:
    def __init__(self, method='infogain', scatter_method='ave', prune_method=None):
        """
        initilize with method
        :param method: the method of dividing attributes sets,information gain, gini index and logistic regression.
        :param scatter_method: the method of split sequence value, average of every class's average and dichotomy
        :param prune_method: the method of pruning, pre-pruning and post-pruning
        """
        self.method = method
        self.scatter_method = scatter_method
        self.prune_method = prune_method
        self.my_tree = None  # the result tree
        self.prune_tree = None  # the tree after pruning
        self.node_dict = {}
        # the validation set
        self.valid_data = None
        self.attri_vals = get_attributes(get_vals=True)

    def __cal_entropy(self, y):
        """
        calculate the entropy according to the laebls
        :param y: the vector of label
        :return: the information entropy
        """
        entropy = 0.0
        y = list(y)
        for unique_val in set(y):
            p = (y.count(unique_val) / len(y))
            entropy -= p * np.log2(p)
        return entropy

    def __cal_gini(self, y):
        """
        calculate the gini index according to the labels
        :param y: the vector of label
        :return: the gini index
        """
        gini_index = 1
        y = list(y)
        for unique_val in set(y):
            p = y.count(unique_val) / len(y)
            gini_index -= p**2
        return gini_index

    def __split_valid_data(self, x, y, valid_ind):
        """
        split the validation data
        :param x: the full samples
        :param y: the full labels
        :param valid_ind: the index of validation samples
        :return: the validation data
        """
        samples_num = x.shape[0]
        valid_ind = [ind-1 for ind in valid_ind]
        train_ind = []
        for i in range(samples_num):
            if i not in valid_ind:
                train_ind.append(i)
        x_valid = np.delete(x, train_ind, axis=0)
        y_valid = np.delete(y, train_ind, axis=0)
        x_train = np.delete(x, valid_ind, axis=0)
        y_train = np.delete(y, valid_ind, axis=0)
        df = np.c_[x_valid, y_valid]
        col = get_attributes()
        col.append('label')
        data_valid = pd.DataFrame(data=df, columns=col)
        self.valid_data = data_valid
        return x_train, y_train

    def __pruning(self, x_train, y_train):
        """
        the processing of pre-pruning and post-pruning
        :param x_train: ndarray, samples
        :param y_train: ndarray, labels
        :return: the tree after pruning
        """
        if self.prune_method == 'pre':
            root_name = list(self.my_tree.keys())[0]
            prune_tree = {root_name: {}}  # initialize the root node
            label = max(y_train, key=list(y_train).count)
            valid_vals = list(self.valid_data['label'].values)
            pre_accu = valid_vals.count(label) / len(valid_vals)
            tree = copy.deepcopy(self.my_tree)
            feat_branch = tree[root_name]
            attributes = get_attributes()
            # tree after the process of pre-pruning
            valid_data = np.array(self.valid_data)
            self.prune_tree = self.__pre_pruning(prune_tree, root_name, feat_branch, x_train, y_train, pre_accu, label, attributes, valid_data)
        elif self.prune_method == 'post':
            self.__get_dict(self.my_tree, self.node_dict, 0)
            copy_tree = copy.deepcopy(self.my_tree)
            node_nums = 0
            attributes = get_attributes()
            for key in self.node_dict:
                node_nums += len(self.node_dict[key])
            # delete the node one by one except the root node(-1)
            for i in range(node_nums - 1):
                key_name, key_loc = self.__modify_dict(self.node_dict)
                pred_labels = self.predict(use_valid=self.valid_data, tree=copy_tree)
                true_bool = np.array(pred_labels) == np.array(self.valid_data)[:, -1]
                pre_accu = np.mean(true_bool)
                tmp_tree = copy.deepcopy(copy_tree)
                self.__post_pruning(tmp_tree, key_name, key_loc, x_train, y_train, 0, attributes[:])
                pred_labels = self.predict(use_valid=self.valid_data, tree=tmp_tree)
                true_bool = np.array(pred_labels) == np.array(self.valid_data)[:, -1]
                later_accu = np.mean(true_bool)
                if later_accu > pre_accu:
                    # if the pruning bring the boosting, replace the origin tree
                    copy_tree = copy.deepcopy(tmp_tree)
            self.prune_tree = copy_tree
        return self.prune_tree

    def __pre_pruning(self, prune_tree, node_name, tree_node, x_train, y_train, pre_accu, leaf_label, attributes, valid_data):
        """
        :param prune_tree: the pre-pruning tree
        :param node_name: the feature's node
        :param tree_node: node of the tree without pruning
        :param x_train: the training samples
        :param y_train: the traaining labels
        :param pre_accu: the accuracy of before pruning
        :param leaf_label: the label of leaf node
        :param attributes: the attributes set
        :param valid_data: the valid_data after changing
        :return: the prune_tree
        """
        if isinstance(tree_node, str):
            # if this node is a leaf node
            return tree_node
        feat_name = node_name
        feat_id = attributes.index(feat_name)
        del attributes[feat_id]
        for feat_val in list(tree_node.keys()):
            # spread a feature node
            x, y = self.__split_data(x_train, y_train, feat_id, feat_val)
            if y.size == 0 & x.size == 0:
                prune_tree[feat_name][feat_val] = max(y_train, key=list(y_train).count)
            else:
                most_label = max(y, key=list(y).count)
                prune_tree[feat_name][feat_val] = most_label
        pred_labels = self.predict(use_valid=valid_data, tree=prune_tree)
        true_bool = np.array(pred_labels) == valid_data[:, -1]
        if valid_data.shape[0] < self.valid_data.shape[0]:
            # already have a leaf's division
            if np.mean(true_bool) < 1:
                # division can't get a boosting, return leaf label
                return leaf_label
        # reorganize validation data
        false_ind = []
        for i in range(len(true_bool)):
            if not true_bool[i]:
                false_ind.append(i)
        valid_data = np.delete(valid_data, false_ind, axis=0)
        later_accu = np.mean(true_bool)
        threshold = 0
        if (pre_accu - later_accu) >= threshold:
            # if division can't boost the performance, and set threhold to 0
            # the bigger threshold, the less the probability of division
            return leaf_label
        else:
            # else spread the leaf node
            pre_accu = later_accu
            for feat_val in list(tree_node.keys()):
                x, y = self.__split_data(x_train, y_train, feat_id, feat_val)
                if y.size == 0 & x.size == 0:
                    prune_tree[feat_name][feat_val] = max(y_train, key=list(y_train).count)
                else:
                    most_label = max(y, key=list(y).count)
                    leaf_label = most_label  # get the label of current node
                    feature_node = tree_node[feat_val]  # get feat_val's next node
                    if isinstance(feature_node, str):
                        prune_tree[feat_name][feat_val] = feature_node
                    else:
                        copy_attr = attributes[:]
                        feature_node_name = list(feature_node.keys())[0]
                        prune_tree[feat_name][feat_val] = self.__pre_pruning(feature_node,
                                                                             feature_node_name,
                                                                             feature_node[feature_node_name],
                                                                             x, y,
                                                                             pre_accu, leaf_label,
                                                                             copy_attr,
                                                                             valid_data)
            return prune_tree

    def __post_pruning(self, tree, key_name, key_loc, x_train, y_train, loc, attributes):
        """
        find the target node, replace it by the represent label and return
        :param tree: the full tree
        :param key_name: the spliting node's name
        :param key_loc: the location of node
        :param x_train: the training samples
        :param y_train: the training labels
        :param loc: the current location
        :param attributes: the attributes set
        :return: nothing
        """
        if isinstance(tree, str):
            # if the node is leaf node
            pass
        else:
            keys = list(tree.keys())
            feat_name = keys[0]
            feat_id = attributes.index(feat_name)
            feat_vals = list(tree[feat_name].keys())
            del attributes[feat_id]
            loc += 1
            for val in feat_vals:
                x, y = self.__split_data(x_train, y_train, feat_id, val)
                sub_attributes = attributes[:]
                # if the data after spliting is empty, the next_node is leaf node
                next_node = tree[feat_name][val]
                node_name = 'None'
                if not isinstance(next_node, str):
                    node_name = list(next_node.keys())[0]
                if (node_name == key_name) & (loc == key_loc):
                    # if next node satisfies that both feature's name and location is same
                    tree[feat_name][val] = max(y, key=list(y).count)
                    return
                else:
                    self.__post_pruning(tree[feat_name][val], key_name, key_loc, x, y, loc, sub_attributes)

    def __modify_dict(self, node_dict):
        """
        modify the dict of features node of tree
        :param node_dict: the dict of tree's node
        :return: the dict after modifying
        """
        max_key = None
        max_id = 0
        for key_name in list(node_dict.keys()):
            m = max(node_dict[key_name])
            if m >= max_id:
                max_key = key_name
                max_id = m
        node_dict[max_key].remove(max_id)
        if not node_dict[max_key]:
            del node_dict[max_key]
        return max_key, max_id

    def __get_dict(self, tree, feat_dict, seq):
        """
        generate the feature node sequence by the decision without pruning
        :param tree: a dict, the decision without pruning
        :param feat_dict: a dict, the feature node with sequence
        :return: None
        """
        keys = list(tree.keys())
        feat_name = keys[0]
        if seq == 0:
            tmp = dict.fromkeys([feat_name], [seq])
            feat_dict.update(tmp)
        else:
            if feat_name in feat_dict:
                feat_dict[feat_name].append(seq)
            else:
                tmp = dict.fromkeys([feat_name], [seq])
                feat_dict.update(tmp)
        feat_vals = tree[feat_name]
        seq += 1
        for val in list(feat_vals.keys()):
            if isinstance(tree[feat_name][val], str):
                pass
            else:
                self.__get_dict(tree[feat_name][val], feat_dict, seq)

    def __scatter(self, x, y, attributes):
        """
        the process of scatter continous values
        :param x: a ndarray, samples
        :param y: a ndarray, label
        :param attributes: a list, attributes set
        :return: samples after scattering values
        """
        scatter_index = []
        for i in range(x.shape[1]):
            if not isinstance(x[0, i], str):  # find the index whose value is continuous.
                scatter_index.append(i)
        for ind in scatter_index:
            split_threshold = 0.0
            if self.scatter_method == 'ave':  # specify the average of values is the threshold
                split_threshold = np.mean(x[:, ind])
            elif self.scatter_method == 'dicho':  # specify the best information gain of values is the threshlod
                vals_sorted = sorted(x[:, ind])
                total_ent = self.__cal_entropy(y)
                best_ent = 0.0
                best_point = 0.0
                for a, b in zip(vals_sorted[0:-1], vals_sorted[1:]):
                    med_point = (a + b) / 2
                    copy_x = x.copy()  # create a deep copy of x(ndarray)
                    split_ent = 0.0
                    for vec in copy_x:
                        if vec[ind] >= med_point:
                            vec[ind] = 'more'
                        else:
                            vec[ind] = 'less'
                    for val in set(copy_x[:, ind]):
                        vals_unique, y_val = self.__split_data(copy_x, y, ind, val)
                        count = vals_unique.shape[0]
                        split_ent += (count / copy_x.shape[0]) * self.__cal_entropy(y_val)
                    if (total_ent - split_ent) >= best_ent:
                        best_ent = (total_ent - split_ent)
                        best_point = med_point
                split_threshold = best_point
            more = "%s>=%.3f" % (attributes[ind], split_threshold)
            less = "%s<%.3f" % (attributes[ind], split_threshold)
            func = np.vectorize(lambda e: more if e >= split_threshold else less)
            x[:, ind] = func(x[:, ind])
        for ind in scatter_index:
            # modify the self values at the same time
            self.attri_vals[attributes[ind]] = x[:, ind]
        return x

    def __split_data(self, x, y, best_feat_id, val):
        """
        split the data which is not the best feature's specific value
        :param x: samples
        :param y: labels
        :param best_feat_id: the index of best features in the samples
        :param val: the specific value
        :return x, y: strip the samples with specific value
        """
        split_ind = []
        for i in range(x.shape[0]):
            if x[i, best_feat_id] != val:  # drop others
                split_ind.append(i)
        x = np.delete(x, split_ind, axis=0)
        x = np.delete(x, best_feat_id, axis=1)
        y = np.delete(y, split_ind, axis=0)
        return x, y

    def __choose_best_feature(self, x, y, m):
        """
        choose different method to choose best feature
        :param x nd-array: the samples matrix
        :param y nd-array: the labels
        :param m: the method of spliting attributes
        :return: the id of the best feature
        """
        total_ent = self.__cal_entropy(y)
        samples_num = x.shape[0]
        best_feature = 0
        if m == 'infogain':  # method is infogain
            max_gain = 0.0
            for i in range(x.shape[1]):  # for every feature
                x_unique = set(x[:, i])  # unique value of every feature
                split_ent = 0.0
                for val in x_unique:
                    vals_unique, y_val = self.__split_data(x, y, i, val)
                    count = vals_unique.shape[0]
                    split_ent += (count / samples_num) * self.__cal_entropy(y_val)
                if (total_ent - split_ent) >= max_gain:  # compare the information gain to the total entropy
                    max_gain = (total_ent - split_ent)
                    best_feature = i
        elif m == 'gini':
            min_gini = 9999
            for i in range(x.shape[1]):
                x_unique = set(x[:, i])
                feat_gini = 0.0
                for val in x_unique:
                    vals_unique, y_val = self.__split_data(x, y, i, val)
                    count = vals_unique.shape[0]
                    feat_gini += (count / samples_num) * self.__cal_gini(y_val)
                if feat_gini <= min_gini:
                    min_gini = feat_gini
                    best_feature = i
        elif m == 'logistic':
            # TODO: implement logistic function
            pass
        return best_feature

    def __create_tree(self, samples, samples_y, attributes):
        """
        create the decision tree recursively
        :param samples: a nd-array, which is the samples
        :param samples_y: a 1-d nd-array, which is the label of every sample
        :param attributes: the list of attributes
        :return: a decision tree
        """
        if len(set(samples_y)) == 1:
            return samples_y[0]
        ans = 1
        for i in range(samples.shape[1]):
            ans *= len(set(samples[:, i]))
        if not attributes or ans == 1:
            return max(samples_y, key=list(samples_y).count)
        best_feat_id = self.__choose_best_feature(samples, samples_y, self.method)
        best_feat_name = attributes[best_feat_id]
        best_feat_vals = np.array(self.attri_vals[best_feat_name])
        del attributes[best_feat_id]
        my_tree = {best_feat_name: {}}  # generate a new node for the best feature
        for val in set(best_feat_vals):
            # split specific value data from full data
            x, y = self.__split_data(samples, samples_y, best_feat_id, val)
            if y.size == 0 & x.size == 0:
                my_tree[best_feat_name][val] = max(samples_y, key=list(samples_y).count)
            else:
                sub_attributes = attributes[:]  # pass the copy of attributes, because list is a mutable value
                my_tree[best_feat_name][val] = self.__create_tree(x, y, sub_attributes)
        return my_tree

    def __traverse_tree(self, sample, tree):
        """
        predict a sample by tree
        :param sample: a 1-d ndarray
        :param tree: the tree dict
        :return: str, the label value
        """
        attributes = get_attributes()
        if isinstance(tree, set):
            #  if the tree is a label
            return tree.pop()
        elif isinstance(tree, str):
            return tree
        else:
            feat_name = list(tree.keys())
            feat_ind = attributes.index(feat_name[0])
            sample_val = sample[feat_ind]
            return self.__traverse_tree(sample, tree[feat_name[0]][sample_val])

    def fit(self, samples, y):
        """
        use the train data to create the decison tree.
        :param samples: a ndarray, which is the sample data
        :param y: a ndarray, which is the class vector
        :return:
        """
        attributes = get_attributes()
        samples = self.__scatter(samples, y, attributes)
        if self.prune_method:
            # if it needs pruning, choose the index of valid dataset and split two parts
            # and create the decision tree with training set
            valid_ind = [4, 5, 8, 9, 11, 12, 13]
            x_train, y_train = self.__split_valid_data(samples, y, valid_ind)
            self.my_tree = self.__create_tree(x_train, y_train, attributes)
            self.prune_tree = self.__pruning(x_train, y_train)
        else:
            self.my_tree = self.__create_tree(samples, y, attributes)
        print('my tree:', self.my_tree)
        if self.prune_method:
            print('pruning tree:', self.prune_tree)
            pred = self.predict(samples, use_valid=self.valid_data, tree=self.my_tree)
            print('prediction:', pred)
            print('accuracy: %.1f%%' % (self.predict_accu(pred) * 100))
        else:
            pred = self.predict(samples, use_valid=None, tree=self.my_tree)
            print('prediction:', pred)

    def predict(self, test_x=None, use_valid=None, tree=None):
        """
        predict the label
        :param test_x: test samples ndarray
        :param use_valid: bool, if use valid data or not
        :param tree: dict, the decision tree
        :return: a list, prediciton
        """
        predict_y = []
        if use_valid is not None:
            # use validation set to predict
            valid_data = np.delete(np.array(use_valid), -1, axis=1)
            samples = np.array(valid_data)
        else:
            # use the test data
            samples = test_x
            tree = self.my_tree
        # get the predicted labels
        for sample in samples:
            copy_tree = tree.copy()
            predict_y.append(self.__traverse_tree(sample, copy_tree))
        return predict_y

    def predict_accu(self, predict_y):
        """
        the prediction of labels
        :param predict_y: a list, the prediction
        :return: a float, accuracy
        """
        ans = self.valid_data['label']
        accu = np.mean(ans == np.array(predict_y))
        return accu


if __name__ == "__main__":
    train_X, train_y = load_data()
    model = DecisionTree(method='infogain', scatter_method='dicho', prune_method='post')
    model.fit(train_X, train_y)