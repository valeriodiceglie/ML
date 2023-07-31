import numpy as np

class TrainTestSplit:

    def __init__(self, method='holdout'):
        self.method = method

    def perform_split(self, x, y, k):
        if self.method == 'holdout':
            self.holdout(x, y)
        elif self.method == 'kfold':
            self.k_fold(x, y, k)

    def holdout(self, x, y, holdout = 0.8):
        '''
        Perform an holdout split of the dataset
        :param x: the feature matrix
        :param y: the target variable
        :param holdout: the holdout percentage
        :return: x_train, x_test, y_train, y_test --- The Training and Test set
        '''
        # compute train_index for holdout 80/20
        train_index = round(len(x) * holdout)

        x_train = x[:train_index]
        y_train = y[:train_index]

        x_test = x[train_index:]
        y_test = y[train_index:]

        return x_train, x_test, y_train, y_test

    def k_fold(self, x, y, k):
        '''
        Perform a k-fold split of the given dataset
        :param x: The feature matrix
        :param y: The target variable nd-array
        :param k: The number of folds
        :return:
        '''
        n = len(x)
        fold_size = n // k
        indices = np.arange(n)
        np.random.shuffle(indices)

        for i in range(k):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < k - 1 else n
                val_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

                yield x[train_indices], x[val_indices], y[train_indices], y[val_indices]

