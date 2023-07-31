import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, n_features=1, learning_rate=1e-2, steps=2000, lambda_=0, reg_type=None):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.steps = steps
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.theta = np.random.rand(n_features + 1)
        self.theta_history = np.zeros((self.steps, self.theta.shape[0]))
        self.history = []

    def compute_cost(self, m, error):
        cost = None
        if self.reg_type is None:
            cost = 1/(2 * m) * np.sum(error)**2
        elif self.reg_type == 'L1':
            cost = 1/(2 * m) * np.sum(error)**2 + (self.lambda_/2*m) * np.sum(np.abs(self.theta))
        elif self.reg_type == 'L2':
            cost = 1/(2*m) * np.sum(error)**2 + (self.lambda_/2*m) * np.sum(self.theta**2)
        return cost

    def get_scores(self, pred, y):
        mae = self.mae(pred, y)
        mse = self.mse(pred, y)
        rmse = self.rmse(pred, y)
        r2 = self.r2(pred, y)
        return mae, mse, rmse, r2

    @staticmethod
    def mae(prediction, y):
        return np.average(np.abs(prediction - y))

    @staticmethod
    def mse(prediction, y):
        return np.average(np.square(prediction - y))

    @staticmethod
    def rmse(prediction, y):
        return np.sqrt(np.average(np.square(prediction - y)))

    @staticmethod
    def r2(prediction, y):
        ssr = np.sum(prediction - np.average(y))
        sst = np.sum(y - np.average(y))
        return 1 - ssr/sst


class LinearRegressionMiniBatch(LinearRegression):
    def __init__(self, learning_rate=1e-2, steps=2000, n_features=1, num_batch=1,
                 lambda_=0, reg_type=None):
        super().__init__(n_features, learning_rate, steps, lambda_, reg_type)
        self.num_batch = num_batch

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        for i in range(self.steps):
            indices = np.random.permutation(m)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            x_batch = np.array_split(x_train_shuffled, self.num_batch)
            y_batch = np.array_split(y_train_shuffled, self.num_batch)
            for j in range(0, self.num_batch):
                pred = np.dot(x_batch[j], self.theta)
                error = pred - y_batch[j]
                reg = 1 - (self.lambda_ * self.learning_rate) / x_batch[j].shape[0]
                cost = self.compute_cost(x_batch[j].shape[0], error)
                self.history.append(cost)
                self.theta = self.theta * reg - (self.learning_rate / x_batch[j].shape[0]) * np.dot(x_batch[j].T, error)
            self.theta_history[i, :] = self.theta

    def predict(self, x_test):
        x_pred = np.c_[np.ones(x_test.shape[0]), x_test]
        predictions = np.dot(x_pred, self.theta)
        return predictions

    def plot(self):
        plt.plot(np.arange(1, self.steps * self.num_batch + 1), self.history, color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("LOSS")
        plt.show()
        plt.close()


class LinearRegressionFullBatch(LinearRegression):

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        for i in range(self.steps):
            predictions = np.dot(x_train, self.theta)
            error = predictions - y_train
            cost = self.compute_cost(m, error)
            self.history.append(cost)
            reg = 1 - (self.learning_rate * self.lambda_) / m
            self.theta = self.theta * reg - self.learning_rate/m * np.dot(x_train.T, error)
            self.theta_history[i, :] = self.theta

    def predict(self, x_test):
        x_pred = np.c_[np.ones(x_test.shape[0]), x_test]
        predictions = np.dot(x_pred, self.theta)
        return predictions

    def plot(self):
        plt.plot(np.arange(1, self.steps + 1), self.history, color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close()


class LinearRegressionStochastic(LinearRegression):

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        for i in range(self.steps):
            for j in range(m):
                indices = np.random.permutation(m)
                x_shuffle_train = x_train[indices]
                y_shuffle_train = y_train[indices]

                predictions = np.dot(x_shuffle_train[j, :], self.theta)
                error = predictions - y_shuffle_train[j]
                cost = self.compute_cost(error, 1)
                self.history.append(cost)
                reg = 1 - self.learning_rate * self.lambda_
                self.theta = self.theta * reg - self.learning_rate * np.dot(x_shuffle_train[j], error)
                self.theta_history[j, :] = self.theta

    def predict(self, x_test):
        x_pred = np.c_[np.ones(x_test.shape[0]), x_test]
        prediction = np.dot(x_pred, self.theta)
        return prediction


    def plot(self, m):
        plt.plot(np.arange(1, self.steps * m + 1), self.history, color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close()


