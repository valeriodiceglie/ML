import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionMiniBatch:
    def __init__(self, learning_rate=1e-2, steps=2000, n_features=1, num_batch=1,
                 lambda_=1, reg_type='L2'):
        self.learning_rate = learning_rate
        self.steps = steps
        self.n_features = n_features
        self.theta = np.random.rand(self.n_features)
        self.num_batch = num_batch
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.theta_history = np.zeros((self.steps, self.theta.shape[0]))
        self.history = []

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
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

    def compute_cost(self, m, error):
        cost = None
        if self.reg_type == 'L2':
            cost = 1/(2 * m) * np.sum(error)**2 + self.lambda_/(2 * m) * (np.sum(np.square(self.theta)))
        elif self.reg_type == 'L1':
            cost = 1/(2 * m) * np.sum(error)**2 + self.lambda_/(2 * m) * (np.sum(np.abs(self.theta)))
        return cost

    def predict(self, x_test):
        x_pred = np.c_[x_test, np.ones(x_test.shape[0])]
        predictions = np.dot(x_pred, self.theta)
        return predictions

    @staticmethod
    def mae(pred, y):
        return np.average(np.abs(pred - y))

    @staticmethod
    def mse(pred, y):
        return np.average((pred - y)**2)

    @staticmethod
    def rmse(pred, y):
        return np.sqrt(np.average((pred - y)**2))

    @staticmethod
    def r2(pred, y):
        ssr = np.sum((pred - np.average(y))**2)
        sst = np.sum((y - np.average(y))**2)
        return 1 - ssr/sst

    def get_scores(self, pred, y):
        mae = self.mae(pred, y)
        mse = self.mse(pred, y)
        rmse = self.rmse(pred, y)
        r2 = self.r2(pred, y)
        return mae, mse, rmse, r2

    def plot(self):
        plt.plot(np.arange(1, self.steps * self.num_batch + 1), self.history, color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("LOSS")
        plt.show()
        plt.close()


class LinearRegressionFullBatch:
    def __init__(self, n_features=1, lr=1e-2, steps=2000, lambda_=0, reg_type=None):
        self.n_features = n_features
        self.lr = lr
        self.steps = steps
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.theta = np.random.rand(n_features)
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

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        for i in range(self.steps):
            predictions = np.dot(x_train, self.theta)
            error = predictions - y_train
            cost = self.compute_cost(m, error)
            self.history.append(cost)
            reg = 1 - (self.lr * self.lambda_) / m
            self.theta = self.theta * reg - self.lr/m * np.dot(x_train.T, error)
            self.theta_history[i, :] = self.theta

    def predict(self, x_test):
        x_pred = np.c_[x_test, np.ones(x_test.shape[0])]
        predictions = np.dot(x_pred, self.theta)
        return predictions

    @staticmethod
    def mae(predictions, y):
        return np.average(np.abs(predictions - y))

    @staticmethod
    def mse(predictions, y):
        return np.average(np.square(predictions - y))

    @staticmethod
    def rmse(predictions, y):
        return np.sqrt(np.average(np.square(predictions - y)))

    @staticmethod
    def r2(predictions, y):
        ssr = np.sum((predictions - np.average(y))**2)
        sst = np.sum((y - np.average(y))**2)
        return 1 - ssr/sst

    def get_scores(self, predictions, y):
        mae = self.mae(predictions, y)
        mse = self.mse(predictions, y)
        rmse = self.rmse(predictions, y)
        r2 = self.r2(predictions, y)
        return mae, mse, rmse, r2

    def plot(self):
        plt.plot(np.arange(1, self.steps + 1), self.history, color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close()


class LinearRegressionStochastic:
    def __init__(self, n_features=1, lr=1e-2, reg_type=None, lambda_=0, steps=2000):
        self.n_features = n_features
        self.lr = lr
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.steps = steps
        self.theta = np.random.rand(self.n_features)
        self.theta_history = np.zeros((self.steps, self.theta.shape[0]))
        self.history = []

    def compute_cost(self, error, m):
        cost = None
        if self.reg_type is None:
            cost = 1/(2 * m) * np.sum(error)**2
        if self.reg_type == 'L1':
            cost = 1/(2 * m) * np.sum(error)**2 + (self.lambda_ / (2 * m)) * np.sum(np.abs(self.theta))
        if self.reg_type == 'L2':
            cost = 1/(2 * m) * np.sum(error)**2 + (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta))
        return cost

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        for i in range(self.steps):
            for j in range(m):
                indices = np.random.permutation(m)
                x_shuffle_train = x_train[indices]
                y_shuffle_train = y_train[indices]

                predictions = np.dot(x_shuffle_train[j, :], self.theta)
                error = predictions - y_shuffle_train[j]
                cost = self.compute_cost(error, 1)
                self.history.append(cost)
                reg = 1 - self.lr * self.lambda_
                self.theta = self.theta * reg - self.lr * np.dot(x_shuffle_train[j], error)
                self.theta_history[j, :] = self.theta

    def predict(self, x_test):
        x_pred = np.c_[x_test, np.ones(x_test.shape[0])]
        prediction = np.dot(x_pred, self.theta)
        return prediction

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

    def get_scores(self, prediction, y_test):
        mae = self.mae(prediction, y_test)
        mse = self.mse(prediction, y_test)
        rmse = self.rmse(prediction, y_test)
        r2 = self.r2(prediction, y_test)
        return mae, mse, rmse, r2

    def plot(self, m):
        plt.plot(np.arange(1, self.steps*m + 1), self.history, color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.close()

