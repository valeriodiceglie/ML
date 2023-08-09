import numpy as np
class LogisticRegression:

    def __init__(self, n_features=1, learning_rate=1e-3, steps=1000):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.steps = steps
        self.theta = np.random.rand(self.n_features + 1)
        self.theta_history = np.zeros((self.steps, self.theta.shape[0]))
        self.history = []

    def sigmoid(self, z):
        return 1/(1 + np.e**(-z))

    def compute_cost(self, m, x, y):
        z = np.dot(x, self.theta)
        cost = -1/m * (y.T.dot(np.log(self.sigmoid(z))) + (1 - y).T.dot(np.log(1 - self.sigmoid(z))))
        return cost

    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        for i in range(self.steps):
            pred = np.dot(x_train, self.theta)
            error = pred - y_train
            cost = self.compute_cost(m, x_train, y_train)
            self.history.append(cost)
            self.theta_history[i, :] = self.theta
            self.theta = self.theta - self.learning_rate/m * np.dot(x_train.T, error)

    def predict(self, x_test):
        x_test = np.c_[x_test, np.ones(x_test.shape[0])]
        pred = []
        z = np.dot(x_test, self.theta)
        for i in self.sigmoid(z):
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def f1_score(self, pred, y):
        tp,fp,tn,fn = 0,0,0,0
        for i in range(len(y)):
            if pred[i] == y[i] == 1:
                tp+=1
            elif pred[i] == y[i] == 0:
                tn+=1
            elif pred[i] == 1 and pred[i] != y[i]:
                fp+=1
            elif pred[i] == 0 and pred[i] != y[i]:
                fn+=1
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    
