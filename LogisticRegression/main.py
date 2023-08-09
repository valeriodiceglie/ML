from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from LogisticRegression import LogisticRegression

dataset = make_classification()

x, y = make_classification(n_features=4)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Normalization
mean = np.mean(x_train, axis=0)
std_dev = np.std(x_train, axis=0)
x_train = (x_train - mean)/std_dev
x_test = (x_test-mean)/std_dev

log_reg = LogisticRegression(n_features=4)
model = log_reg.fit(x_train, y_train)

pred = log_reg.predict(x_test)

#Let's see the f1-score for training and testing data
f1_score = log_reg.f1_score(pred, y_test)

print(f1_score)
