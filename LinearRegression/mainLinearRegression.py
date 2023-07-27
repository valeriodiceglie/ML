from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegressionFullBatch, LinearRegressionStochastic, LinearRegressionMiniBatch

use_sklearn = False

# load dataset
Diabetes = load_diabetes()
df = pd.DataFrame(columns=Diabetes["feature_names"], data=Diabetes["data"])
df["target"] = Diabetes["target"]
print(df.describe())

# shuffle samples
df = df.sample(frac=1).reset_index(drop=True)
x = df[df.columns[:-1]].values
y = df[df.columns[-1:]].values

if use_sklearn:
    [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size=0.25)
else:
    # compute train_index for holdout 80/20
    train_index = round(len(x) * 0.8)

    x_train = x[:train_index]
    y_train = y[:train_index]

    x_test = x[train_index:]
    y_test = y[train_index:]

y_train = np.array(y_train).squeeze()
y_test = np.array(y_test).squeeze()

# normalization
mean = np.mean(x_train, axis=0)
std_dev = np.std(x_train, axis=0)
x_train = (x_train - mean)/std_dev
x_test = (x_test-mean)/std_dev

# add bias to training data
x_train = np.c_[x_train, np.ones(x_train.shape[0])]

# Linear regression using full_batch
model_full_batches = LinearRegressionFullBatch(n_features=x_train.shape[1], lambda_=0.1, reg_type='L2', steps=1000)
model_full_batches.fit(x_train, y_train)

prediction = model_full_batches.predict(x_test)
model_full_batches.plot()
mae_fb, mse_fb, rmse_fb, r2_fb = model_full_batches.get_scores(prediction, y_test)
print('----------SCORES FOR FULL BATCH-------------')
print(f'|\t MAE \t|\t MSE \t|\t RMSE \t|\t R^2 \t|')
print('|----------------------------------------------------------------|')
print(f"|\t {mae_fb:0,.2f} \t|\t {mse_fb:0,.2f} \t|\t {rmse_fb:0,.2f} \t|\t {r2_fb:0,.2f} \t|")

model_stochastic = LinearRegressionStochastic(n_features=x_train.shape[1], reg_type='L2', lambda_=0.1, steps=1000)

model_stochastic.fit(x_train, y_train)

prediction = model_stochastic.predict(x_test)
model_stochastic.plot(x_train.shape[0])

mae_s, mse_s, rmse_s, r2_s = model_stochastic.get_scores(prediction, y_test)
print('-----------SCORES FOR STOCHASTIC---------------')
print(f'|\t MAE \t|\t MSE \t|\t RMSE \t|\t R^2 \t|')
print('|----------------------------------------------------------------|')
print(f"|\t {mae_s:0,.2f} \t|\t {mse_s:0,.2f} \t|\t {rmse_s:0,.2f} \t|\t {r2_s:0,.2f} \t|")

model_mini_batch = LinearRegressionMiniBatch(n_features=x_train.shape[1], reg_type='L2', lambda_=0.1, steps=1000,
                                             num_batch=5)

model_mini_batch.fit(x_train, y_train)

prediction = model_mini_batch.predict(x_test)
model_mini_batch.plot()

mae_mb, mse_mb, rmse_mb, r2_mb = model_mini_batch.get_scores(prediction, y_test)
print('-----------SCORES FOR MINI BATCH---------------')
print(f'|\t MAE \t|\t MSE \t|\t RMSE \t|\t R^2 \t|')
print('|----------------------------------------------------------------|')
print(f"|\t {mae_mb:0,.2f} \t|\t {mse_mb:0,.2f} \t|\t {rmse_mb:0,.2f} \t|\t {r2_mb:0,.2f} \t|")





