import pickle
import numpy as np
import sys
sys.path.insert(0, sys.path[0]+"/../")
from models.ensemble_model import ReweightModel, PerfomanceBasedModel

np.random.seed(0)

# load model

with open('output/ensemble_model/sjtumodel.pkl', 'rb') as f:
    model = pickle.load(f)

# generate data

'''
Model parameters:

model.train( x_train: np.array, y_train: np.array ) -> None:
    x_train: n rows of past predictions from m models, shape = (n, m)
    y_train: corresponding true values for x_train, shape = (n, 1)
    (n should be at least 2000, recommended to be greater than 10000.)

model(x_test: np.array) -> y_predict: np.array :
    x_test: predictions of the models to be ensembled, shape = (n', m)
    y_predict: ensembled prediction results of the models, shape = (n', 1)

model.predict( x_train: np.array, y_train: np.array, x_test: np.array, y_predict: np.array, return_interval: int, max_retrain_samples: int, progress_bar: Bool) -> y_predict: np.array :
    retrain_interval (defalt None): This parameter represents how often the model is retrained, based on the number of data points. For example, when using the CSI300 dataset, specifying retrain_interval=300 means retraining the model every 300 data points, which corresponds to retraining the model daily. Set to None to not retrain the model. It is recommended to retrain the model every 1-2 trading days.
    
    max_retrain_samples (defalt None): This parameter represents the maximum number of historical data used when retraining model. Setting to None represents use len(x_train) as the maximum number, set to -1 to use all historical data for each retraining of the model.
    
    progress_bar (default False): whether to show progress bar

'''

n = 100000
n_ = 50000
m = 5

x_train = np.random.randn(n, m)
y_train = np.random.randn(n, 1)
x_test = np.random.randn(n_, m)
y_test = np.random.randn(n_, 1)

# train and predict without retrain

model.train(x_train, y_train)
y_predict = model(x_test)
print(y_predict.shape)

# Another way to train and predict without retrain
y_predict1 = model.predict(x_train, y_train, x_test, y_test)
# this asert that those two methods have the same output
print(np.all(y_predict1 == y_predict))

# predict with retrain
y_predict = model.predict(x_train, y_train, x_test, y_test, retrain_interval=300, max_retrain_samples=-1, progress_bar=True)
print(y_predict.shape)


'''compare with baselines'''

# generate ground-truth data
y = np.random.randn(n + n_, 1)
# generate perturbed prediction results
x = np.hstack([np.random.randn(n + n_, 1) * 0.1 * np.random.rand() + y for _ in range(10)])

x_train = x[:n]
y_train = y[:n]
x_test = x[n:]
y_predict = y[n:]

model.train(x_train, y_train)
pbm = PerfomanceBasedModel()
pbm.train(x_train, y_train)

print(f"Average MSE: {((y_predict - x_test)**2).mean():.2e}")
print(f"Average ensemble MSE: {((y_predict - x_test).mean(axis=1)**2).mean():.2e}")
print(f"RPerformance based model MSE: {((y_predict - pbm(x_test))**2).mean():.2e}")
print(f"Reweight model MSE: {((y_predict - model(x_test))**2).mean():.2e}")