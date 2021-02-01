import requests
import numpy as np
from settings.constants import TRAIN_CSV, VAL_CSV
import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from utils import Predictor
from utils import DataLoader

from utils.dataloader import DataLoader

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



with open('settings/specifications.json') as f:
    specifications = json.load(f)

info = specifications['description']
x_columns, y_column, metrics = info['X'], info['y'], info['metrics']

train_set = pd.read_csv(TRAIN_CSV, header=0)
val_set = pd.read_csv(VAL_CSV, header=0)

train_x, train_y = train_set[x_columns], train_set[y_column]
val_x, val_y = val_set[x_columns], val_set[y_column]

loader = DataLoader()
loader.fit(val_x)
val_processed = loader.load_data()
print('data: ', val_processed[:10]["TotalSqr"])

with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

x_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
y = raw_train["SalePrice"]

# model = GradientBoostingRegressor(loss='huber', learning_rate=0.11, n_estimators=100, random_state=1)
# model.fit(X, y)
# # with open('models/GradientBoostingRegressor_docker.pickle', 'wb')as f:
# #     pickle.dump(model, f)
#
# model_predict = model.predict(val_processed)
# api_score = eval(metrics)(val_y, model_predict)
# print('accuracy trained: ', api_score)


predictor = Predictor()
predict = predictor.predict(val_processed)


api_score = eval(metrics)(val_y, predict)
print('accuracy: ', api_score)


