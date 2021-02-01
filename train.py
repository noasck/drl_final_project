import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV


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

model = GradientBoostingRegressor(loss='huber', learning_rate=0.11, n_estimators=100, random_state=1)
model.fit(X, y)
with open('models/GradientBoostingRegressor.pickle', 'wb')as f:
    pickle.dump(model, f)
