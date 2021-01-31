import pickle

from settings.constants import SAVED_ESTIMATOR


class Predictor:
    def __init__(self):
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        return self.loaded_estimator.predict(data)