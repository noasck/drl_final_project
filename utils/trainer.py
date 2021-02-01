from sklearn.ensemble import GradientBoostingRegressor
import pickle


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GradientBoostingRegressor(loss='huber', learning_rate=0.11, n_estimators=100, random_state=1).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
