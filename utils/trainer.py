from sklearn.ensemble import GradientBoostingRegressor


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GradientBoostingRegressor(loss='huber', learning_rate=0.11, n_estimators=100).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
