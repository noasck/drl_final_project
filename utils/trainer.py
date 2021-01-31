from sklearn.linear_model import ElasticNet


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return ElasticNet(alpha=0.4).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
