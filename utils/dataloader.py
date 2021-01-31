import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        drop = ["PoolQC", "GarageFinish", "GarageYrBlt", "FireplaceQu", "BsmtUnfSF", "Id",
                "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
                "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr"]

        # New column describes sum of area

        self.dataset["YearBuilt"] = 2020 - self.dataset["YearBuilt"]
        self.dataset["YearRemodAdd"] = 2020 - self.dataset["YearRemodAdd"]

        self.dataset = self.dataset.drop(drop, axis=1)

        return self.dataset
