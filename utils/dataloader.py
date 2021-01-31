import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        strcols = ["SaleCondition", "SaleType", "Functional", "KitchenQual",
                   "Electrical", "CentralAir", "BsmtFinType2",
                   "HeatingQC", "Heating", "MiscFeature", "Fence", "PavedDrive",
                   "GarageCond", "GarageQual", "GarageType",
                   "BsmtFinSF2", "BsmtFinType1", "BsmtFinType1", "BsmtExposure",
                   "BsmtCond", "BsmtQual", "Foundation", "ExterCond", "ExterQual",
                   "MasVnrType", "Exterior2nd", "Exterior1st", "RoofMatl",
                   "RoofStyle", "OverallCond", "HouseStyle", "BldgType", "Condition2",
                   "Condition1", "Neighborhood", "LandSlope", "LotConfig",
                   "Utilities", "LandContour", "LotShape", "Alley", "Street",
                   "MSZoning", "MSSubClass"
                   ]

        drop = ["PoolQC", "GarageFinish", "GarageYrBlt", "FireplaceQu", "BsmtUnfSF", "Id",
                "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
                "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr"]

        numcols = ["YrSold", "MiscVal", "MoSold", "PoolArea", "ScreenPorch", "3SsnPorch",
                   "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "GarageArea", "GarageCars",
                   "TotRmsAbvGrd", "Fireplaces", "TotalBsmtSF", "BsmtFinSF1", "MasVnrArea", "OverallQual",
                   "LotArea", "LotFrontage"]

        # New column describes sum of area

        self.dataset["TotalSqr"] = self.dataset["1stFlrSF"] + self.dataset["2ndFlrSF"] + self.dataset["LowQualFinSF"]\
        + self.dataset["GrLivArea"] + self.dataset["BsmtFullBath"] + self.dataset["BsmtHalfBath"]\
        + self.dataset["FullBath"] + self.dataset["HalfBath"] + self.dataset["BedroomAbvGr"]\
        + self.dataset["KitchenAbvGr"]

        self.dataset["YearBuilt"] = 2020 - self.dataset["YearBuilt"]
        self.dataset["YearRemodAdd"] = 2020 - self.dataset["YearRemodAdd"]

        self.dataset = self.dataset.drop(drop, axis=1)

        for column in strcols:
            self.dataset[column] = self.dataset[column].fillna("None")
            self.dataset[column] = LabelEncoder().fit_transform(self.dataset[column])

        for column in numcols:
            self.dataset[column] = self.dataset[column].fillna(np.nanmedian(self.dataset[column]))
        #         dataframe[column] = StandartScaler().fit_transform(dataframe[column])

        return self.dataset