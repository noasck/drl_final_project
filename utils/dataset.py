import pandas as pd
import csv


class Dataset:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def len(self):
        """
        Get number of examples
        @return: int
        """
        with open(self.csv_file, 'rt') as f:
            return sum(1 for row in f) - 1

    def columns(self):
        """
        Get list of columns names
        @return: list
        """
        with open(self.csv_file, 'rt') as f:
            columns = f.readline().rstrip().split(',')
        del columns[1]
        return columns

    def getitem(self, index):
        """
        Get example by index
        @param index: int
        @return: list, int
        """
        'Generates one sample of data'
        # Select sample
        idx = index + 1
        # Load data and get label
        with open(self.csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                if str(idx) in line:
                    break

        y = int(line[1])
        del line[1]
        x = line
        return x, y

    def get_items(self, items_number):
        """
        Get specific amount of examples
        @param items_number:
        @return: pd.DataFrame, pd.Series
        """
        data = pd.read_csv(self.csv_file, nrows=items_number)
        y = data['SalePrice']
        x = data.drop(['SalePrice'], axis=1)
        return x, y