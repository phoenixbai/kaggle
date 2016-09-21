
import pandas as pd

class Preprocessor:

    def load_dataset(self, filename):
        """

        :param filename:
        :param has_header
        :return:
        """
        data = pd.read_csv(filename, header=0)
        shape = data.shape
        rows = shape[0]
        cols = shape[1]
        target = data.iloc[cols-1]
        x = data.iloc[0:cols-1]

        return (x, target)

