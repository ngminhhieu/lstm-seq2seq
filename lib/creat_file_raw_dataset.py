from pandas import read_csv
import numpy as np
from constant import LOAD_AREAS

if __name__ == "__main__":
    data = read_csv('data/forecast_data.csv', usecols=['NYISO'])
    # shape (8759,1)
    np.savez('data/forecast_data.npz', data = data)
    