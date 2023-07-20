import pandas as pd

class DataFrameWork():

    def get_data_frame():
        path = "../data/_data_clean_to_model.csv"
        immo = pd.read_csv(path, index_col='id')
        return immo
