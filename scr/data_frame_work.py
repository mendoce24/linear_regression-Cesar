import pandas as pd

class DataFrameWork():

    def get_data_frame():
        """Geting data from a csv file

        Returns:
            dataFrame: DataFrame with the information of the csv file
        """        
        path = "../data/_data_clean_to_model.csv"
        immo = pd.read_csv(path, index_col='id')
        return immo
