from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class RegressorModels:

    def get_X_y(df):
        X = pd.DataFrame(df.drop(columns=['price', 'gardenSurface', 'terraceSurface', 'floor']))
        y = np.array(df.price).reshape(-1, 1)#target

        return X, y


    def get_train_test(X, y, normalize = False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

        if normalize:
            scalar = StandardScaler()

            X_train = np.array(pd.DataFrame(
                scalar.fit_transform(X_train),
                columns = X_train.columns
            ))

            X_test = np.array(pd.DataFrame(
                scalar.transform(X_test),
                columns = X_test.columns
            ))
        
        return X_train, X_test, y_train, y_test


    def get_performance(y_train, pred_train):
        score = r2_score(y_train, pred_train)
        mse = mean_squared_error(y_train, pred_train)
        rmse = mse**0.5
        mae = mean_absolute_error(y_train, pred_train)

        return score, mse, rmse, mae
