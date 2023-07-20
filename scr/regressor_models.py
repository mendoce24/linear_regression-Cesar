from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class RegressorModels:

    def get_X_y(df):
        """Get the target (y) and features (X)

        Args:
            df (DataFrame): Framework with data from immo Eliza

        Returns:
            NumpyArray: Target, Features
        """
        # Drop clumns with not impat in the prediction
        X = pd.DataFrame(df.drop(columns=['price', 'gardenSurface', 'terraceSurface', 'floor']))#features
        y = np.array(df.price).reshape(-1, 1)#target

        return X, y


    def get_train_test(X, y, normalize = False):
        """Split the target and features in training and test data

        Args:
            X (NumpyArray): Feature
            y (NumpyArray): Target
            normalize (bool, optional): Normalize the Features to train and test. Defaults to False.

        Returns:
            NumpyArray: X_train, X_test, y_train, y_test
        """        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

        # validation to normalize or not the Xtrain and Xtest
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


    def get_performance(y, pred):
        """Get the performace of a model prediction

        Args:
            y (NumpyArray): Target
            pred (NumpyArray): Prediction

        Returns:
            Floats: score, mse, rmse, mae
        """        
        score = r2_score(y, pred) # Score
        mse = mean_squared_error(y, pred) # Mean Squared Error
        rmse = mse**0.5 # Root Mean Squared Error
        mae = mean_absolute_error(y, pred) # Mean Absolute Error

        return score, mse, rmse, mae
