"""
This is a template for the APIs of models to be used into the stacking framework. 
"""
from time import time, ctime
from sklearn.linear_model import LinearRegression
import pandas as pd

class model_example():
    """base class for the model
    
    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name):
        self.name  = name
        self.model = None
        self.type  = LinearRegression
        print("\ninit model {}".format(self.name))

    def _generate_features(self, market_data, news_data, verbose=False):
        """
        given the original market_data and news_data
        generate new features, doesn't change original data.
        NOTE: data cleaning and preprocessing is not here,
        here is only feats engineering

        Args:
            [market_train_df, news_train_df]: pandas.DataFrame
        Returns:
            complete_features: pandas.DataFrame
        """
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()
        complete_features['open+close'] = complete_features['open'] + complete_features['close']
        complete_features.drop(['time','assetCode','assetName'],axis=1,inplace=True)
        complete_features.fillna(0, inplace=True)

        if verbose: print("Finished features generation for model {}, TIME {}".format(self.name, time()-start_time))
        return complete_features

    def train(self, X, Y, verbose=False):
        """
        basic method to train a model with given data
        model will be inside self.model after training

        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
        Returns:
            (optional) training_results
        """
        start_time = time()
        if verbose: print("Starting training for model {}, {}".format(self.name, ctime()))

        X_train = self._generate_features(X[0], X[1])
        if verbose: print("X_train shape {}".format(X_train.shape))
        self.model = LinearRegression()
        self.model.fit(X_train, Y) 
        del X_train

        if verbose: print("Finished training for model {}, TIME {}".format(self.name, time()-start_time))


    def predict(self, X, verbose=False):
        """
        given a block of X features gives prediction for everyrow

        Args:
            X: [market_train_df, news_train_df]
        Returns:
            y: pandas.Series
        """
        start_time = time()
        if verbose: print("Starting prediction for model {}, {}".format(self.name, ctime()))
        if self.model is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(X[0], X[1])
        if verbose: print("X_test shape {}".format(X_test.shape))
        y_test = self.model.predict(X_test)

        if verbose: print("Finished prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test


    def predict_rolling(self, historical_df, prediction_length, verbose=False):
        """
        predict features from X, uses historical for (lagged) feature generation
        to be used with rolling prediciton structure from competition

        Args:
            historical_df: [market_train_df, news_train_df]
            prediction_length: generate features on historical_df, predict only on the last rows 
        """
        start_time = time()
        if verbose: print("Starting rolled prediction for model {}, {}".format(self.name, ctime()))

        processed_historical_df = self._generate_features(historical_df[0], historical_df[1])
        X_test = processed_historical_df.iloc[-prediction_length:]
        if verbose: print("X_test shape {}".format(X_test.shape))
        y_test = self.model.predict(X_test)

        if verbose: print("Finished rolled prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

