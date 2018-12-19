"""
This is a template for the APIs of models to be used into the stacking framework. 
"""
from time import time

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
        from sklearn.linear_model import LinearRegression
        self.type  = LinearRegression
        print("Initialiazing model {}".format(self.name))

    def _generate_features(self, X):
        """
        in place generate additional features from 
        original features X, the method will free 
        all the memory used

        Args:
            X: [market_train_df, news_train_df]
        """

    def train(self, X, Y):
        """
        basic method to train a model with given data
        model will be inside self.model after training

        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
        Returns:
            training_results
        """
        start_time = time()
        print("Starting training for model {}, TIME {}".format(self.name, start_time))
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        print("Finished training for model {}, TIME {}".format(self.name, time()-start_time))


    def predict(self, X):
        """
        given a block of X features gives prediction for everyrow

        Args:
            X: [market_train_df, news_train_df]
        """
        if self.model is None:
            raise "Error: model is not trained!"


    def predict_rolling(self, historical_df, X):
        """
        predict features from X, uses historical for (lagged) feature generation
        to be used with rolling prediciton structure from competition

        Args:
            historical_df: same features as X, previous point of time series
            X: [market_train_df, news_train_df]
        """

