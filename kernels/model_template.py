"""
This is a template for the APIs of models to be used into the stacking framework.
run with Python 3.x
"""
from time import time, ctime
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle as pk
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from datetime import datetime, date
import shap
import sys
import os


class model():
    """
    model description

    FEATURES:

    ISSUES:

    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name):
        self.name             = name
        self.type             = None
        self.model = None
        self.training_results = None
        self.assetCode_mapping = None
        print("\ninit model {}".format(self.name))
        sys.path.insert(0, '../') # this is for imports from /kernels

    def _preprocess(self, market_data):
        """optional data preprocessing
        NOTE: use of this method is DEPRECATED and is only kept
        for backward compatibility
        """
        try:
            market_data = market_data.loc[market_data['time']>=date(2010, 1, 1)]
        except TypeError: # if 'time' is a string value
            print("[_generate_features] 'time' is of type str and not datetime")
            if not market_data.loc[market_data['time']>="2010"].empty:
                # if dates are before 2010 means dataset is for testing
                market_data = market_data.loc[market_data['time']>="2010"]
        assert market_data.empty == False
        return market_data
        

    def _generate_features(self, market_data, news_data):
        """
        GENERAL:
        given the original market_data and news_data
        generate new features, doesn't change original data.
        NOTE: data cleaning and preprocessing is not here,
        here is only feats engineering
        
        MODEL SPECIFIC:

        Args:
            market_train_df: pandas.DataFrame
            news_train_df: pandas.DataFrame

        Returns:
            complete_features: pandas.DataFrame
        """
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()

        if 'returnsOpenNextMktres10' in complete_features.columns:
            complete_features.drop(['returnsOpenNextMktres10'],axis=1,inplace=True)


        # generate features here..


        if verbose: print("Finished features generation for model {}, TIME {}".format(self.name, time()-start_time))
        return complete_features

    def _generate_target(self, Y):
        """
        given Y generate binary labels
        returns:
            up, r : (binary labels), (returns)
        """
        binary_labels = Y >= 0
        return binary_labels.astype(int).values, Y.values

    def train(self, X, Y, verbose=False, load=True):
        """
        GENERAL:
        basic method to train a model with given data
        model will be inside self.model after training
        
        MODEL SPECIFIC:
        
        
        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
            load: load model if possible instead of training
        Returns:
            (optional) training_results
        """

        start_time = time()
        if verbose: print("Starting training for model {}, {}".format(self.name, ctime()))
            
        time_reference = X[0]['time'] #time is dropped in preprocessing, but is needed later for metrics eval
        universe_reference = X[0]['universe']

        X = self._generate_features(X[0], X[1], verbose=verbose, normalize=normalize, normalize_vals=normalize_vals)
        binary_Y, Y = self._generate_target(Y)

        try:
            assert X.shape[0] == binary_Y.shape[0] == Y.shape[0]
        except AssertionError:
            import pdb;pdb.set_trace()
            pass


        # training code here..


        if load:
            try:
                self._load()
                print("#"*30)
                print("\n[WARNING] TRAINING SKIPPED, MODEL LOADED FROM MEMORY\n")
                print("[INFO] if you want to avoid skipping training, change model name")
                print("#"*30)
                return
            except:
                print("Tried to load model but didn't find any")
                pass

        del X, X_train, X_val

        if verbose: print("Finished training for model {}, TIME {}".format(self.name, time()-start_time))

        if load:
            try:
                self._save()
            except:
                print("[train] WARNING: couldn't save the model")

        return None #training results


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

        X_test = self._generate_features(X[0], X[1], verbose=verbose)
        if verbose: print("X_test shape {}".format(X_test.shape))

        # predict code here..
        y_test = []

        if verbose: print("Finished prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

    def predict_rolling(self, historical_df, market_obs_df, verbose=False):
        """
        predict features from X, uses historical for (lagged) feature generation
        to be used with rolling prediciton structure from competition

        Args:
            historical_df: [market_train_df, news_train_df]
            market_obs_df: from rolling prediction generator
        """
        start_time = time()
        if verbose: print("Starting rolled prediction for model {}, {}".format(self.name, ctime()))
        if self.model1 is None or self.model2 is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(historical_df[0], historical_df[1], verbose=verbose, normalize=normalize, normalize_vals=normalize_vals, output_len=len(market_obs_df))
        X_test.reset_index(drop=True,inplace=True)
        if verbose: print("X_test shape {}".format(X_test.shape))

        # prediction code here..


        y_test = None

        if verbose: print("Finished rolled prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test


    def inspect(self, X):
        """
        visualize and examine the training of the model
        ONLY FOR GRADIENT BOOSTED TREES
        Args:
            X: for the shap values

        MODEL SPECIFIC:
        plots training results and feature importance
        """
        if not self.training_results:
            print("Error: No training results available")
        else:
            print("printing training results..")
            for _label, key in self.training_results.items():
                for label, result in key.items():
                    plt.plot(result,label=_label+" "+label)
            plt.title("Training results")
            plt.legend()
            plt.show()

        if not self.model1:
            print("Error: No model available")
        else:
            print("printing feature importance..")
            f=lgb.plot_importance(self.model1)
            f.figure.set_size_inches(10, 30) 
            plt.show()

    def _postprocess(self, predictions, normalize=True):
        """
        post processing of predictions

        Args:
            predictions: list(np.array) might be from
                different models
        Return:
            predictions: np.array

        MODEL SPECIFIC:
        the postprocessing is needed to ensemble bagged
        models and to map prediction interval from [0, 1] 
        to [-1, 1]
        """
        y_test = sum(predictions)/len(predictions)
        if normalize:
            y_test = (y_test-y_test.min())/(y_test.max()-y_test.min())
        y_test = y_test * 2 - 1
        return y_test

    def _clean_data(self, data):
        """
        originally from function mis_impute in
        https://www.kaggle.com/guowenrui/sigma-eda-versionnew

        Args:
            data: pd.DataFrame
        returns:
            cleaned data (not in place)
        """
        for i in data.columns:
            if data[i].dtype == "object":
                    data[i] = data[i].fillna("other")
            elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
                    data[i] = data[i].fillna(data[i].mean())
                    # I am just filling the mean of all stocks together?
                    # should fill with the mean of the singular stock
            else:
                    pass
        return data

    def _save(self):
        """
        save models to memory into pickle/self.name

        RaiseException: if can't save
        """
        to_save = None
        # save code here..
        import pdb;pdb.set_trace() 
        
        if not all(to_save):
            print("[_save] Error: not all models are trained")
            print(to_save)
        else:
            try: #try to save in different folders based on where we are
                save_name = os.path.join("../pickle",self.name+"_.pkl")
                with open(save_name,"wb") as f:
                    pk.dump(to_save, f)
                    print("[_save] saved models to "+save_name)
            except: 
                save_name = os.path.join("pickle",self.name+"_.pkl")
                with open(save_name,"wb") as f:
                    pk.dump(to_save, f)
                    print("[_save] saved models to "+save_name)

    def _load(self):
        """
        load models to memory from pickle/self.name

        RaiseExcpetion: can't find model
        """
        save_name = os.path.join("../pickle",self.name)+"_.pkl"
        save_name_attempt = os.path.join("pickle",self.name)+"_.pkl"
        try:
            with open(save_name,"rb") as f:
                models = pk.load(f)
        except:
            with open(save_name_attempt,"rb") as f:
                models = pk.load(f)

        # load code here..
        import pdb;pdb.set_trace() 
        print("[_load] models loaded succesfully")

