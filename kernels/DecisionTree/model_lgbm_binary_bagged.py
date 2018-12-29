"""
This is a template for the APIs of models to be used into the stacking framework.
run with Python 3.x
"""
from time import time, ctime
import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import shap
import sys


def sigma_score(preds, valid_data):
    """
    this is a custom metric used to train the model_lgbm_baseline
    """
    df_time = valid_data.params['extra_time'] # will be injected afterwards
    labels = valid_data.get_label()
    
    #    assert len(labels) == len(df_time)

    x_t = preds * labels #  * df_valid['universe'] -> Here we take out the 'universe' term because we already keep only those equals to 1.
    
    # Here we take advantage of the fact that `labels` (used to calculate `x_t`)
    # is a pd.Series and call `group_by`
    x_t_sum = x_t.groupby(df_time).sum()
    score = x_t_sum.mean() / x_t_sum.std()

    return 'sigma_score', score, True

class model():
    """this is a baseline lightLGB model with simple features

    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name):
        self.name             = name
        self.type             = lgb.Booster
        self.model1, self.model2 = None, None
        self.training_results = None
        print("\ninit model {}".format(self.name))
        sys.path.insert(0, '../') # this is for imports from /kernels

    def _generate_features(self, market_data, news_data, verbose=False):
        """
        GENERAL:
        given the original market_data and news_data
        generate new features, doesn't change original data.
        NOTE: data cleaning and preprocessing is not here,
        here is only feats engineering
        
        MODEL SPECIFIC:
        as as a baseline for decision trees model we add
        features that are the most popular among public
        kernels on Kaggle:
        
        - [36] short-term lagged features on returns
        - has been removed (cant pass tests) [6]  long-term moving averages
        - [1]  day of the week

        Args:
            [market_train_df, news_train_df]: pandas.DataFrame
        Returns:
            complete_features: pandas.DataFrame
        """
        from utils import progress
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()
        
        if 'returnsOpenNextMktres10' in complete_features.columns:
            complete_features.drop(['returnsOpenNextMktres10'],axis=1,inplace=True)

        # [36] short-term lagged features on returns
        features = ['returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10'] 
        lags = [3,7,14] 
        progress(0, len(lags)*len(features), prefix = 'Lagged features generation:', length=50)
        for _feature, feature in enumerate(features):
            for _lag, lag in enumerate(lags):
                assetGroups = complete_features.groupby(['assetCode'])

                complete_features['lag_{}_{}_max'.format(lag, feature)] = assetGroups[feature].rolling(lag, min_periods=1).max().reset_index().set_index('level_1').iloc[:, 1].sort_index()

                complete_features['lag_{}_{}_min'.format(lag, feature)] = assetGroups[feature].rolling(lag, min_periods=1).min().reset_index().set_index('level_1').iloc[:, 1].sort_index()

                complete_features['lag_{}_{}_mean'.format(lag, feature)] = assetGroups[feature].rolling(lag, min_periods=1).mean().reset_index().set_index('level_1').iloc[:, 1].sort_index()

                progress(_feature * len(lags) + _lag, len(lags) * len(features), 
                        prefix = 'Lagged features generation:', length = 50)


        self.max_lag = max(lags)
                
        # [1]  day of the week
        #if type(complete_features['time'][0]) == pd._libs.tslibs.timestamps.Timestamp:
        try:    # this is Kaggle environment
            complete_features['weekday'] = complete_features['time'].apply(lambda x: x.dayofweek)
        except: # in test environment 'time' got to converted to str so need formatting
            complete_features['weekday'] = complete_features['time'].apply(lambda x: datetime.strptime(x.split()[0], "%Y-%M-%d").weekday())

            
                
        complete_features.drop(['time','assetCode','assetName'],axis=1,inplace=True)
        complete_features.fillna(0, inplace=True) # TODO: for next models control this fillna with EDA

        if verbose: print("Finished features generation for model {}, TIME {}".format(self.name, time()-start_time))
        return complete_features

    def train(self, X, Y, verbose=False):
        """
        GENERAL:
        basic method to train a model with given data
        model will be inside self.model after training
        
        MODEL SPECIFIC:
        
        - split 0.8 train validation
        - universe filter on validation
        - binary classification
            need to put 'metric':'None' in parameters
        - bagging of two models
        
        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
        Returns:
            (optional) training_results
        """
        start_time = time()
        if verbose: print("Starting training for model {}, {}".format(self.name, ctime()))

            
        time_reference = X[0]['time'] #time is dropped in preprocessing, but is needed later for metrics eval

        X = self._generate_features(X[0], X[1], verbose=verbose)
        Y = Y.clip(Y.quantile(0.001), Y.quantile(0.999))

        # split X in X_train and Y_val
        split = int(len(X) * 0.8)
        test_train_distsance = 0
        X_train, X_val = X[:split - test_train_distsance], X[split:]
        Y_train, Y_val = Y[:split - test_train_distsance], Y[split:]

        if verbose: print("X_train shape {}".format(X_train.shape))
        if verbose: print("X_val shape {}".format(X_val.shape))
        assert X_train.shape[0] != X_val.shape[0]
        assert X_train.shape[1] == X_val.shape[1]

        # universe filtering on validation set
        universe_filter = X['universe'][split:] == 1.0
        X_val = X_val[universe_filter]
        Y_val = Y_val[universe_filter]
        
        # this is a time_val series used to calc the sigma_score later, applied split and universe filter
        time_val = time_reference[split:][universe_filter]
        assert len(time_val) == len(X_val) 
        time_train = time_reference[:split - test_train_distsance]
        assert len(time_train) == len(X_train)
        
        # train parameters prearation
        train_cols = X.columns.tolist()
        assert 'returnsOpenNextMktres10' not in train_cols 
        lgb_train = lgb.Dataset(X_train.values, Y_train, feature_name=train_cols, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val.values, Y_val, feature_name=train_cols, free_raw_data=False)

        lgb_train.params = {
            'extra_time' : time_train.factorize()[0]
        }
        lgb_val.params = {
            'extra_time' : time_val.factorize()[0]
        }

        x_1 = [0.19000424246380565, 2452, 212, 328, 202]
        x_2 = [0.19016805202090095, 2583, 213, 312, 220]

        params_1 = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': x_1[0],
                'num_leaves': x_1[1],
                'min_data_in_leaf': x_1[2],
                'num_iteration': 239,
                'max_bin': x_1[4],
                'verbose': 1
            }

        params_2 = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': x_2[0],
                'num_leaves': x_2[1],
                'min_data_in_leaf': x_2[2],
                'num_iteration': 172,
                'max_bin': x_2[4],
                'verbose': 1
            }

        training_results = {}
        self.model1 = lgb.train(params_1,
                lgb_train,
                num_boost_round=1000,
                valid_sets=(lgb_val,lgb_train),
                valid_names=('valid','train'),
                early_stopping_rounds=15,
                verbose_eval=1,
                evals_result=training_results)

        self.model2 = lgb.train(params_2,
                lgb_train,
                valid_sets=(lgb_val,lgb_train),
                valid_names=('valid','train'),
                num_boost_round=1000,
                verbose_eval=1,
                early_stopping_rounds=15,
                evals_result=training_results)

        del X, X_train, X_val

        if verbose: print("Finished training for model {}, TIME {}".format(self.name, time()-start_time))

        self.training_results = training_results
        return training_results 

    def predict(self, X, verbose=False, do_shap=False):
        """
        given a block of X features gives prediction for everyrow

        Args:
            X: [market_train_df, news_train_df]
            shap: perform shap analysis
        Returns:
            y: pandas.Series
        """
        start_time = time()
        if verbose: print("Starting prediction for model {}, {}".format(self.name, ctime()))
        if self.model1 is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(X[0], X[1], verbose=verbose)
        if verbose: print("X_test shape {}".format(X_test.shape))

        y_test = (self.model1.predict(X_test) + self.model2.predict(X_test))/2
        y_test = (y_test-y_test.min())/(y_test.max()-y_test.min())
        y_test = y_test * 2 - 1

        if do_shap:
            #import pdb;pdb.set_trace()
            print("printing shap analysis..")
            explainer = shap.TreeExplainer(self.model1)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)


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

        processed_historical_df = self._generate_features(historical_df[0], historical_df[1], verbose=verbose)
        X_test = processed_historical_df.iloc[-prediction_length:]
        if verbose: print("X_test shape {}".format(X_test.shape))
        y_test = (self.model1.predict(X_test) + self.model2.predict(X_test))/2
        y_test = (y_test-y_test.min())/(y_test.max()-y_test.min())
        y_test = y_test * 2 - 1

        if verbose: print("Finished rolled prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

    def inspect(self, X):
        """
        visualize and examine the training of the model
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
