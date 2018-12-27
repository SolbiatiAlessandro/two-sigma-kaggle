"""
This is a template for the APIs of models to be used into the stacking framework.
run with Python 3.x
"""
from time import time, ctime
import lightgbm as lgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from datetime import datetime
import shap


def sigma_score(preds, valid_data):
    """
    this is a custom metric used to train the model_lgbm_baseline
    """
    #TODO: put debugger here and try to figure out why sigma_score not computed
    df_time = valid_data.params['extra_time'] # will be injected afterwards
    labels = valid_data.get_label()
    
    #    assert len(labels) == len(df_time)

    x_t = preds * labels #  * df_valid['universe'] -> Here we take out the 'universe' term because we already keep only those equals to 1.
    
    # Here we take advantage of the fact that `labels` (used to calculate `x_t`)
    # is a pd.Series and call `group_by`
    x_t_sum = x_t.groupby(df_time).sum()
    score = x_t_sum.mean() / x_t_sum.std()

    return 'sigma_score', score, True

class model_lgbm():
    """this is a replica of the original non-standardized
    lgbm model that get 0.63109 (id=5 in history.html)

    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name):
        self.name             = name
        self.model            = None
        self.type             = lgb.Booster
        self.training_results = None
        print("\ninit model {}".format(self.name))

    def _generate_features(self, market_data, news_data, verbose=False):
        """
        GENERAL:
        given the original market_data and news_data
        generate new features, doesn't change original data.
        NOTE: data cleaning and preprocessing is not here,
        here is only feats engineering
        
        MODEL SPECIFIC:
        the feats of the model are

        - assetName_mean_close, assetName_mean_open, close_to_open
        - lagged feats on periods 3,7,14

        the code is not super clear since I am copy pasting
        from the non standardized. now the goal is only
        replicate

        for full dataset takes TIME 337.8714208602905

        Args:
            [market_train_df, news_train_df]: pandas.DataFrame
        Returns:
            complete_features: pandas.DataFrame
        """
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()
        
        if 'returnsOpenNextMktres10' in complete_features.columns:
            complete_features.drop(['returnsOpenNextMktres10'],axis=1,inplace=True)


        # [21] short-term lagged features on returns
        return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']

        def create_lag(df_code,n_lag=[3,7,14,],shift_size=1):
            """internal util of eda 67"""
            code = df_code['assetCode'].unique()
            for col in return_features:
                for window in n_lag:
                    rolled = df_code[col].shift(shift_size).rolling(window=window)
                    lag_mean = rolled.mean()
                    lag_max = rolled.max()
                    lag_min = rolled.min()
                    #lag_std = rolled.std()
                    df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
                    df_code['%s_lag_%s_max'%(col,window)] = lag_max
                    df_code['%s_lag_%s_min'%(col,window)] = lag_min
                    #df_code['%s_lag_%s_std'%(col,window)] = lag_std
            return df_code.fillna(-1)

        def generate_lag_features(df,n_lag = [3,7,14]):
            """internal util of eda 67"""
            features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
               'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
               'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
               'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
               'returnsOpenNextMktres10', 'universe']
            
            assetCodes = df['assetCode'].unique()
            print(assetCodes)
            all_df = []
            df_codes = df.groupby('assetCode')
            df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
            print('total %s df'%len(df_codes))
            
            #TODO: fix multiprocess
            """Can't use multiprocess, got:
            AttributeError: Can't pickle local object 'model_lgbm._generate_features.<locals>.create_lag'

            pool = Pool(4)
            all_df = pool.map(create_lag, df_codes)
            
            new_df = pd.concat(all_df)  
            
            new_df.drop(return_features,axis=1,inplace=True)
            pool.close()
            """

            all_df = [create_lag(single_asset) for single_asset in df_codes]
            new_df = pd.concat(all_df)
            new_df.drop(return_features,axis=1,inplace=True)
            return new_df

        n_lag = [3,7,14]
        self.max_lag = 14
        new_df = generate_lag_features(complete_features,n_lag)
        complete_features = pd.merge(complete_features,new_df,how='left',on=['time','assetCode'])



        # [3] asset features
        complete_features['close_to_open'] =  np.abs(complete_features['close'] / complete_features['open'])
        complete_features['assetName_mean_open'] = complete_features.groupby('assetName')['open'].transform('mean')
        complete_features['assetName_mean_close'] = complete_features.groupby('assetName')['close'].transform('mean')

                
        complete_features.drop(['time','assetCode','assetName'],axis=1,inplace=True)

        def mis_impute(data):
            """this is a fillna util from eda 67"""
            for i in data.columns:
                if data[i].dtype == "object":
                    data[i] = data[i].fillna("other")
                elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
                    data[i] = data[i].fillna(data[i].mean())
                else:
                    pass
            return data

        complete_features = mis_impute(complete_features)

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
        - custom metric used (sigma_scored) , 
            need to put 'metric':'None' in parameters
        - one single lgbm with params_1 from script 67
        
        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
        Returns:
            (optional) training_results
        """
        start_time = time()
        if verbose: print("Starting training for model {}, {}".format(self.name, ctime()))

            
        time_reference = X[0]['time'].reset_index(drop=True) #time is dropped in preprocessing, but is needed later for metrics eval

        X = self._generate_features(X[0], X[1], verbose=verbose).reset_index(drop=True)
        Y = Y.clip(Y.quantile(0.001), Y.quantile(0.999)).reset_index(drop=True)

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
        import pdb;pdb.set_trace()
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
        #this is from eda script 67
        lgb_params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression_l1',
        #         'objective': 'regression',
                'learning_rate': x_1[0],
                'num_leaves': x_1[1],
                'min_data_in_leaf': x_1[2],
        #         'num_iteration': x_1[3],
                'num_iteration': 239,
                'max_bin': x_1[4],
                'verbose': 1,
                'lambda_l1': 0.0,
                'lambda_l2' : 1.0,
                'metric':'None'
        }
        
        training_results = {}
        self.model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=(lgb_val,lgb_train), 
                valid_names=('valid','train'), 
                verbose_eval=25,
                early_stopping_rounds=10,
                feval=sigma_score,
                evals_result=training_results
                )
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
        if self.model is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(X[0], X[1], verbose=verbose)
        if verbose: print("X_test shape {}".format(X_test.shape))
        y_test = self.model.predict(X_test)

        if do_shap:
            #import pdb;pdb.set_trace()
            print("printing shap analysis..")
            explainer = shap.TreeExplainer(self.model)
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
        y_test = self.model.predict(X_test)

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

        if not self.model:
            print("Error: No model available")
        else:
            print("printing feature importance..")
            f=lgb.plot_importance(self.model)
            f.figure.set_size_inches(10, 30) 
            plt.show()


