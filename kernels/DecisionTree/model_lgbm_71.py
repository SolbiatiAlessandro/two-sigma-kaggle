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
    """this is a bagged lightLGBM model as STANDARD BENCHMARK
    it is the highest scoring STANDARD model (0.704) on public validation
    at date of commit #2f9beb6, some of the features are

    FEATURES:
    - 6 bagged model tuned with hyperopt
    - leaked training (check lgbm_71_no_leak for clean model)
    - global assetMapping
    - global standardization values
    - _load and _save APIs

    ISSUES:
    - the method .predict and .predict_rolling yields different
    results on private validation (0.44, 0.49)
    - private validation is significantly lower than public

    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name):
        self.name             = name
        self.type             = lgb.Booster
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None
        self.model5 = None
        self.model6 = None
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
        

    def _generate_features(self, market_data, news_data, verbose=False, normalize=True, normalize_vals=[None], output_len = None):
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
            normalize: (bool) 
            normalize_vals: None or [maxs, mins], normalize with local vals or with given vals
            unique_assetCodess: list(str),for mapping assetCodeT

        Returns:
            complete_features: pandas.DataFrame
        """
        #from utils import progress
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()

        if 'returnsOpenNextMktres10' in complete_features.columns:
            complete_features.drop(['returnsOpenNextMktres10'],axis=1,inplace=True)

        #### [36] short-term lagged features on returns ####
          

        def create_lag(df_code,n_lag=[3,7,14,],shift_size=1):
            code = df_code['assetCode'].unique()
            
            # commented code in this function was nice legacy code to print progress of generating lagged features
            # should be fixed after competition for completeness, it broke when multiprocessing for implemented

            # progress(0, len(n_lag)*len(return_features), prefix = 'Lagged features generation:', length=50)
            # print("\rcreating lags for {}".format(code))
            for _feature, col in enumerate(return_features):
                for _lag, window in enumerate(n_lag):
                    rolled = df_code[col].shift(shift_size).rolling(window=window)
                    lag_mean = rolled.mean()
                    lag_max = rolled.max()
                    lag_min = rolled.min()
                    lag_std = rolled.std()
                    df_code['lag_%s_%s_mean'%(window,col)] = lag_mean
                    df_code['lag_%s_%s_max'%(window,col)] = lag_max
                    df_code['lag_%s_%s_min'%(window,col)] = lag_min
                    # progress(_feature * len(n_lag) + _lag, len(n_lag) * len(return_features), 
                    # prefix = 'Lagged features generation:', length = 50)
            return df_code.fillna(-1)

        def generate_lag_features(df,n_lag = [3,7,14]):
            """
            NOTE: most of this (ugly) internal functions are copy pasted from the famous
            qianqian kernel 'eda 67', should be all refactored
            """
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

            pool = Pool(4)
            all_df = pool.map(create_lag, df_codes)
            
            new_df = pd.concat(all_df)  
            new_df.drop(return_features,axis=1,inplace=True)
            pool.close()

            # for the next two lines
            # https://stackoverflow.com/questions/49888485/pathos-multiprocessings-pool-appears-to-be-nonlocal
            pool.terminate()
            pool.restart()
            
            return new_df

        return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
        n_lag = [3,7,14]
        new_df = generate_lag_features(complete_features,n_lag=n_lag)
        new_df['time'] = pd.to_datetime(new_df['time'])
        complete_features['time'] = pd.to_datetime(complete_features['time'])
        complete_features = pd.merge(complete_features,new_df,how='left',on=['time','assetCode'])
        self.max_lag = max(n_lag)

        if output_len is not None:
            # this is used during rolling predictions, where were
            # we need only the last len(market_obs_df) rows
            complete_features = complete_features[-output_len:]

        complete_features = self._clean_data(complete_features)

        #### [1]  generate labels encoding for assetCode ####

        # this sets a universal mapping for assetCodes
        # didn't verify it doesn't raise KeyError for new assetCodes
        # not encountered during training phase

        # TODO: check whether previous version of the model
        # not including this feature fail on the relative test

        if self.assetCode_mapping is None:
            self.assetCode_mapping = {asset_code: mapped_value\
                    for mapped_value, asset_code in\
                    enumerate(complete_features['assetCode'].unique())}

        complete_features['assetCodeT'] = complete_features['assetCode'].map(self.assetCode_mapping).dropna(axis = 0)

        #### drop columns ####

        fcol = [c for c in complete_features if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                                         'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                                                                                      'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
        complete_features = complete_features[fcol]


        #### normalization of input ####

        if normalize:
            if len(normalize_vals) == 1:
                mins = np.min(complete_features, axis=0)
                maxs = np.max(complete_features, axis=0)
                self.mins = mins #saved for prediction phase
                self.maxs = maxs #saved for prediction phase
                rng = maxs - mins
                complete_features = 1 - ((maxs - complete_features) / rng)
            else:
                # if method was called with arbitrary normalize values
                # (because in prediction phase)
                mins = normalize_vals[1]
                maxs = normalize_vals[0]
                rng = maxs - mins
                complete_features = 1 - ((maxs - complete_features) / rng)


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

    def train(self, X, Y, verbose=False, normalize=True, normalize_vals=[None], load=True):
        """
        GENERAL:
        basic method to train a model with given data
        model will be inside self.model after training
        
        MODEL SPECIFIC:
        
        - sklearn random split
        - universe filter on validation
        - binary classification
            need to put 'metric':'None' in parameters
        - target is Y > 0 
        
        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
            normalize: (bool)
            normalize_vals: recommmended self.maxs, self.mins
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

        from sklearn import model_selection
        X_train, X_val,\
        binary_Y_train, binary_Y_val,\
        Y_train, Y_val,\
        universe_train, universe_val,\
        time_train, time_val = model_selection.train_test_split(
                X, 
                binary_Y,
                Y,
                universe_reference.values,
                time_reference, test_size=0.25, random_state=99)

        assert X_train.shape[0] == Y_train.shape[0] == binary_Y_train.shape[0]

        if verbose: print("X_train shape {}".format(X_train.shape))
        if verbose: print("X_val shape {}".format(X_val.shape))
        assert X_train.shape[0] != X_val.shape[0]
        assert X_train.shape[1] == X_val.shape[1]

        # train parameters prearation
        train_cols = X.columns.tolist()
        assert 'returnsOpenNextMktres10' not in train_cols 
        train_data = lgb.Dataset(X.values, binary_Y, feature_name=train_cols)
        test_data = lgb.Dataset(X_val.values, binary_Y_val, feature_name=train_cols)

        x_1 = [0.19000424246380565, 2452, 212, 328, 202]
        x_2 = [0.19016805202090095, 2583, 213, 312, 220]
        x_3 = [0.19564034613157152, 2452, 210, 160, 219]
        x_4 = [0.19016805202090095, 2500, 213, 150, 202]
        x_5 = [0.19000424246380565, 2600, 215, 140, 220]
        x_6 = [0.19000424246380565, 2652, 216, 152, 202]

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


        params_3 = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': x_3[0],
                'num_leaves': x_3[1],
                'min_data_in_leaf': x_3[2],
                'num_iteration': x_3[3],
                'max_bin': x_3[4],
                'verbose': 1
            }

        params_4 = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': x_4[0],
                'num_leaves': x_4[1],
                'min_data_in_leaf': x_4[2],
                'num_iteration': x_4[3],
                'max_bin': x_4[4],
                'verbose': 1
            }

        params_5 = {
                'task': 'train',
                'boosting_type': 'gbdt',#dart
                # what is this 'dart'? added by GuoWenRui
                'objective': 'binary',
                'learning_rate': x_5[0],
                'num_leaves': x_5[1],
                'min_data_in_leaf': x_5[2],
                'num_iteration': x_5[3],
                'max_bin': x_5[4],
                'verbose': 1
            }

        params_6 = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': x_6[0],
                'num_leaves': x_6[1],
                'min_data_in_leaf': x_6[2],
                'num_iteration': x_6[3],
                'max_bin': x_6[4],
                'verbose': 1
            }

        if load:
            # training might be extremely long, try to load when possible
            # NOTE: saving a model is about 300MB
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

        training_results = {}
        self.model1 = lgb.train(params_1,
                train_data,
                num_boost_round=100,
                valid_sets=(test_data, train_data),
                valid_names=('valid','train'),
                early_stopping_rounds=5,
                verbose_eval=1,
                evals_result=training_results)

        self.model2 = lgb.train(params_2,
                train_data,
                valid_sets=(test_data, train_data),
                valid_names=('valid','train'),
                num_boost_round=100,
                verbose_eval=1,
                early_stopping_rounds=5,
                evals_result=training_results)


        self.model3 = lgb.train(params_3,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=5,
                )

        self.model4 = lgb.train(params_4,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=5,
                )

        self.model5 = lgb.train(params_5,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=5,
                )


        self.model6 = lgb.train(params_6,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=10,
                )

        del X, X_train, X_val

        if verbose: print("Finished training for model {}, TIME {}".format(self.name, time()-start_time))


        try:
            self._save()
        except:
            print("[train] WARNING: couldn't save the model")

        self.training_results = training_results
        return training_results 

    def predict(self, X, verbose=False, do_shap=False, normalize=True, normalize_vals = [None]):
        """
        given a block of X features gives prediction for everyrow+".pkl"

        (commit #2f9beb6)
        ISSUE: the method is currently not accurate and doesn't yield
        same predictions as predict_rolling (official benchmark), thus
        the use in stacking is discouraged, use instead the (temporary)
        method .predict_accurate

        Args:
            X: [market_train_df, news_train_df]
            do_shap: perform shap analysis [DEPRECATED]
            normalize: (bool)
            normalize_vals: recommmended self.maxs, self.mins
        Returns:
            y: pandas.Series
        """
        start_time = time()
        if verbose: print("Starting prediction for model {}, {}".format(self.name, ctime()))
        if self.model1 is None or self.model2 is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(X[0], X[1], verbose=verbose, normalize=normalize, normalize_vals=normalize_vals)
        if verbose: print("X_test shape {}".format(X_test.shape))
        preds= [self.model1.predict(X_test), self.model2.predict(X_test)]
        preds.append(self.model3.predict(X_test))
        preds.append(self.model4.predict(X_test))
        preds.append(self.model5.predict(X_test))
        preds.append(self.model6.predict(X_test))
        y_test = self._postprocess(preds, normalize=False)

        if do_shap:
            #import pdb;pdb.set_trace()
            print("printing shap analysis..")
            explainer = shap.TreeExplainer(self.model1)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)

        if verbose: print("Finished prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

    def predict_rolling(self, historical_df, market_obs_df, verbose=False, normalize=True, normalize_vals=[None]):
        """
        predict features from X, uses historical for (lagged) feature generation
        to be used with rolling prediciton structure from competition

        Args:
            historical_df: [market_train_df, news_train_df]
            market_obs_df: from rolling prediction generator
            normalize: (bool)
            normalize_vals: recommmended self.maxs, self.mins
        """
        start_time = time()
        if verbose: print("Starting rolled prediction for model {}, {}".format(self.name, ctime()))
        if self.model1 is None or self.model2 is None:
            raise "Error: model is not trained!"

        X_test = self._generate_features(historical_df[0], historical_df[1], verbose=verbose, normalize=normalize, normalize_vals=normalize_vals, output_len=len(market_obs_df))
        X_test.reset_index(drop=True,inplace=True)
        if verbose: print("X_test shape {}".format(X_test.shape))
        preds= []
        preds.append(self.model1.predict(X_test))
        preds.append(self.model2.predict(X_test))
        preds.append(self.model3.predict(X_test))
        preds.append(self.model4.predict(X_test))
        preds.append(self.model5.predict(X_test))
        preds.append(self.model6.predict(X_test))
        y_test = self._postprocess(preds)

        if verbose: print("Finished rolled prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

    def predict_accurate(self, market_test_df, verbose=False):
        """
        this is a temporary substitute method for self.predict, is slower
        since uses rolling predictions but accurate 

        Args:
            market_test_df

        Attributes (required):
          self.predict_rolling: instance of model class defined above
          self.maxs, self.mins: (pd.DataFrame)
          self.max_lag: (int)

        NOTE:
        the method is NOT unit tested, it was just run and
        didn't break. It was copy pasted from a verified code.
        Also testing this method is not in scope.
        """
        from time import time
        if verbose: print("Starting predict_accurate for model {}, {}".format(self.name, ctime()))
        start_time = time()

        # the following 9 lines are a simulation of two-sigma
        # prediction 'rolling' framework
        PREDICTIONS, days = [], []
        for date in market_test_df['time'].unique():
            market_obs_df = market_test_df[market_test_df['time'] == date].drop(['returnsOpenNextMktres10','universe'],axis=1)
            predictions_template_df = pd.DataFrame({'assetCode':market_test_df[market_test_df['time'] == date]['assetCode'],
                                                    'confidenceValue':0.0})
            days.append([market_obs_df,None,predictions_template_df])
        
        # the following 25~ lines are the official benchmark
        # code for rolling predictions
        n_days, prep_time, prediction_time, packaging_time = 0, 0, 0, 0
        total_market_obs_df = []
        for (market_obs_df, news_obs_df, predictions_template_df) in days:
            n_days +=1
            if (n_days%50==0): print(n_days,end=' ')
            t = time()
            try:
                market_obs_df['time'] = market_obs_df['time'].dt.date
            except:
                pass

            total_market_obs_df.append(market_obs_df)
            if len(total_market_obs_df) == 1:
                history_df = total_market_obs_df[0]
            else:
                history_df = pd.concat(total_market_obs_df[-(self.max_lag + 1):])

            confidence = self.predict_rolling([history_df, None], market_obs_df, verbose=True, normalize=True, normalize_vals = [self.maxs, self.mins])      

            preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
            predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
            PREDICTIONS.append(predictions_template_df)
            packaging_time += time() - t
        
        del days
        if verbose: print("Finished predict_accurate for model {}, TIME {}".format(self.name, time()-start_time))
        return pd.concat(PREDICTIONS).reset_index(drop=True,inplace=True)

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
        to_save = [self.model1, self.model2, self.model3, self.model4, self.model5, self.model6]
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
        self.model1 = models[0]
        self.model2 = models[1]
        self.model3 = models[2]
        self.model4 = models[3]
        self.model5 = models[4]
        self.model6 = models[5]
        print("[_load] models loaded succesfully")

