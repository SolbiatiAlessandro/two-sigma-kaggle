import unittest
import model_lgbm_71
import pandas as pd
import numpy as np


class testcase(unittest.TestCase):

    def setUp(self):
        self.market_train_df = pd.read_csv("../data/market_train_df_head.csv").drop('Unnamed: 0', axis=1)
        self.market_train_df['time'] = pd.to_datetime(self.market_train_df['time'])
        self.news_train_df = pd.read_csv("../data/news_train_df_head.csv").drop('Unnamed: 0', axis=1)
        
        self.market_cols = list(self.market_train_df.columns)
        self.news_cols = list(self.news_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop(['returnsOpenNextMktres10'], axis=1)
        self.market_train_df['time'] = pd.to_datetime(self.market_train_df['time'])

    @unittest.skip("wait")
    def test_generate_features(self):
        """
        this is one of the most important tests,
        the idea is that it needs to  make sure that all the
        generated features are exactly as imagined.
        !!NOT ONLY DIMENTIONAL AND SANITY CHECK!!
        you actually need to validate your hypotesis on the feats

        example:
        in two-sigma-kaggle I was generating lagged features but
        I forgot to add groupby('asset') and so all the features
        were basically crap. I got low score and I had no idea why.
        I was only check that the feature were generated! 
        """
        m = model_lgbm_71.model('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True, normalize=False)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        self.assertFalse(complete_features.empty)

        # assert here on newly generated features
        # (test are not splitted on different functions
        # since it takes a lot ot compute)

        ##### TEST ON LAGGED FEATURES ####

        # this check takes some lagged features and
        # manually compute expected lagged values
        # to check the values are correct

        # note: using eda67 feats generation tests need to change
        # since they implement shift while in my version
        # shifting is not implemented

        poss = [90000, 92000, 95000]
        cols = ['returnsClosePrevRaw10', 'returnsClosePrevMktres10']
        lags = ['3', '7','14']
        for pos in poss:
            for lag in lags:
                for col in cols:
                    #import pdb;pdb.set_trace()
                    # value computed in feature generation
                    got = complete_features.iloc[pos]['lag_'+lag+'_'+col+'_mean']
                    # code of the asset that on which are checking lagged feat
                    code = self.market_train_df.iloc[pos]['assetCode']
                    # this is the time_serie of the asset 'code'
                    time_serie = self.market_train_df[self.market_train_df['assetCode'] == code][col].reset_index()
                    # we are checking the i-th day 'check_day' of the time_serie
                    check_day = np.where(time_serie['index'] == pos)[0][0]
                    # time window
                    time_window = time_serie.iloc[check_day-int(lag):check_day,1]
                    real = time_window.fillna(0).mean()
                    try:
                        self.assertTrue(abs(got -  real) < 0.001)
                    except:
                        import pdb;pdb.set_trace()
                        exit("test_generate_features failed on : {}, {}, {}".format(pos,lag,col))

        print("generate features test (lagged) OK")


        #### TEST ON LABEL ENCODING ####

        # NOTE: this test is extremely weak and doens't spot
        # universal mapping bug mentioned in commit 4327f4
        # additional testing on mapping is in test_generate_features_labels
        self.assertTrue('assetCodeT' in complete_features.columns)
        self.assertTrue(complete_features['assetCodeT'].dtype == int)
        self.assertEqual(
                len(complete_features['assetCodeT'].unique()),
                len(self.market_train_df['assetCode'].unique())
                )
        self.assertFalse(any(complete_features['assetCodeT'].isna()))

        print("generate features test (label encoding) OK")

        # NOTE: normalization is still not tested here
        # it was tested empirically on notebook (not enough)
        # solving the normalization bug mentioned in commit 2af2ac5 
        # complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True, normalize=True)

    #@unittest.skip("for later")
    def test_generate_features_labels(self):
        m = model_lgbm_71.model('example')
        market_test_df = self.market_train_df

        day40 = market_test_df['time'].unique()[40]
        x_train = market_test_df[market_test_df['time'] <= day40]
        #the model would be trained with this features
        train_features = m._generate_features(x_train, None, verbose=True, normalize=False)

        day44 = market_test_df['time'].unique()[44]
        x_test = market_test_df[market_test_df['time'] == day44]
        #the model would predict with this features
        test_features = m._generate_features(x_test, None, verbose=True, normalize=False)

        #asset code mapping in training dataset
        train_mapping_df = pd.DataFrame({'assetCode':x_train['assetCode'].reset_index(drop=True),'mappedTo':train_features['assetCodeT']})

        #asset code mapping in prediction dataset
        test_mapping_df = pd.DataFrame({'assetCode':x_test['assetCode'].reset_index(drop=True),'mappedTo':test_features['assetCodeT']})

        for i, code in enumerate(list(test_mapping_df['assetCode'])):
            if i % 300 == 0: print("testing mapping for code "+str(code))
            test_map_value = test_mapping_df[test_mapping_df['assetCode'] == code]['mappedTo'].iloc[0]
            train_map_values = list(train_mapping_df[train_mapping_df['assetCode'] == code]['mappedTo'])

            # a unique asset (code) should be mapped to same value
            # in test and train dataset
            for train_map_value in train_map_values:
                self.assertEqual(test_map_value, train_map_value)

    @unittest.skip("for later")
    def test_train(self):
        m = model_lgbm_71.model('example')
        self.assertTrue(m.model1 is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model1), m.type)
        print("train test OK")

    @unittest.skip("for later")
    def test_predict(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_71.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict(X_test, verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(X_test[0].shape[0], len(got))
        self.assertEqual(len(y_test), len(got))
        print("predictions test OK")

    @unittest.skip("for later")
    def test_shap(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_71.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        m.predict(X_test, verbose=True, do_shap=True)

        print("shap OK")

    @unittest.skip("for later")
    def test_predict_rolling(self):
        import pickle as pk
        with open("pickle/rolling_predictions_dataset.pkl","rb") as f:
            days = pk.load(f)
        mins,maxs,rng = pk.load(open("pickle/normalizing.pkl","rb"))
        model = model_lgbm_71.model('DecisionTree.model_lgbm_71')
        model._load()

        PREDICTIONS = pk.load(open("pickle/_ref_rolling_predictions.pkl","rb"))

        # the following is simulation code from submission kernel

        import time
        _COMPARE_PREDICTIONS = []
        n_days = 0
        prep_time = 0
        prediction_time = 0
        n_lag=[3,7,14]
        packaging_time = 0
        total_market_obs_df = []
        for (market_obs_df, news_obs_df, predictions_template_df) in days[:2]:
            n_days +=1
            if (n_days%50==0):
                pass
                #print(n_days,end=' ')
            t = time.time()
            #market_obs_df['time'] = market_obs_df['time'].dt.date

            total_market_obs_df.append(market_obs_df)
            if len(total_market_obs_df)==1:
                history_df = total_market_obs_df[0]
            else:
                history_df = pd.concat(total_market_obs_df[-(np.max(n_lag)+1):])
                
            
            confidence = model.predict_rolling([history_df, None], market_obs_df, verbose=True, normalize=True, normalize_vals = [maxs,mins])      
               
            preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
            predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
            _COMPARE_PREDICTIONS.append(predictions_template_df)

        for i, ref in enumerate(_COMPARE_PREDICTIONS):
            df = pd.DataFrame({'assetCode':PREDICTIONS[i]['assetCode'],'ref':PREDICTIONS[i]['confidenceValue'],'compare':_COMPARE_PREDICTIONS[i]['confidenceValue']})
            try:
                self.assertTrue(all(df.iloc[:,1] == df.iloc[:,2]))
            except:
                print("AssertionError: rolling predictions not correct")
                import pdb;pdb.set_trace()
                pass

    
    @unittest.skip("for later")
    def test_lagged_eatures(self):
        """simulate historical_df pattern to check
        historical features work properly"""

        # this is the length in days of the hist dataset
        # emulating the submission system
        max_lag = 40

        # this add the column period to the dataframe
        # as written in submissino loop
        times = self.market_train_df.time.unique()

        # we are currently simulating at the last period
        current_period = len(times) - 1
        self.market_train_df['period'] = self.market_train_df['time'].apply(lambda s: np.where(times == s)[0][0])

        # create historical and process it
        historical_df  = self.market_train_df[self.market_train_df['period'] > current_period - max_lag - 1], None
        m = model_lgbm_71.model('example')

        # processed_historical_df goes from current_period - max lag, to current_period (even if it is zero indexed)
        processed_historical_df = m._generate_features(historical_df[0], historical_df[1])

        X_test = processed_historical_df[processed_historical_df.period == current_period]

        # to test the lagged feature let's examine the first asset of the current period
        first_asset_open = self.market_train_df[self.market_train_df['period'] == 68].iloc[0].open
        first_asset_code = self.market_train_df[self.market_train_df['period'] == 68].iloc[0].assetCode
        print("testing rolling predictions on "+first_asset_code)
        self.assertEqual(first_asset_open, X_test.iloc[0].open)

        # this is the value in the prediction batch
        value_in_test = X_test.iloc[0][ 'lag_7_returnsClosePrevRaw10_max']
        # this is the computation of the real value from market_train
        real_lagged_value = self.market_train_df[self.market_train_df.assetCode == first_asset_code][ self.market_train_df.period > current_period - 7 ]['returnsClosePrevRaw10'].max()

        self.assertEqual(value_in_test, real_lagged_value)
        self.assertEqual(m.max_lag, 14)
        print("lagged feature test OK")

    @unittest.skip("do not print")
    def test_inspect(self):
        m = model_lgbm_71.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        m.inspect(self.market_train_df)

    @unittest.skip("this is computationally heavy")
    def test_train_with_fulldataset(self):
        m = model_lgbm_71.model('example')
        self.assertTrue(m.model is None)

        print("loading full dataset ..")
        self.market_train_df = pd.read_csv("../data/market_train_df.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = None
        
        self.market_cols = list(self.market_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop('returnsOpenNextMktres10',axis=1)

        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

        m.inspect(self.market_train_df) #looks healthy

        got = m.predict([self.market_train_df[-100:], None], verbose=True, do_shap=True)

        print(got.describe())

    @unittest.skip("wait")
    def test_prediction_postprocessing(self):
        m = model_lgbm_71.model('example')
        model1_predictions = np.full(100, 0.4)
        model2_predictions = np.full(100, 0.6)
        y_test = m._postprocess([model1_predictions, model2_predictions])
        # test bagging 
        self.assertEqual(y_test.shape, (100, ))
        # test mapping
        self.assertTrue(all(np.full(100, 0) == y_test))
        print("test_prediction_postprocessing OK")

    @unittest.skip("wait")
    def test_clean_data(self):
        m = model_lgbm_71.model('example')
        dirty_array = np.full(10,5,dtype=float)
        dirty_array[4] = np.nan # generate artificial nans

        m._clean_data(pd.DataFrame(dirty_array))
        self.assertEqual(dirty_array[4], 5.0)

    @unittest.skip("wait")
    def test_save_load(self):
        m = model_lgbm_71.model('example')
        m.name = "save_test"
        m.model1 = 7
        m.model2 = 1
        m.model3 = 2
        m.model4 = 3
        m.model5 = 4
        m.model6 = 5
        m._save()

        n = model_lgbm_71.model('example')
        n.name = "save_test"
        n._load()
        self.assertEqual(n.model1,  7)
        self.assertEqual(n.model2,  1)
        self.assertEqual(n.model3,  2)
        self.assertEqual(n.model4,  3)
        self.assertEqual(n.model5,  4)
        self.assertEqual(n.model6,  5)
        

if __name__=="__main__":
    unittest.main()
