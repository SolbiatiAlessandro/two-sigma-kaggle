import unittest
import model_lgbm_binary_bagged_random_validation
import pandas as pd
import numpy as np


class testcase(unittest.TestCase):

    def setUp(self):
        self.market_train_df = pd.read_csv("../data/market_train_df_head.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = pd.read_csv("../data/news_train_df_head.csv").drop('Unnamed: 0', axis=1)
        
        self.market_cols = list(self.market_train_df.columns)
        self.news_cols = list(self.news_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop(['returnsOpenNextMktres10'], axis=1)

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
        m = model_lgbm_binary_bagged_random_validation.model('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        # assert here on newly generated features
        self.assertFalse(complete_features.empty)
        self.assertTrue('weekday' in complete_features.columns)

        # this check takes some lagged features and
        # manually compute expected lagged values
        # to check the values are correct

        poss = [90000, 92000, 95000]
        cols = ['returnsClosePrevRaw10', 'returnsOpenPrevMktres10']
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
                    time_window = time_serie.iloc[check_day-int(lag)+1:check_day+1,1]
                    real = time_window.fillna(0).mean()
                    try:
                        self.assertTrue(abs(got -  real) < 0.001)
                    except:
                        exit("test_generate_features failed on : {}, {}, {}".format(pos,lag,col))

        print("generate features test OK")

    @unittest.skip("for later")
    def test_train(self):
        m = model_lgbm_binary_bagged_random_validation.model('example')
        self.assertTrue(m.model1 is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model1), m.type)
        print("train test OK")

    #@unittest.skip("for later")
    def test_predict(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_binary_bagged_random_validation.model('example')
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
        
        m = model_lgbm_binary_bagged_random_validation.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        m.predict(X_test, verbose=True, do_shap=True)

        print("shap OK")

    @unittest.skip("for later")
    def test_predict_rolling(self):
        historical_df  = [self.market_train_df.iloc[-40:], self.news_train_df[-40:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_binary_bagged_random_validation.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict_rolling(historical_df, len(y_test), verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(len(y_test), len(got))
        print("rolling predictions test OK")
    
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
        m = model_lgbm_binary_bagged_random_validation.model('example')

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
        m = model_lgbm_binary_bagged_random_validation.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        m.inspect(self.market_train_df)

    @unittest.skip("this is computationally heavy")
    def test_train_with_fulldataset(self):
        m = model_lgbm_binary_bagged_random_validation.model('example')
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

    #@unittest.skip("wait")
    def test_prediction_postprocessing(self):
        m = model_lgbm_binary_bagged_random_validation.model('example')
        model1_predictions = np.full(100, 0.4)
        model2_predictions = np.full(100, 0.6)
        y_test = m._postprocess([model1_predictions, model2_predictions])
        # test bagging 
        self.assertEqual(y_test.shape, (100, ))
        # test mapping
        self.assertTrue(all(np.full(100, 0) == y_test))
        print("test_prediction_postprocessing OK")


if __name__=="__main__":
    unittest.main()
