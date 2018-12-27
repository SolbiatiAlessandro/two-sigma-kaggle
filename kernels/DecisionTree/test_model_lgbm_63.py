import unittest
import model_lgbm_63
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
        m = model_lgbm_63.model_lgbm('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        # assert here on newly generated features
        self.assertFalse(complete_features.empty)
        top_features = [
        'returnsClosePrevRaw10_lag_7_mean',
        'assetName_mean_open',
        'returnsClosePrevRaw10_lag_14_max',
        'returnsClosePrevRaw10_lag_14_min',
        'returnsClosePrevMktres10_lag_14_max',
        'returnsClosePrevMktres10_lag_7_mean',
        'returnsClosePrevMktres10_lag_14_min',
        'assetName_mean_close',
        'returnsClosePrevRaw10_lag_14_mean',
        'volume'
        ]
        for feat in top_features:
            self.assertTrue(feat in complete_features.columns)
        print(complete_features.columns)
        print("generate features test OK")

    #@unittest.skip("wait")
    def test_train(self):
        m = model_lgbm_63.model_lgbm('example')
        self.assertTrue(m.model is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

    @unittest.skip("for later")
    def test_predict(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_63.model_lgbm('example')
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
        
        m = model_lgbm_63.model_lgbm('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        m.predict(X_test, verbose=True, do_shap=True)

        print("shap OK")

    @unittest.skip("for later")
    def test_predict_rolling(self):
        historical_df  = [self.market_train_df.iloc[-40:], self.news_train_df[-40:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_63.model_lgbm('example')
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
        m = model_lgbm_63.model_lgbm('example')

        # processed_historical_df goes from current_period - max lag, to current_period (even if it is zero indexed)
        processed_historical_df = m._generate_features(historical_df[0], historical_df[1])

        X_test = processed_historical_df[processed_historical_df.period == current_period]

        # to test the lagged feature let's examine the first asset of the current period
        first_asset_open = self.market_train_df[self.market_train_df['period'] == 68].iloc[0].open
        first_asset_code = self.market_train_df[self.market_train_df['period'] == 68].iloc[0].assetCode
        print("testing rolling predictions on "+first_asset_code)
        self.assertEqual(first_asset_open, X_test.iloc[0].open)

        # this is the value in the prediction batch
        value_in_test = X_test.iloc[0][ 'returnsClosePrevRaw10_lag_7_max']
        # this is the computation of the real value from market_train
        real_lagged_value = self.market_train_df[self.market_train_df.assetCode == first_asset_code][ self.market_train_df.period > current_period - 7 ]['returnsClosePrevRaw10'].max()

        self.assertEqual(value_in_test, real_lagged_value)
        self.assertEqual(m.max_lag, 14)
        print("lagged feature test OK")

    @unittest.skip("do not print")
    def test_inspect(self):
        m = model_lgbm_63.model_lgbm('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        m.inspect(self.market_train_df)

    @unittest.skip("this is computationally heavy")
    def test_train_with_fulldataset(self):
        m = model_lgbm_63.model_lgbm('example')
        self.assertTrue(m.model is None)

        print("loading full dataset ..")
        self.market_train_df = pd.read_csv("../data/market_train_df.csv").drop('Unnamed: 0', axis=1)
        self.market_train_df = self.market_train_df.loc[self.market_train_df['time'] >= '2016-01-01 22:00:00+0000']

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


if __name__=="__main__":
    unittest.main()
