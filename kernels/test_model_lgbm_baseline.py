import unittest
import model_lgbm_baseline
import pandas as pd


class testcase(unittest.TestCase):

    def setUp(self):
        self.market_train_df = pd.read_csv("data/market_train_df_head.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = pd.read_csv("data/news_train_df_head.csv").drop('Unnamed: 0', axis=1)
        
        self.market_cols = list(self.market_train_df.columns)
        self.news_cols = list(self.news_train_df.columns)

        #augmenting data to make a bit more realistic for lgb
        self.market_train_df = pd.concat([self.market_train_df for _ in range(10)])
        self.news_train_df = pd.concat([self.news_train_df for _ in range(10)])
        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop(['returnsOpenNextMktres10'], axis=1)

    def test_generate_features(self):
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        # assert here on newly generated features
        self.assertFalse(complete_features.empty)
        self.assertTrue('weekday' in complete_features.columns)
        print("generate features test OK")

    def test_train(self):
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        self.assertTrue(m.model is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

    #@unittest.skip("for later")
    def test_predict(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict(X_test, verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(X_test[0].shape[0], len(got))
        self.assertEqual(len(y_test), len(got))
        print("predictions test OK")

    #@unittest.skip("for later")
    def test_predict_rolling(self):
        historical_df  = [self.market_train_df.iloc[-40:], self.news_train_df[-40:]]
        y_test = self.target[-20:]
        
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict_rolling(historical_df, len(y_test), verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(len(y_test), len(got))
        print("rolling predictions test OK")

    #@unittest.skip("for later")
    def test_lagged_features(self):
        """simulate historical_df pattern to check 
        historical features work properly"""

        historical_len = 40
        prediction_len = 20

        historical_df  = [self.market_train_df.iloc[-historical_len:], self.news_train_df[-historical_len:]]
        y_test = self.target[-prediction_len:]

        m = model_lgbm_baseline.model_lgbm_baseline('example')
        processed_historical_df = m._generate_features(historical_df[0], historical_df[1])
        X_test = processed_historical_df.iloc[-prediction_len:]

        X_test.reset_index(inplace=True)

        #import pdb;pdb.set_trace()
        value_in_test = X_test.loc[0, 'lag_7_returnsClosePrevRaw10_max']
        real_lagged_value = processed_historical_df[(-prediction_len - 6):(-prediction_len + 1)]['returnsClosePrevRaw10'].max() 
        self.assertEqual(value_in_test, real_lagged_value)
        self.assertEqual(m.max_lag, 200)
        print("lagged feature test OK")

    #@unittest.skip("do not print")
    def test_inspect(self):
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        m.inspect()

    @unittest.skip("this is computationally heavy")
    def test_train_with_fulldataset(self):
        m = model_lgbm_baseline.model_lgbm_baseline('example')
        self.assertTrue(m.model is None)

        print("loading full dataset ..")
        self.market_train_df = pd.read_csv("data/market_train_df.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = None
        
        self.market_cols = list(self.market_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop('returnsOpenNextMktres10',axis=1)

        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

        m.inspect() #looks healthy


if __name__=="__main__":
    unittest.main()
