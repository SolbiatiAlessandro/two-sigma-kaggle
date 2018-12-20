import unittest
import model_template
import pandas as pd


class testcase(unittest.TestCase):

    def setUp(self):
        self.market_train_df = pd.read_csv("data/market_train_df_head.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = pd.read_csv("data/news_train_df_head.csv").drop('Unnamed: 0', axis=1)
        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop(['returnsOpenNextMktres10'], axis=1)
        
        self.market_cols = list(self.market_train_df.columns)
        self.news_cols = list(self.news_train_df.columns)

    def test_generate_features(self):
        m = model_template.model_example('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        # assert here on newly generated features
        self.assertFalse(complete_features.empty)
        self.assertTrue('open+close' in complete_features.columns)
        print("generate features test OK")

    def test_train(self):
        m = model_template.model_example('example')
        self.assertTrue(m.model is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

    def test_predict(self):
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_template.model_example('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict(X_test, verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(X_test[0].shape[0], len(got))
        self.assertEqual(len(y_test), len(got))
        print("predictions test OK")

    def test_predict_rolling(self):
        historical_df  = [self.market_train_df.iloc[-40:], self.news_train_df[-40:]]
        y_test = self.target[-20:]
        
        m = model_template.model_example('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict_rolling(historical_df, len(y_test), verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(len(y_test), len(got))
        print("rolling predictions test OK")

    def test_lagged_features(self):
        """simulate historical_df pattern to check 
        historical features work properly"""

        historical_len = 40
        prediction_len = 20

        historical_df  = [self.market_train_df.iloc[-historical_len:], self.news_train_df[-historical_len:]]
        y_test = self.target[-prediction_len:]

        m = model_template.model_example('example')
        processed_historical_df = m._generate_features(historical_df[0], historical_df[1])
        X_test = processed_historical_df.iloc[-prediction_len:]

        # the feature engineering process should have generated
        # a column called df['lag_10_open_max'] that is a lagged
        # max value of last 10 time points of open column
        # import pdb;pdb.set_trace() 
        X_test.reset_index(inplace=True)

        value_in_test = X_test.loc[0, 'lag_10_open_max']
        real_lagged_value = processed_historical_df[(-prediction_len - 10):(-prediction_len + 1)]['open'].max() 
        self.assertEqual(value_in_test, real_lagged_value)
        print("lagged feature test OK")


if __name__=="__main__":
    unittest.main()
