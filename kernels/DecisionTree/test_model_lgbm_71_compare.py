import unittest
import model_lgbm_71
import pandas as pd
import numpy as np
import pickle as pk
from datetime import datetime, date


class testcase(unittest.TestCase):
    """this test cases are conducted as a step by step comparison with https://www.kaggle.com/guowenrui/sigma-eda-versionnew"""

    def test_compare(self):
        print('[test_compare] loading ../data/ market_train_df')
        market_train_df = pd.read_csv("../data/market_train_df.csv").drop('Unnamed: 0', axis=1)
        market_train_df['time'] = pd.to_datetime(market_train_df['time']).dt.date
        market_train_df = market_train_df.loc[market_train_df['time']>=date(2010, 1, 1)]

        model = model_lgbm_71.model("comparison")
        generated_features = model._generate_features(market_train_df, None, verbose=True)

        reference = pk.load("pickle/_ref_market_train_df.pkl","rb")
        print('[test_compare] reference loaded succefully')

        import pdb;pdb.set_trace()
        self.assertEqual(len(generated_features.columns), len(reference.columns))
        for i, col in enumerate(generated_features.columns):
            self.assertEqual(col, reference.columns[i])
        

if __name__=="__main__":
    unittest.main()
