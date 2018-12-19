import unittest
import model_template


class testcase(unittest.TestCase):

    def setUp(self):
        import pandas as pd

        market_train_df_cols = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10', 'universe']
        news_train_df_cols = ['time', 'sourceTimestamp', 'firstCreated', 'sourceId', 'headline',
       'urgency', 'takeSequence', 'provider', 'subjects', 'audiences',
       'bodySize', 'companyCount', 'headlineTag', 'marketCommentary',
       'sentenceCount', 'wordCount', 'assetCodes', 'assetName',
       'firstMentionSentence', 'relevance', 'sentimentClass',
       'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
       'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D']

        self.X = pd.DataFrame(columns=market_train_df_cols), pd.DataFrame(columns=news_train_df_cols)
        self.Y = pd.Series([3,5,8])

    def test_generate_features(self):
        m = model_template.model_example('example')
        m._generate_features(self.X)
        #assert on new generated features

    def test_train(self):
        m = model_template.model_example('example')
        self.assertTrue(m.model is None)
        m.train(self.X, self.Y)
        self.assertEqual(type(m.model), m.type)

if __name__=="__main__":
    unittest.main()
