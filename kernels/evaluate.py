"""this is a command-line script to evaluate one or more standard models and compare predictions"""
import argparse
import time
from sigma_score import sigma_score
import pandas as pd
import numpy as np
import pickle as pk
import os

def main():
    parser = argparse.ArgumentParser(description='this is a command-line script to evaluate one or more standard models and compare predictions: python evaluate.py DecisionTrees/model1.py NeuralNets/model2.py')
    parser.add_argument('model_name', type=str, nargs='+',
                        help=':name of model in the form DecisionTree.model_lgbm_baseline')
    args = parser.parse_args()
    paths =  args.model_name
    print("\n\nCalled evaluate.py with arguments:"+" ".join(paths))
    models = []
    for i, path in enumerate(paths):
        print("\nimporting "+path)
        exec("from "+path+" import model as model"+str(i))
        exec("models.append(model"+str(i)+")")

    print("Preparing train data..")
    market_train_df = pd.read_csv("data/market_train_df.csv").drop('Unnamed: 0', axis=1)
    train_target = market_train_df['returnsOpenNextMktres10']
    market_train_df.drop(['returnsOpenNextMktres10'], axis=1)

    print("Preparing test data..")
    market_test_df = pd.read_csv("data/market_test_df.csv").drop('Unnamed: 0', axis=1)
    test_target = market_test_df['returnsOpenNextMktres10']
    test_time = market_test_df['time']
    market_test_df.drop(['returnsOpenNextMktres10'], axis=1)

    for i, model in enumerate(models):
        if os.path.isfile("predictions/"+paths[i]):
            print("\n\nPrediction file already found for "+paths[i])
            print("[NO FOR SKIP] do you want to train model and predict anyway? (it will overwrite predictions) y/n ")
            got = input()
            if got == "n":
                continue
        _model = model(paths[i])
        _model.train([market_train_df, None], train_target, verbose=True)
        predictions = _model.predict(market_test_df)
        print("Generated prediction files for "+paths[i])

        pk.dump(model_predictions, open("predictions/"+paths[i], "wb"))
        score = sigma_score(model_predictions, test_target, test_time)
        print("######### "+paths[i]+" #########")
        print("sigma score: "+str(score))
        print("#########             #########")



if __name__ == "__main__":
    main()
