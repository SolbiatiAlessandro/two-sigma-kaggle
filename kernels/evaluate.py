"""this is a command-line script to evaluate one or more standard models and compare predictions"""
import argparse
import time
from sigma_score import sigma_score
import pandas as pd
import numpy as np
import pickle as pk
import os
from utils import progress
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='this is a command-line script to evaluate one or more standard models and compare predictions: python evaluate.py DecisionTrees.model1.py NeuralNets.model2.py')
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

    skip_training = False
    if all(os.path.isfile("predictions/"+paths[i]) for i, model in enumerate(models)):
        print("[YES -> SKIP, NO -> TRAIN]: all models already have predictions. Do you want to skip training? y/n")
        got = input()
        if got == "y": skip_training = True

    print("Preparing test data..")
    market_test_df = pd.read_csv("data/market_test_df.csv").drop('Unnamed: 0', axis=1)
    test_target = market_test_df['returnsOpenNextMktres10']
    test_target = test_target.clip(test_target.quantile(0.001), test_target.quantile(0.999))
    test_time = market_test_df['time']
    market_test_df.drop(['returnsOpenNextMktres10'], axis=1, inplace=True)

    if not skip_training:
        print("\n\nStart training phase..")

        print("Preparing train data..")
        market_train_df = pd.read_csv("data/market_train_df.csv").drop('Unnamed: 0', axis=1)

        for i, model in enumerate(models):
            if os.path.isfile("predictions/"+paths[i]):
                print("\n\nPrediction file already found for "+paths[i])
                print("[YES -> SKIP, NO -> TRAIN] do you want to skip training for this model and using existing predictions? (training will overwrite predictions) y/n ")
                got = input()
                if got == "y":
                    continue
            _model = model(paths[i])
            market_train_df = _model._preprocess(market_train_df)
            train_target = market_train_df['returnsOpenNextMktres10']
            market_train_df.drop(['returnsOpenNextMktres10'], axis=1)
            _model.train([market_train_df, None], train_target, verbose=True)
            model_predictions = _model.predict([market_test_df, None])
            print("Generated prediction files for "+paths[i])

            pk.dump(model_predictions, open("predictions/"+paths[i], "wb"))
    for path in paths:
        loaded_predictions = pk.load(open("predictions/"+path, "rb"))

        score = sigma_score(loaded_predictions, test_target, test_time)
        
        print("\n"+"#"*10+path+"#"*10)
        print("sigma score: "+str(score))
        print("#"*(20+len(path)))
        plt.hist(loaded_predictions, bins='auto', label=path, alpha=0.5)

    score = sigma_score(test_target, test_target, test_time)
    print("\n"+"#"*10+"target values"+"#"*10)
    print("sigma score: "+str(score))
    print("#"*(20+len(paths[i])))
    plt.hist(test_target, bins='auto', label="target values", alpha=0.5)
    print("printing predictions distribution..")
    plt.title("historgram for different prediction distributions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
