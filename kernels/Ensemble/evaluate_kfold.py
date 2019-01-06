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
import sys;sys.path.append('../')
from sigma_score import sigma_score
    
def prepare_levels(market_train_df, level_2_range):
    """
    this method prepare data for level1 and level2 of stacking
    it works ONLY with the whole market_train_df.csv

    Args:
        market_train_df: from market_train_df.csv
        level_2_range: (int, int) for choosing size of the ensemble
            (more details in code)
    Returns:
        X_train_level1, y_train_level1,
        X_train_level2, y_train_level2, periods_level2
    """
    print("[prepare_levels] starting..")
    start_time = time()

    # folds are periods of 6 month and 15 days length
    periods = ['2010-01-01 22:00:00+0000',
               '2010-06-15 22:00:00+0000',
               '2011-01-01 22:00:00+0000',
               '2011-06-15 22:00:00+0000',
               '2012-01-01 22:00:00+0000',
               '2012-06-15 22:00:00+0000',
               '2013-01-01 22:00:00+0000',
               '2013-06-15 22:00:00+0000',
               '2014-01-01 22:00:00+0000',
               '2014-06-15 22:00:00+0000',
               '2015-01-01 22:00:00+0000',
               '2015-06-15 22:00:00+0000',
               '2016-01-01 22:00:00+0000',
               '2016-06-15 22:00:00+0000',
               '2017-01-01 22:00:00+0000']
    market_train_df['period'] = -1
    for i, period in enumerate(periods[:-1]):
        market_train_df.loc[(market_train_df['time'] < periods[i + 1]) & (period <= market_train_df['time']), ['period']] = i
    cols = market_train_df.columns.tolist()
    cols.insert(1, cols[-1])
    cols.pop()

    # what we did here we created a new column period to easily move 
    # between fold, then we prepare level_1 features
    X_train_level1 = market_train_df[cols]
    y_train_level1 = X_train_level1['returnsOpenNextMktres10']
    X_train_level1 = X_train_level1.drop('returnsOpenNextMktres10',axis=1)

    # this break if you market_train_df doesn't goe from 2010-01-01 to 2017-06-31
    assert len(X_train_level1['period'].unique()) == 14
    
    # here we make the important choice
    # of how long we want to make the stacking for level 2
    # in case our predict method is extremely slow
    # we might want to decrease level_2_range size

    periods_level2 = X_train_level1['period'][X_train_level1['period'].isin(range(level_2_range[0], level_2_range[1]))]
    y_train_level2 = y_train_level1[X_train_level1['period'].isin(range(level_2_range[0], level_2_range[1]))]
    
    level1_models = 1 # how many level1 models do we have?
    # here we have only one since is CV kfold validation
    # and is not still the ultimate stacking script
    X_train_level2 = np.zeros([y_train_level2.shape[0], level1_models])

    print("[prepare_levels] done in {}".format(time()-start_time))

    return X_train_level1, y_train_level1, X_train_level2, y_train_level2, periods_level2


def evaluate_kfold(model, MODEL_NAME, DATA_FOLDER, PREDICTIONS_FOLDER, inspect=False, STACKING_RANGE=(8, 14)):
    """
    this is a evolution of evaluate_rolling.py
    it takes the structure of stacking-script.ipynb to do a 
    kfold cross validation. The strucure will be used later
    for stacking using more then one model as MODEL_NAME

    Args:
        model: class model (from model_lgbm_71 import model)
            to evaluate (not instance, but callable)
        MODEL_NAME: 'lgbm_71_leak'
        DATA_FOLDER: "../../data"
        PREDICTIONS_FOLDER: "../../predictions"
        STACKING_RANGE: used to determine size of the stacking
             , level 2 of the training data (ref in code comments)
        inspect: bool, call model.inspect and print training
         (will stop pipeline and wait for user prompt)

    Returns:
        None: saves in PREDICTIONS_FOLDER+"rolling_"+MODEL_NAME 
        a .pkl file with predictions (without extension)

    NOTE: the function is a bit too long, I didnt' incorporate
    in classes because it will be done in stacking later, I
    separated in section though using ### symbols
    """

    ########################
    ### [script details] ###
    ########################

    try:
        os.path.isdir(DATA_FOLDER)
    except: exit("DATA_FOLDER not valid")
    try:
        os.path.isdir(PREDICTIONS_FOLDER)
    except: exit("DATA_FOLDER not valid")
    # ISSUE: there is some problems with saving file with extension .pkl
    # they don't get saved with the extension and then not recognized by
    # check for rewrite alert. Should look into it and standardize
    # saving format for pkl files
    SAVE_PATH = os.path.join(PREDICTIONS_FOLDER,"rolling_"+MODEL_NAME)
    print("[evaluate_kfold] SAVE_PATH = "+SAVE_PATH)
    if os.path.isfile(SAVE_PATH):
        print("[evalute_kfold] overwriting exsisting predictions at '"+SAVE_PATH+"' , Continue? y/n")
        got = raw_input()
        if got == 'n': exit("Quitting")
            
    
    ######################
    ### [data reading] ###
    ######################
    
    print("[evaluate_kfold] reading data")
    news_train_df = None
    market_train_df = pd.read_csv(os.path.join(DATA_FOLDER,'market_train_df.csv')).drop('Unnamed: 0', axis=1)
    market_test_df = pd.read_csv(os.path.join(DATA_FOLDER,'market_test_df.csv')).drop('Unnamed: 0', axis=1)
    # here we concat market_train_df and market_test_df
    # since the whole CV is handled by the stacking
    market_train_df = pd.concat([market_train_df, market_test_df])
    market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']


    ##########################
    ### [prepare stacking] ###
    ##########################

    X_train_level1,\
    y_train_level1,\
    X_train_level2,\
    y_train_level2,\
    periods_level2 = prepare_levels(market_train_df, STACKING_RANGE)

    total_len = X_train_level1.shape[0]
    kfold_len = X_train_level2.shape[0]
    print("[STACKING] stacking will be trained on {} of the dataset".format(kfold_len / total_len))

    # if the following two break is probably because
    # data input is not correct
    assert y_train_level1.shape[0] == total_len
    assert y_train_level2.shape[0] == kfold_len
    # X_train_level2.shape[1] is the numbers of model
    # in the stacking framework, here is only 1 since kfold
    assert X_train_level2.shape[1] == 1


    #######################################
    ### [start k-fold cross validation] ###
    #######################################

    # Now fill `X_train_level2` with metafeatures
    # and execute cross validation with Y_train_level2
    block_score_results = {}

    for cur_block_num in range(STACKING_RANGE[0], STACKING_RANGE[1]):
        print("#"*30)
        print("[STACKING] starting block number "+str(cur_block_num))
        print("#"*30)
        start_time = time()
        
        # [all the following points are description from coursera]
        #    1. Split `X_train` into parts
        #       Remember, that corresponding dates are stored in X_train_level1['period'] 
        cur_block_X = X_train_level1[X_train_level1['period'] < cur_block_num]
        cur_block_Y = y_train_level1[X_train_level1['period'] < cur_block_num]
        assert 'returnsOpenNextMktres10' not in list(cur_block_X.columns)
        assert 'target' not in list(cur_block_X.columns)
        
        cur_block_X_test = X_train_level1[X_train_level1['period'] == cur_block_num]
        cur_block_Y_test = y_train_level1[X_train_level1['period'] == cur_block_num]
        assert 'returnsOpenNextMktres10' not in list(cur_block_X_test.columns)
        assert 'target' not in list(cur_block_X_test.columns)
        # note:   there is NA in values above
        # UPDATE: this should be handled by ._generate_features, to check
        
        #    2. Fit model and put predictions          

        #    SAVE: as default I would not save models, it would become to memory
        #    expensive for my machine, but if there is some model that 
        #    I want to examine just need to change MODEL_NAME and force saving
        model = model(MODEL_NAME)
        model.train([cur_block_X, None], cur_block_Y, verbose=True, load=True)
        cur_block_pred = model.predict_accurate(cur_block_X_test,verbose=True)['confidenceValue']
        #cur_block_pred = model.predict([cur_block_X_test,None], verbose=True)
        
        # this np.c_ will be essential for stacking later
        # cur_block_X_train_level2 = np.c_[cur_block_pred_model1, cur_block_pred_model2, ..] 
        # cur_block_X_train_level2 = cur_block_pred 
    
        #    3. Store predictions from 2. in the right place of `X_train_level2`. 
        #       You can use `periods_level2` for it
        #       Make sure the order of the meta-features is the same as in `X_test_level2`

        #    this is not really necessary for cross validation
        #  X_train_level2[periods_level2 == cur_block_num] = cur_block_X_train_level2

        #    4. evaluate cross validation
        cur_block_score = sigma_score(cur_block_pred, cur_block_Y_test, cur_block_X_test['time'])

        with open(os.path.join(PREDICTIONS_FOLDER, MODEL_NAME), "a") as output_file:
            output_file.write("{} : {}\n".format(cur_block_num, cur_block_score))
        block_score_results[cur_block_num] = cur_block_score
        print('[SCORE] block_num = {}, sigma_score = {}'.format(cur_block_num, cur_block_score))

        print("[STACKING] finished block "+str(cur_block_num))
        print("[STACKING] TIME: "+str(time()-start_time))


    ###########################
    ### [write predictions] ###
    ###########################

        output_file.write("\n\n[KFOLD PREDICTION RESULTS]\n[TIME: {}]\n".format(ctime()))
        for block_num, score in block_score_results.items():
            output_file.write("{} : {}".format(block_num, score))
    print("[evaluate_kfold] PREDICTIONS SAVED SUCCESFULLY IN "+os.path.join(PREDICTIONS_FOLDER, MODEL_NAME))


if __name__ == "__main__":
    print(['[evaluate_kfold] called as script, this is a test'])
    data_folder = "./../data"
    predictions_folder = "kfold_results"
    import sys; sys.path.append('../')
    from DecisionTree.model_lgbm_71 import model
    model_name = "lgbm_71_leak_benchmark_kfold"
    evaluate_kfold(model, model_name, data_folder, predictions_folder)
