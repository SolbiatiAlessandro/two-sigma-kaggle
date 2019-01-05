def evaluate_rolling(model, MODEL_NAME, DATA_FOLDER, PREDICTIONS_FOLDER, fold=None, inspect=False):
    """
    simulate kernel rolling predictions to validatiate on private dataset
    Args:
        model: class model (from model_lgbm_71 import model)
        MODEL_NAME: 'lgbm_71_leak'
        DATA_FOLDER: "../../data"
        PREDICTIONS_FOLDER: "../../predictions"
        fold: None or (int) for different testing folds
        inspect: bool, call model.inspect and print training
         (will stop pipeline and wait for user prompt)
    
    save in PREDICTIONS_FOLDER+"rolling_"+MODEL_NAME a .pkl file with predictions
    """
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
    
    try:
        os.path.isdir(DATA_FOLDER)
    except: exit("DATA_FOLDER not valid")
    try:
        os.path.isdir(PREDICTIONS_FOLDER)
    except: exit("DATA_FOLDER not valid")
    SAVE_PATH = os.path.join(PREDICTIONS_FOLDER,"rolling_"+MODEL_NAME+".pkl")
    print("[evaluate_rolling] SAVE_PATH = "+SAVE_PATH)
    if os.path.isfile(SAVE_PATH):
        print("[evalute_rolling] overwriting exsisting predictions at '"+SAVE_PATH+"' , Continue? y/n")
        got = raw_input()
        if got == 'n': exit("Quitting")
            
    print("[evaluate_rolling] reading data")
    news_train_df = None
    market_train_df = pd.read_csv(os.path.join(DATA_FOLDER,'market_train_df.csv')).drop('Unnamed: 0', axis=1)
    market_test_df = pd.read_csv(os.path.join(DATA_FOLDER,'market_test_df.csv')).drop('Unnamed: 0', axis=1)
    market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']
    
    #initialize and train model
    model = model(MODEL_NAME)
    target = market_train_df.returnsOpenNextMktres10
    market_train_df.drop('returnsOpenNextMktres10', axis=1, inplace=True)
    model.train([market_train_df, news_train_df], target, verbose=True)
    max_values, min_values, max_lag = model.maxs, model.mins, model.max_lag # values used for normalization during predictions
    if inspect: model.inspect(None)
    
    import pdb;pdb.set_trace()
    #prediction loop
    PREDICTIONS = []
    days = []
    for date in market_test_df['time'].unique():
        market_obs_df = market_test_df[market_test_df['time'] == date].drop(['returnsOpenNextMktres10','universe'],axis=1)
        predictions_template_df = pd.DataFrame({'assetCode':market_test_df[market_test_df['time'] == date]['assetCode'],
                                                'confidenceValue':0.0})
        days.append([market_obs_df,None,predictions_template_df])
    
    """locals required
    model: instance of model class defined above
    max_values, min_values: (pd.DataFrame)
    max_lag: (int)
    """
    from time import time
    n_days, prep_time, prediction_time, packaging_time = 0, 0, 0, 0
    total_market_obs_df = []
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        n_days +=1
        if (n_days%50==0): print(n_days,end=' ')
        t = time()
        market_obs_df['time'] = market_obs_df['time'].dt.date

        total_market_obs_df.append(market_obs_df)
        if len(total_market_obs_df) == 1:
            history_df = total_market_obs_df[0]
        else:
            history_df = pd.concat(total_market_obs_df[-(max_lag + 1):])

        confidence = model.predict_rolling([history_df, None], market_obs_df, verbose=True, normalize=True, normalize_vals = [max_values, min_values])      

        preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
        predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
        PREDICTIONS.append(predictions_template_df)
        packaging_time += time() - t
        
    import pickle as pk
    pk.dump(PREDICTIONS, open(os.path.join(PREDICTIONS_FOLDER, "rolling_"+MODEL_NAME), "wb"))
    print("[evaluate_rolling] PREDICTIONS SAVED SUCCESFULLY IN"+os.path.join(PREDICTIONS_FOLDER, "rolling_"+MODEL_NAME))

if __name__ == "__main__":
    print(['[evaluate_rolling] called as script, this is a test'])
    data_folder = "./data"
    predictions_folder = "./predictions"
    from DecisionTree.model_lgbm_71 import model
    model_name = "lgbm_71_leak_bechmark"
    evaluate_rolling(model, model_name, data_folder, predictions_folder)
