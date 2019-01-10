SCOPE
=====
I am competing in this [Kaggle Competition](https://www.kaggle.com/c/two-sigma-financial-news) hosted by Two Sigma. I will keep this repo to be organized and to put the resources I use and potentially some code.

CONTENTS
========
Here you will find an organize structure of all my models and test I use to explore the competition, plus a collection of resources I collected

All the model inside this folder are standardized with requirementes on high-level APIs that the model must offer to be used in other environments. You can read more in kernels/README.md

----

## Submission model

You can find inside DecisionTree/model_lgbm_71.py a prediction model base on LGBM (boosted decision trees) with high-level APIs to be used for predictions including 

- model.train
- model.predict
- model.predict_rolling
- model.save
- model.load
- model.generate_features

You can see how the model can be used inside test_model_lgbm_71.py

The models scores on public leaderboard top 5% in the competition.

## Scripts

In the folder you can also find many scripts, naming a few

evaluate.py : a standardize way to validate models
evaluate_rolling.py : a standardized way to validate models using prediction data pipeline
evaluate_kfold.py : a standardized way to validate model using stacking

## Notebooks

There are also a lot of notebook arounds, mainly self explainatory.
The most interesting one are inside 

- Kernels/Ensemble/stacking-script.ipynb : outline the strategy of stacking differnt models for timeseries prediction

- Kernels/DecistionTree/ipynb : folder for all the notebooks for decision tree based models
