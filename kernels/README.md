DOCS

Guidelines [follow carefully]
==========

 - every model to be used must satisfy the APIs requirements of DecisionTree/model_lgbm_baseline.py, and must pass all the tests like DecisionTree/test_model_lgbm_baseline.py

- data preprocessing is done at the beggining over the whole tranining dataset, and it MUST not add new features. New features must be added only during the _generate_features method.

- prediction post processing is done at the ending outside the model, this is done to avoid bias in benchmarking

- you can see a complete example on how to fit a single model in the kernel Linear/SINGLE MODEL SUBMISSION (TEMPLATE)

ISSUES
======

[SOLVED] After writing SUBMISSION LGBM TEMPLATE and run it has some bug but I cant really find what is wrong. I tried to drop additional feature but the model look bugged even without those. I need to take this as an opportunity for a detail debug process of the whole model. [SOLUTION: the model was trained with objective function as binary and so was predicting in the range [0, 1] and not regression]

[SOLVED] sigma-score metric validation (doesn't pass test), now passes test but not sure why

[LOW PRIORITY] shap goes in segmentatio fault 

sigma-score metric in evaluate.py not working properly

DEV
===

( not important ? ) - implement hyperopt in standard model

[ DONE ] - standardize eda71 (make sure to replicate results)

- match local validation with LB (fold predcitions) validation and than create history

LOCAL RUNS:

- not random split
- dont normalize asset
- create asset categories
- cut useless features

KAGGLE RUNS:

- lgbm_model_71 train leak: OK (standard 0.704)
- [2] lgbm_model_71 train-leak map-bug: OK (didn't fix bug, 0.66)
- [3] lgbm_model_71 no train leak: with no train leak scores less 0.68
- [4] lgbm_model_71 no train leak: map bug look was fixed badly, it scores 0.64
- SigMA EDA (map-fix): non standardized model_71 with map-bug fix and check if predictions are the same with mine from 2,probably need to doublecheck mapping

[ NEXT ] - standardize stacking

- ensemble with NN ( standardized stacking )

- ensemble with XGBoost and Catboost (later)

- clean data better [ there is NO data cleaning right now!! ]

- select only top features, dont use all

- try regularization of prediction (iso-metric)

[ high priority ] - finish LSTM (stock-rnn) and later ensemble
