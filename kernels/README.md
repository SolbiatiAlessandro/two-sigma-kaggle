DOCS

Guidelines [follow carefully]
==========

 - every model to be used must satisfy the APIs requirements of model_template.py, and must pass all the tests like test_model_template.py

- data preprocessing is done at the beggining over the whole tranining dataset, and it MUST not add new features. New features must be added only during the _generate_features method.

- you can see a complete example on how to fit a single model in the kernel SINGLE MODEL SUBMISSION (TEMPLATE)

ISSUES
======

[SOLVED] After writing SUBMISSION LGBM TEMPLATE and run it has some bug but I cant really find what is wrong. I tried to drop additional feature but the model look bugged even without those. I need to take this as an opportunity for a detail debug process of the whole model. [SOLUTION: the model was trained with objective function as binary and so was predicting in the range [0, 1] and not regression]

sigma-score metric valid (doesn't pass test)

shap goes in segmentatio fault 

DEV
===

- add SHAP for feature importance: OK
- have a look at hyperopt

