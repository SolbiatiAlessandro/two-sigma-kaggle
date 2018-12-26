DOCS

Guidelines [follow carefully]
==========

 - every model to be used must satisfy the APIs requirements of model_template.py, and must pass all the tests like test_model_template.py

- data preprocessing is done at the beggining over the whole tranining dataset, and it MUST not add new features. New features must be added only during the _generate_features method.

- you can see a complete example on how to fit a single model in the kernel SINGLE MODEL SUBMISSION (TEMPLATE)

