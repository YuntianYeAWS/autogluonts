# autogluonts

Please run temp.py to create an AutoEstimator
AutoEstimator:
  input: 
    dictionary of hyperparameter: dictionary of autogluon search space
    dataset:Gluonts.dataset
  Method:
    train(): starting hyperparameter searching
    get_best_estimator():create an estimator with best configuration
  
