import xgboost as xgb

dtrain = xgb.DMatrix('dataset/train_svm.csv')
dtest = xgb.DMatrix('dataset/test_svm.csv')
params = {
            'max_depth': 3,
            'min_child_weight': 1,
            'objective': 'reg:linear',
            'silent': 1,
            'stratified': True,
            'verbose_eval': 1,
            'evals': [(dtrain, 'train'), (dtest, 'test')]
            }

reg1 = xgb.train(params, dtrain, evals=params['evals'], verbose_eval=1)
print reg1

reg = xgb.cv(params, dtrain, num_boost_round=10, nfold=5)
print reg
