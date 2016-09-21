import numpy as np
import xgboost as xgb

dtrain = xgb.DMatrix('train_svm.txt#dtrain_cv.cache')
param = {'max_depth': 6, 'eta': 0.2, 'silent': 1,
         'objective': 'binary:logistic',
         'eval_metric': 'auc',
         'tree_method': 'exact'
         }
num_round = 50

print ('running cross validation')

res = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
print res

