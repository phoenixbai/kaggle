
import xgboost as xgb

# issue 1478: https://github.com/dmlc/xgboost/issues/1478

# data are from demo/rank/
train = xgb.DMatrix('mq2008.train')
vali = xgb.DMatrix('mq2008.vali')

xgb_params = {'silent': 0, 'eval_metric': ['map','ndcg'], 'nthread': 4, 'bst:max_depth': 6, 'objective': 'rank:pairwise', 'lambda': 0.0001, 'bst:eta': 0.3, 'booster': 'gblinear'}
num_rounds = 10
evallist = [(train, 'train'),(vali, 'vali')]
bst = xgb.train(xgb_params, train, num_rounds, evallist, early_stopping_rounds=5)