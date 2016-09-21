
import xgboost as xgb

# map metric issue

# train = xgb.DMatrix('mq2008.train')
# vali = xgb.DMatrix('mq2008.vali')
#
# xgb_params = {'silent': 0, 'eval_metric': 'map', 'nthread': 4, 'bst:max_depth': 6, 'objective': 'rank:pairwise', 'lambda': 0.0001, 'bst:eta': 0.3, 'booster': 'gblinear'}
# num_rounds = 10
# evallist = [(train, 'train'),(vali, 'vali')]
# bst = xgb.train(xgb_params, train, num_rounds, evallist, early_stopping_rounds=5)
#


# max_delta_step is required issue
import xgboost as xgb

train = xgb.DMatrix('machine.txt.train')
test = xgb.DMatrix('machine.txt.test')

xgb_params = {'silent': 0, 'eval_metric': 'poisson-nloglik', 'nthread': 2, 'max_depth': 4, 'objective': 'count:poisson', 'lambda': 0.0001, 'eta': 0.3, 'max_delta_step':0.8}

evallist = [(train, 'train'), (test, 'test')]
bst = xgb.train(xgb_params, train, 10, evallist)
bst.save_model('reg_count_poisson.model')

bst2 = xgb.Booster(params={'max_delta_step':0.8}, model_file='reg_count_poisson.model')
pred = bst2.predict(test)