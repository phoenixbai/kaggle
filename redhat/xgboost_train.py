import xgboost as xgb
import csv
import sys
import pandas as pd

def train_with_all():

    dtrain = xgb.DMatrix('train_svm_ohe.txt#dtrain_all.cache')

    param = {'max_depth': 8,
             'eta': 0.5,
             'silent': 1,
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'tree_method': 'exact',
             # 'min_child_weight': 0.5,
             'gamma': 0.01,
             'subsample': 1,
             'colsample_bytree': 1
             }

    watch_list = [(dtrain, 'train')]  #, (dvali, 'vali')
    num_round = 200
    bst = xgb.train(param, dtrain, num_round, watch_list)
    bst.save_model('xgboost_rh_all.model')
    bst.dump_model('xgboost_rh_model_all.dump.txt', fmap='featmap.txt', with_stats=True)


#[199]	train-auc:0.970753	test-auc:0.970006

def train():

    dtrain = xgb.DMatrix('train_svm_ohe.txt.train#dtrain.cache')
    dtest = xgb.DMatrix('train_svm_ohe.txt.test#dtest.cache')
    # dvali = xgb.DMatrix('train_svm.txt.vali#dvali.cache')

    param = {'max_depth': 6,
             'eta': 0.1,
             'silent': 1,
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'tree_method': 'exact',
             'min_child_weight': 0.5,
             'gamma': 0.01,
             'subsample': 0.8,
             'colsample_bytree': 0.8
             }

    watch_list = [(dtrain, 'train'), (dtest, 'test')]  #, (dvali, 'vali')
    num_round = 200
    bst = xgb.train(param, dtrain, num_round, watch_list)
    bst.save_model('xgboost_rh.model')
    bst.dump_model('xgboost_rh_model.dump.txt', fmap='featmap.txt', with_stats=True)

# total vs. positive: 2197291 vs. 975497
#[49]	train-error:0.126806	train-auc:0.945091	eval-error:0.126246	eval-auc:0.94477
# [99]	train-error:0.124863	train-auc:0.957172	eval-error:0.124409	eval-auc:0.956874
# [99]	train-error:0.115402	train-auc:0.9578	eval-error:0.114919	eval-auc:0.957377
# [99]	train-error:0.093441	train-auc:0.97395	eval-error:0.093546	eval-auc:0.973305   #depth=8
# [99]	train-error:0.088891	train-auc:0.976205	eval-error:0.089178	eval-auc:0.975668   #depth=8 , add ppl_act_cnt feature
# [99]	train-error:0.092597	train-auc:0.97235	eval-error:0.092251	eval-auc:0.971927  eta=0.2 depth=6, ppl_act_cnt
# [99]	train-error:0.068355	train-auc:0.984164	eval-error:0.069159	eval-auc:0.98349    eta=0.5


def predict(model_file='xgboost_rh_all.model'):

    act_map = pd.read_csv('test_svm_map_ohe.txt', names=['index', 'activity_id'], dtype=str, converters={'index': int})
    feats = xgb.DMatrix('test_svm_ohe.txt')
    bst = xgb.Booster(model_file=model_file)
    pred_prob = bst.predict(feats)

    pred_map = pd.DataFrame(zip(feats.get_label(), pred_prob), columns=['index', 'prob'])
    result = pd.merge(act_map, pred_map, on=['index'])

    with open('test_submission_ohe.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['activity_id', 'outcome'])
        last_prob = 0
        num = 0
        for x in result.values:
            # print x[1:]
            if x[2] != last_prob:
                num += 1
            last_prob = x[2]
            writer.writerow([x[1], "%.8f" % x[2]])
        print "different prob count: %d" % num

def main():
    if len(sys.argv) < 1:
        print 'Usage: xgboost_train.py train|predict'

    op_type = sys.argv[1]

    if op_type == 'train':
        train()
    elif op_type == 'train_all':
        train_with_all()
    elif op_type == 'predict':
        predict()


if __name__ == '__main__':
    main()