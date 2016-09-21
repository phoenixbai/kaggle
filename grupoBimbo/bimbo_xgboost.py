#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

np.random.seed(0)

#trying out external memory computation of xgboost

def train_xgboost(fold, rounds = 1000):
    train_df = 'train_folds_{}.txt#dtrain.cache'.format(fold)
    valid_df = 'valid_folds_{}.txt#dvalid.cache'.format(fold)
    xg_train = xgb.DMatrix(train_df, missing=np.nan)
    xg_valid = xgb.DMatrix(valid_df, missing=np.nan)

    ## setup parameters for xgboost
    evals = dict()
    params = {
            'eta': 0.1,
            'gamma': 0,
            'max_depth': 11,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 0.5,
            'target': 'target',
            'validation_set': xg_valid,
            'num_class' : 71,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            'silent': 1,
            }

    watchlist = [ (xg_train, 'train'), (xg_valid, 'valid') ]
    print('Training...')
    bst = xgb.train(params, xg_train, rounds, watchlist,
                    early_stopping_rounds=100, evals_result=evals)
    return bst, min(evals['valid'])

