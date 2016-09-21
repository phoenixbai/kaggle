import xgboost as xgb
import numpy as np
from sklearn.cross_validation import KFold,train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_svmlight_file

rng = np.random.RandomState(31333)

data = load_svmlight_file('train_svm.txt.train')
X, y = data[0], data[1]

test_data = load_svmlight_file('train_svm.txt.test')
tX, ty = test_data[0], test_data[1]

print 'Parameter Optimization'
param = {'max_depth': [4, 6, 8], 'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.3, 1]}
xgb_model = xgb.XGBClassifier()
clf = GridSearchCV(xgb_model, param, verbose=1, scoring='roc_auc')
clf.fit(X, y)
auc = roc_auc_score(ty, clf.predict(tX))
print("auc score: %.5f" % auc)
print(clf.best_score_)
print(clf.best_params_)


# single classifier training and prediction
xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.5, n_estimators=50, silent=True)
xgb_model.fit(X, y)
auc = roc_auc_score(ty, xgb_model.predict(tX))
print("auc score: %.5f" % auc)

