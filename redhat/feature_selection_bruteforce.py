import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from collections import defaultdict
import csv

#people dataset!!! - one-hot encoding
ppl_input = pd.read_csv('dataset/people.csv', keep_default_na=True).fillna("-1")
ppl_feat = ppl_input[ppl_input.columns.difference(['people_id','date','group_1','char_3'])]
ppl_dict = ppl_feat.T.to_dict().values()
vectorizer = DV(sparse=False)
vec_ppl_feat = vectorizer.fit_transform(ppl_dict)
ppl_df = pd.DataFrame(vec_ppl_feat, dtype=int)
ppl_df['people_id']=ppl_input['people_id']
print 'ppl_df!!!!'
print ppl_df.head()
# train dataset!!!!!

train_input = pd.read_csv('dataset/act_train.csv',keep_default_na=True, usecols=['outcome','people_id','activity_category','char_10']).fillna("-1")

def my_onehot(data, clist, col_name):

    print 'column list: %s' % ','.join(clist)
    for i,v in enumerate(clist):
        print i,v
        key = str(i)+v
        data[key] = data[col_name]==v
        data[key] = data[col_name]!=v
        data[key] = data[key].astype(int)

# s1: activity_category one hot encoding
ac_cnt = train_input.activity_category.value_counts(normalize=True)
ac_cols = ac_cnt.index.values
ac_cols.sort()
my_onehot(train_input, ac_cols, 'activity_category')
# for k, v in ac_cat_d.iteritems():
#     train_input[k+'ac'] = train_input[train_input['activity_category']==k].astype(int)
#     train_input[k+'ac'] = train_input[train_input['activity_category']!=k].astype(int)

# s2: remove categorical value which coverage is below than threshold for char_10 feature
c10_cnt = train_input.char_10.value_counts(normalize=True)
c10_filtered = c10_cnt[c10_cnt>0.01]
print c10_filtered.index
sum(c10_filtered.values)

c10_cols = c10_filtered.index.values
c10_cols.sort()
my_onehot(train_input, c10_cols, 'char_10')
# for k,v in c10_cat_d.iteritems():
#     train_input[k+'c10'] = train_input[train_input['char_10']==k].astype(int)
#     train_input[k+'c10'] = train_input[train_input['char_10']!=k].astype(int)

print train_input.head()

# s3: merging with people feature
train_dataset = pd.merge(train_input, ppl_df, how='inner', on='people_id')
print "train shape: (%d, %d)" % train_dataset.shape

# s4: train dataset to svm format
print 'train_dataset columns:'
print train_dataset.columns.values
train_dataset = train_dataset.drop(['people_id', 'activity_category', 'char_10'], axis=1)
feat_size = train_dataset.shape[1]-1
with open('train_svm_ohe.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    rows = [' '.join([str(x[0])] + [str(i)+':'+str(x[i+1]) for i in range(feat_size) if x[i+1]]) for x in train_dataset.values]
    for r in rows:
        writer.writerow([r.strip()])


# test dataset - one-hot encoding
test_input = pd.read_csv('dataset/act_test.csv',keep_default_na=True, usecols=['activity_id','people_id','activity_category','char_10']).fillna("-1")

# s1 & s2
my_onehot(test_input, ac_cols, 'activity_category')
my_onehot(test_input, c10_cols, 'char_10')
print test_input.head()
# test_dict = test_input[['activity_category','char_10']].T.to_dict().values()
# vec_test_feat = train_vectorizer.transform(test_dict)
# test_df = pd.concat([test_input[['activity_id','people_id']], pd.DataFrame(vec_test_feat.toarray(), dtype=int)],axis=1)
# print test_df.shape
# print test_df.head()

# s3:
test_dataset = pd.merge(test_input, ppl_df, how='inner', on='people_id')
print "test shape: (%d, %d)" % test_dataset.shape
print test_dataset.head()

# s4: test dataset to svm format
print 'test_dataset columns:'
print test_dataset.columns.values
test_dataset = test_dataset.drop(['people_id','activity_category', 'char_10'], axis=1)
print test_dataset.head()
test_feat_size = test_dataset.shape[1]-1
with open('test_svm_ohe.txt', 'w') as ft, open('test_svm_map_ohe.txt', 'w') as fm:
    twriter = csv.writer(ft, delimiter='\t')
    mwriter = csv.writer(fm, delimiter='\t')
    ids = test_dataset.ix[:, 0].values
    idx = test_dataset.index.values

    rows = [(str(x) + ',' + z, str(x) + ' ' + ' '.join([str(i)+':'+str(y[i]) for i in range(test_feat_size) if y[i]]))
            for x, z, y in zip(idx, ids, (test_dataset.ix[:, 1:]).values)]

    for (x, y) in rows:
        twriter.writerow([y.strip()])
        mwriter.writerow([x.strip()])



