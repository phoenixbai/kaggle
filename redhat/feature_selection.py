import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from collections import defaultdict
import csv

train_input = pd.read_csv('dataset/act_train_t.csv',keep_default_na=True, usecols=['activity_category','char_10','outcome']).fillna("-1")

# remove categorical value which coverage is below than threshold for char_10 feature
c10_cnt = train_input.char_10.value_counts(normalize=True)
c10_filtered = c10_cnt[c10_cnt>0.01]
print c10_filtered.index
sum(c10_filtered.values)

# one-hot encoding for char_10
c10_cat_d = defaultdict(lambda: 't0')
c10_cols = c10_filtered.index.values
c10_cols.sort()
for k,v in enumerate(c10_cols):
    c10_cat_d[v]='t'+str(k+1)

print c10_cat_d
train_input['char_10'].replace(c10_cat_d, inplace=True)


#activity_category one hot encoding
ac_cnt = train_input.activity_category.value_counts(normalize=True)
ac_cat_d = defaultdict(lambda: 'a0')
ac_cols = ac_cnt.index.values
ac_cols.sort()
for k,v in enumerate(ac_cols):
    ac_cat_d[v]='a'+str(k+1)

print ac_cat_d
train_input['activity_category'].replace(ac_cat_d, inplace=True)
# train_input['activity_category'].head()


# DictVectorizer - string to one-hot encoding
train_dict = train_input[['activity_category','char_10']].T.to_dict().values()
train_input=train_input.drop(['activity_category','char_10'], axis=1)
train_vectorizer = DV(sparse=False)
vec_train_feat = train_vectorizer.fit_transform(train_dict)
print type(vec_train_feat)
print vec_train_feat[0:5, :]
train_df = pd.concat([train_input[['outcome','people_id']], pd.DataFrame(vec_train_feat, dtype=int)],axis=1)
print train_df.shape
print train_df.head()

# test dataset - one-hot encoding
test_input = pd.read_csv('dataset/act_test_t.csv',keep_default_na=True).fillna("-1")
test_input['char_10'].replace(c10_cat_d, inplace=True)
test_input['activity_category'].replace(ac_cat_d, inplace=True)
test_dict = test_input[['activity_category','char_10']].T.to_dict().values()
vec_test_feat = train_vectorizer.transform(test_dict)
test_df = pd.concat([test_input[['activity_id','people_id']], pd.DataFrame(vec_test_feat.toarray(), dtype=int)],axis=1)
print test_df.shape
print test_df.head()

#people dataset - one-hot encoding
ppl_input = pd.read_csv('dataset/people_t.csv', keep_default_na=True).fillna(-1)
ppl_feat = ppl_input[ppl_input.columns.difference(['people_id','date','group_1','char_3'])]
ppl_dict = ppl_feat.T.to_dict().values()
print type(ppl_dict)
print ppl_dict[0]

vectorizer = DV(sparse=True)
vec_ppl_feat = vectorizer.fit_transform(ppl_dict)
print type(vec_ppl_feat)
print vec_ppl_feat.shape
ppl_df = pd.DataFrame(vec_ppl_feat.toarray(), dtype=int)
ppl_df['people_id']=ppl_input['people_id']

# merging with people feature
train_dataset = pd.merge(train_df, ppl_df, how='inner', on='people_id')
print train_dataset.shape
# print train_dataset.head()

test_dataset = pd.merge(test_df, ppl_df, how='inner', on='people_id')
print test_dataset.shape
# print test_dataset.head()


# train dataset to svm format
train_dataset = train_dataset.drop(['people_id'], axis=1)
print train_dataset.head()
feat_size = train_dataset.shape[1]-1
with open('train_svm_ohe.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    rows = [' '.join([str(x[0])] + [str(i)+':'+str(x[i+1]) for i in range(feat_size) if x[i+1]]) for x in train_dataset.values]
    for r in rows:
        writer.writerow([r.strip()])


# test dataset to svm format
test_dataset = test_dataset.drop(['people_id'], axis=1)
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



