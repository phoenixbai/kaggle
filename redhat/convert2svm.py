import pandas as pd
import re
import csv

names = ['people_id','activity_id','date','activity_category','char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10','outcome']
pl_names = ['people_id','char_1','group_1','char_2','date','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37','char_38']


def convert2int(v):
    return '1' if 'True' == v else '0' if 'False' == v else None if not v else re.search('\d+$', v).group(0)


act_train = pd.read_csv('test.csv', names=names, converters={x: convert2int for x in names[3:-1]}, dtype=str)
act_test = pd.read_csv('act_test_test.csv', names=names[:-1], converters={x: convert2int for x in names[3:-1]}, dtype=str)
people = pd.read_csv('people_test.csv', names=pl_names, converters={x: convert2int for x in pl_names[1:4]+pl_names[5:-1]}, dtype=str)

cnt = act_train.people.value_counts()
people_act_cnt = pd.DataFrame({'people_id': cnt.index, 'ppl_act_cnt': cnt.values})
print people_act_cnt

# print 'act_train shape: (%d, %d)' % act_train.shape
print 'act_test shape: (%d, %d)' % act_test.shape
print 'people shape: (%d, %d)' % people.shape

# train_data = pd.merge(act_train, people, on=['people_id'])
test_data = pd.merge(act_test, people, on=['people_id'])

# label = act_train['outcome']
# feats = pd.concat([train_data.ix[:, 3:14], train_data.ix[:, 15:18], train_data.ix[:, 19:]], axis=1)
test_feats = pd.concat([test_data.ix[:, 1], test_data.ix[:, 3:17], test_data.ix[:, 18:]], axis=1)

# train_data_final = pd.concat([label, feats], axis=1)
# colum_names = list(train_data_final.columns.values)
test_colum_names = list(test_feats.columns.values)

# print 'train_data shape: (%d, %d)' % train_data_final.shape
# print 'train data colum_names: %s' % ','.join(colum_names)
print 'test data colum_names: %s' % ','.join(test_colum_names)

# feat_size = feats.shape[1]
test_feat_size = test_feats.shape[1]-1

# with open('train_svm.txt', 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     rows = [' '.join([str(x[0])] + [str(i)+':'+str(x[i+1]) for i in range(feat_size) if x[i+1]]) for x in train_data_final.values]
#     for r in rows:
#         writer.writerow([r.strip()])


with open('test_svm.txt', 'w') as ft, open('test_svm_map.txt', 'w') as fm:
    twriter = csv.writer(ft, delimiter='\t')
    mwriter = csv.writer(fm, delimiter='\t')
    ids = test_feats.ix[:, 0].values
    idx = test_feats.index.values

    rows = [(str(x) + ',' + z, str(x) + ' ' + ' '.join([str(i)+':'+str(y[i]) for i in range(test_feat_size) if y[i]]))
            for x, z, y in zip(idx, ids, (test_feats.ix[:, 1:]).values)]

    for (x, y) in rows:
        twriter.writerow([y.strip()])
        mwriter.writerow([x.strip()])

