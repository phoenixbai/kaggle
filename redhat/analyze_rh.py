
import pandas as pd

names = ['people_id','activity_id','date','activity_category','char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10','outcome']
pl_names = ['people_id','char_1','group_1','char_2','date','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37','char_38']

act_train = pd.read_csv('test.csv', names=names, dtype=str)
people = pd.read_csv('people_test.csv', names=pl_names, dtype=str)

print 'act_train sample count: (%d, %d)' % act_train.shape
print 'people count: (%d, %d)' % people.shape

# column distinct values list
print 'act_train column distinct values --------------'
print 'activity_category unique value list: '
print act_train.activity_category.unique()
print act_train.apply(pd.Series.value_counts, axis=0).fillna(0)