import pandas as pd

names = ['people_id','activity_id','date','activity_category','char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10','outcome']
data = pd.read_csv('act_train.csv', names=names, usecols=['outcome'], dtype=str)

# print data.ix[:, 0:10]

num = 0
total = 0
for r in data.values:
    total += 1
    if r[0] == '1':
        num += 1

print num
print total