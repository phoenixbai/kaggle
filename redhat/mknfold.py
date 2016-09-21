#!/usr/bin/python
import sys
import random

if len(sys.argv) < 3:
    print ('Usage:<filename> <k> <m> [nfold = 7]')
    exit(0)

random.seed(11)

k = int(sys.argv[2])
m = int(sys.argv[3])
if len(sys.argv) > 4:
    nfold = int(sys.argv[4])
else:
    nfold = 7

fi = open(sys.argv[1], 'r')
ftr = open(sys.argv[1]+'.train', 'w')
fte = open(sys.argv[1]+'.test', 'w')
fva = open(sys.argv[1]+'.vali', 'w')
te_num = 0
tr_num = 0
va_num = 0
for l in fi:
    label = l.split(' ')[0]
    is_pos = True if label == '1' else False
    r = random.randint(1, nfold)
    if r == k:
        fte.write(l)
        if is_pos:
            te_num += 1
    elif r == m:
        fva.write(l)
        if is_pos:
            va_num += 1
    else:
        ftr.write(l)
        if is_pos:
            tr_num += 1

print 'train_positive_sample_count: %d' % tr_num
print 'test_positive_sample_count: %d' % te_num
print 'vali_positive_sample_count: %d' % va_num
print 'positive_sample_count in total: %d' % (tr_num + te_num + va_num)

fi.close()
ftr.close()
fte.close()