import numpy as np
import pandas as pd
import math

pre = pd.read_csv('test_res',header=None)
test = pd.read_csv('final_test',sep='\t')
#print len(pre)
test = test.drop(['docid'],axis=1).iloc[0:len(pre)]
pre.columns = ['label']
test['label']=pre

test.type=test.type.astype(int)
test.loc[test.type==3,'type']=1
test.loc[test.type==-1,'type']=0
test.loc[test.type==0,'type']=0

test=test.sort_values(['uid','label'],ascending=[1,0])
uids=test.uid.drop_duplicates()

print('uid nums: %d\n' % len(uids))

mm = [1,3,5,7,9,11,13,15,17,20,40,100]
for m in mm:
    acc = 0
    rec = 0
    cnt = 0
    all_pos = 0
    for uid in uids:
        #m = math.ceil(len(test[test.uid==uid]) / 2)
        now = test[test.uid==uid]
        pos = len(now.iloc[0:m].loc[now.type==1])
        all_pos = len(now.loc[now.type==1])
        if all_pos > 0:
           # print all_pos
            rec += 1.0*pos/all_pos
            acc += 1.0*pos/m
            cnt += 1
        #if cnt%1000==0:
        #    print(cnt)
    print('---------------------')
    print(m)
    print(acc/cnt)
    print(rec/cnt)
