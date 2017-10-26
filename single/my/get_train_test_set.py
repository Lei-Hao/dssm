import pandas as pd
import numpy as np

f = pd.read_csv('action.20170809',sep='\t',header=None)
f.columns=['docid','uid','type']
f=f.dropna()
neg=f[(f.type==-1)|(f.type==0)]
pos=f[(f.type==3)]
pos=pos.groupby('uid').filter(lambda x: len(x) > 5).reset_index(drop=True)
final_train = pos[0:500000]
final_test = pos[500000:]
final_train = final_train.sort_values(['uid'])
final_test = final_test.sort_values(['uid'])
final_test=final_test[final_test['uid'].isin(final_train ['uid'])]
neg_only = neg[neg['uid'].isin(final_test ['uid'])]
final_test=final_test.append(neg_only.iloc[0:200000])
all_doc = pd.read_csv('text.20170809',sep='\t',header=None,error_bad_lines=False)
all_doc.columns = ['docid','title','doc']
all_doc = all_doc.drop_duplicates()
all_doc = all_doc.dropna()
neg_only = neg_only[neg_only['docid'].isin(all_doc['docid'])]
final_train = final_train[final_train['docid'].isin(all_doc['docid'])]
final_test = final_test[final_test['docid'].isin(all_doc['docid'])]
neg_only.to_csv('final_neg',index=False,sep='\t')
final_train.to_csv('final_train',index=False,sep='\t')
final_test.to_csv('final_test',index=False,sep='\t')