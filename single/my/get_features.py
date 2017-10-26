## coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
from scipy import sparse as ssp
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder,StandardScaler,Normalizer
import jieba
NEG = 2
doc_all = []
uids = []
docids = []
doc_dict = {}
fenci = False
line_no = 0
with open('all_doc') as doc_all_fenci:
    for line in doc_all_fenci:
        #print line
        try:
            docid, title, doc= line.strip().split('\t')
        except Exception as err:
            docid, title = line.strip().split('\t')
            doc = title
        #doc = title
        #print docid
        if docid == 'docid':
            continue
        #if docid not in doc_dict:
        doc_dict[docid] = line_no
        line_no += 1
        #uids.append(uid)
        #docids.append(docid)
        #print doc.decode('utf-8')
        doc_all.append(doc.decode('utf-8'))

if fenci:
        # define this function to print a list with Chinese
    def PrintListChinese(list):
        for i in range(len(list)):
            print list[i],
    # segment word with jieba
    doc_seg=[]
    for i in range(len(doc_all)):
        doc_seg.append([' '.join(list(jieba.cut(doc_all[i][0:-2],cut_all=False)))])
        print i

    # to test the segment result
    #PrintListChinese(fileTrainSeg[10])

    # save the result
    with open('doc_seg','wb') as fw:
        for i in range(len(doc_seg)):
           #fw.write(uids[i])
           #fw.write('\t')
           #fw.write(docids[i])
           #fw.write('\t')
           fw.write(doc_seg[i][0].encode('utf-8'))
           fw.write('\n')
docs2tf_idf = []
f = open('doc_seg', 'r')
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
for i in range(len(stpwrdlst)):
    stpwrdlst[i] = stpwrdlst[i].decode('utf-8')
batch_uids = []
batch_docids = []
for line in f:
    terms_str = line.strip()
    docs2tf_idf.append(terms_str)
#print docs2tf_idf
#uid_dict_reverse = dict(zip(uid_dict.values(), uid_dict.keys()))
ti = TfidfVectorizer(ngram_range=(1,1),stop_words=stpwrdlst)
#ti = TfidfVectorizer(ngram_range=(1,1))
docs_tfidf_csr = ti.fit_transform(docs2tf_idf).tocsr()
pd.to_pickle(docs_tfidf_csr, 'doc_train.pkl')
#docs_tfidf = docs_tfidf_csr.toarray()
#print docs_tfidf.shape

wordlist = ti.get_feature_names()
with open('tf-idf_feature_names.txt','wb') as fw:
    for i in range(len(wordlist)):
        fw.write(wordlist[i].encode('utf-8') + '\n')
#for i in range(len(docs_tfidf)):  
#    for j in range(len(wordlist)):  
#        print wordlist[j].encode('utf-8'),docs_tfidf[i][j]

#f.seek(0)
line_no = 0
f = open('final_train','r')
for line in f:
    docid,uid, _ = line.strip().split('\t')
    if uid == 'uid':
        continue
    line_no += 1
    batch_uids.append(uid)
    batch_docids.append(docid)

def generate_doc(df, name, concat_name):
    #print df.columns
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name, '%s_doc'%concat_name]
    return res

train = pd.read_csv('final_train',sep='\t',dtype={"uid":str,'type':str,'docid':str, 'doc':str})
le = LabelEncoder()
train['docid'] = le.fit_transform(train['docid'].values)
#print train
iid_doc = generate_doc(train, name='uid', concat_name='docid')

tfidf_u = TfidfVectorizer(ngram_range=(1,1))
iid_tfidf_csr = tfidf_u.fit_transform(iid_doc['docid_doc'].values).tocsr()
pd.to_pickle(iid_tfidf_csr,'user_train.pkl')
#iid_tfidf = iid_tfidf_csr.toarray()
#print iid_tfidf.shape

#print uid_dict_reverse
gen_train = False
gen_test = True
#neg = pd.read_csv('final_neg',sep='\t')
print '+++++++++++++++++++++'
#print neg[neg.uid=='863603032342187_460014649963504'].docid
#print neg[neg.uid=='863603032342187_460014649963504'].docid[0]
if gen_train:
   print('gen train...')
   ssp_uid = []
   ssp_pos_doc = []
   for i in range(len(batch_uids)):
       #print(str(batch_uids[i]))
       print(i)
       ssp_uid.append(iid_tfidf_csr[iid_doc[iid_doc.uid == batch_uids[i]].index[0]])
       ssp_pos_doc.append(docs_tfidf_csr[doc_dict[batch_docids[i]]])
       #for k in range(NEG):
       #    now_neg = neg[neg.uid==batch_uids[i]].docid
       #    now_neg_docid = now_neg.iloc[random.randint(0,len(now_neg)-1)]
           #ssp_neg_doc.append(docs_tfidf_csr[doc_dict[now_neg_docid]]
       #    if k == 0:
       #        ssp_neg_doc_tmp = docs_tfidf_csr[doc_dict[now_neg_docid]]
       #    else:
       #        ssp_neg_doc_tmp = ssp.hstack([ssp_neg_doc_tmp, docs_tfidf_csr[doc_dict[now_neg_docid]]])
       #ssp_neg_doc.append(ssp_neg_doc_tmp)
   ssp_uid = ssp.vstack(ssp_uid)
   ssp_pos_doc = ssp.vstack(ssp_pos_doc)
   #ssp_neg_doc = ssp.vstack(ssp_neg_doc)
   print ssp_uid.shape
   print ssp_pos_doc.shape
   #train = ssp.hstack([ssp_uid, ssp_pos_doc,ssp_neg_doc])
   train = ssp.hstack([ssp_uid, ssp_pos_doc])
   pd.to_pickle(ssp_uid, 'train_user.pkl')
   pd.to_pickle(ssp_pos_doc, 'train_doc.pkl')

if gen_test:
    f_test = open('test_tf-idf2.txt','wb')
    # generate testset
    batch_uids_test=[]
    batch_docids_test=[]

    line_no = 0
    f = open('final_test','r')
    for line in f:
        #print uid,docid
        docid, uid, _ = line.strip().split('\t')
        if uid == 'uid':
            continue
        line_no += 1
        batch_uids_test.append(uid)
        batch_docids_test.append(docid)
    print('gen test...')
    ssp_uid = []
    ssp_doc = []
    for i in range(len(batch_uids_test)):
        #print(str(batch_uids_test[i]))
        
        if i % 1000==0:
            print i
        ssp_uid.append(iid_tfidf_csr[iid_doc[iid_doc.uid == batch_uids_test[i]].index[0]])
        ssp_doc.append(docs_tfidf_csr[doc_dict[batch_docids_test[i]]])
        #for k in range(NEG):
        #    now_neg = neg[neg.uid==batch_uids_test[i]].docid
        #    now_neg_docid = now_neg.iloc[random.randint(0,len(now_neg)-1)]
        #    if k == 0:
        #        ssp_neg_doc_tmp = docs_tfidf_csr[doc_dict[now_neg_docid]]
        #    else:
        #        ssp_neg_doc_tmp = ssp.hstack([ssp_neg_doc_tmp, docs_tfidf_csr[doc_dict[now_neg_docid]]])
       # ssp_neg_doc.append(ssp_neg_doc_tmp)
    ssp_uid = ssp.vstack(ssp_uid)
    ssp_doc = ssp.vstack(ssp_doc)
    #ssp_neg_doc = ssp.vstack(ssp_neg_doc)
    #test = ssp.hstack([ssp_uid, ssp_pos_doc, ssp_neg_doc])
    test = ssp.hstack([ssp_uid, ssp_doc])
    pd.to_pickle(ssp_uid, 'test_user.pkl')
    pd.to_pickle(ssp_doc, 'test_doc.pkl')
    
    """
    print('gen neg test...')
    ssp_neg_doc = []
    for i in range(len(batch_uids_test)):
        #print(str(batch_uids_test[i]))
        print i
        #ssp_uid.append(iid_tfidf_csr[iid_doc[iid_doc.uid == batch_uids_test[i]].index[0]])
        #ssp_pos_doc.append(docs_tfidf_csr[doc_dict[batch_docids_test[i]]])
        #for k in range(NEG):
       # ssp_uid.append(iid_tfidf_csr[iid_doc[iid_doc.uid == batch_uids_test[i]].index[0]])
        now_neg = neg[neg.uid==batch_uids_test[i]].docid
        now_neg_docid = now_neg.iloc[random.randint(0,len(now_neg)-1)]
        #    if k == 0:
        ssp_neg_doc_tmp = docs_tfidf_csr[doc_dict[now_neg_docid]]
        #    else:
        #        ssp_neg_doc_tmp = ssp.hstack([ssp_neg_doc_tmp, docs_tfidf_csr[doc_dict[now_neg_docid]]])
        ssp_neg_doc.append(ssp_neg_doc_tmp)
    #ssp_uid = ssp.vstack(ssp_uid)
    #ssp_pos_doc = ssp.vstack(ssp_pos_doc)
    ssp_neg_doc = ssp.vstack(ssp_neg_doc)
    #test = ssp.hstack([ssp_uid, ssp_pos_doc, ssp_neg_doc])
    #test = ssp.hstack([ssp_uid, ssp_neg_doc])
    #pd.to_pickle(ssp_uid, 'test_neg_user.pkl')
    pd.to_pickle(ssp_neg_doc, 'test_neg_doc.pkl')
    """
#pd.to_pickle(train, 'train.pkl')
#pd.to_pickle(test, 'test.pkl')

