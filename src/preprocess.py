"""Build MIT-BIH dataset from PhysioNet using wfdb.
Downloads records 100-234, cuts 187-sample beats around R-peak, maps symbols to AAMI classes N,S,V,F,Q."""
import wfdb, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
symbol_to_class = {'N':0,'L':0,'R':0,'e':0,'j':0,'A':1,'a':1,'J':1,'S':1,'V':2,'E':2,'F':3,'/':4,'f':4,'Q':4}
records = ['100','101','102','103','105','106','107','108','109','111','112','113','114','115','116','117','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
X_list=[]; y_list=[]
for rec in records:
    signals, fields = wfdb.rdsamp(rec, pn_dir='mitdb')
    ann = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
    ecg = signals[:,0]
    for i,sym in enumerate(ann.symbol):
        if sym not in symbol_to_class: continue
        peak=ann.sample[i]; start=peak-90; end=peak+97
        if start<0 or end>=len(ecg): continue
        X_list.append(ecg[start:end]); y_list.append(symbol_to_class[sym])
X=np.array(X_list); y=np.array(y_list)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pd.DataFrame(X_train).assign(**{187:y_train}).to_csv('../data/mitbih_train.csv',header=False,index=False)
pd.DataFrame(X_test).assign(**{187:y_test}).to_csv('../data/mitbih_test.csv',header=False,index=False)
print("Train",X_train.shape,"Test",X_test.shape)
