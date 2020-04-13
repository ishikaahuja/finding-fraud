import pandas as pd
import numpy as np


data=pd.read_csv("set.csv")

dummy=pd.get_dummies(data['type'])
dataset=pd.concat([data,dummy],axis=1)
#dataset.head()
dataset.drop(['type'],axis=1,inplace=True)
y=dataset[['isFraud']]
x=dataset.drop(['isFraud','nameOrig','nameDest'],axis=1)

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7)
import statsmodels.api as sm

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()

lg.fit(x_train,y_train)
y_pred=lg.predict_proba(x_test)
y_pred_df=pd.DataFrame(y_pred)
y_pred_1 = y_pred_df.iloc[:,[1]]
y_test_df=pd.DataFrame(y_test)
y_test_df['nameOrig'] = y_test_df.index
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
y_pred_final= y_pred_final.rename(columns={ 1 : 'fraud_Prob'})
print(y_pred_final.head())
y_pred_final['predicted'] = y_pred_final.fraud_Prob.map( lambda x: 1 if x > 0.2 else 0)
print(y_pred_final.head())
numbers = [float(x)/10 for x in range(10)]
'''
for i in numbers:
    y_pred_final[i]= y_pred_final.fraud_Prob.map( lambda x: 1 if x > i else 0)
print(y_pred_final.head())
cutoff=pd.DataFrame(columns=['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix
from sklearn import metrics
for i in numbers:
    cm1=metrics.confusion_matrix(y_pred_final.isFraud,y_pred_final[i])
    total1=sum(sum(cm1))
    accuracy=(cm1[0,0]+cm1[1,1])/total1
    speci=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi=cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff.loc[i]=[i,accuracy,sensi,speci]
cutoff.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
'''
def getop(path):
    test2=pd.read_csv(path)
    dummy=pd.DataFrame(columns=['nameOrig','CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT','TRANSFER'])
    for i in range(0,test2.shape[0]):
        if(test2['type'][i]=='CASH_IN'):
            dummy.loc[i]=[test2['nameOrig'][i],1,0,0,0,0]
        elif(test2['type'][i]=='CASH_OUT'):
            dummy.loc[i]=[test2['nameOrig'][i],0,1,0,0,0]
        elif(test2['type'][i]=='DEBIT'):
            dummy.loc[i]=[test2['nameOrig'][i],0,0,1,0,0]
        elif(test2['type'][i]=='PAYMENT'):
            dummy.loc[i]=[test2['nameOrig'][i],0,0,0,1,0]
        elif(test2['type'][i]=='TRANSFER'):
            dummy.loc[i]=[test2['nameOrig'][i],0,0,0,0,1]
        else:
            print("unrecognized input")
            
            
            
            
    test2=pd.concat([test2,dummy],axis=1)
#dataset.head()
    test2.drop(['type'],axis=1,inplace=True)
    train2=test2.drop(['nameOrig','nameDest'],axis=1)
    y_prednew=lg.predict_proba(train2)
    
    #print(y_prednew)
    ydf=pd.DataFrame(y_prednew)
    y_p = ydf.iloc[:,[1]]
    #ydf['nameOrig'] = ydf.index
    print(y_p)
    y_p= y_p.rename(columns={ 1 : 'fraud_Prob'})
    y_p['predicted'] = y_p.fraud_Prob.map( lambda x: 1 if x > 0.2 else 0)
    print(y_p)
getop(r"C:\Users\ISHIKA\4th\firsttry.csv")
    

    
