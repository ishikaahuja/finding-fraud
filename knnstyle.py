import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColomap
from matplotlib.colors import ListedColormap
from sklearn import neighbors
dataset=pd.read_csv(r"C:\Users\ISHIKA\4th\Financial Fraud Dataset\set.csv")
dummy=pd.get_dummies(dataset["type"])
dataset=pd.concat([dummy,dataset],axis=1)
dataset.drop(["type"],axis=1,inplace=True)
dataset.head()
x=dataset.drop(['isFraud','nameOrig','nameDest'],axis=1)
y=dataset[['isFraud']]
#y=y.ravel
clf = neighbors.KNeighborsClassifier(15)




from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8,random_state=100)
clf.fit(x_train,y_train)
Z = clf.predict(x_test)


'''
y_test.shape
Z=Z.reshape(314573,1)
Z.shape

from sklearn.metrics  import confusion_matrix
cutoff=pd.DataFrame(Z)
cm1=confusion_matrix(y_test, Z)
print(cm1)
total1=(sum(sum(cm1)))
accuracy=(cm1[0,0]+cm1[1,1])/total1
speci=cm1[0,0]/(cm1[0,0]+cm1[0,1])
sensi=cm1[0,0]/(cm1[1,0]+cm1[0,0])
print(accuracy)
print(speci)
print(sensi)
cmp=2*(speci*sensi)/(speci+sensi)
print(cmp)
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
    y_prednew=clf.predict(train2)
    
    #print(y_prednew)
    ydf=pd.DataFrame(y_prednew)
    #print(ydf)
    y_p = ydf.iloc[:,[0]]
    #ydf['nameOrig'] = ydf.index
    #print(y_p)
    y_p= y_p.rename(columns={ 0 : 'fraud_Prob'})
    #y_p['predicted'] = y_p.fraud_Prob.map( lambda x: 1 if x > 0.2 else 0)
    print(y_p)
getop(r"C:\Users\ISHIKA\4th\Book1.csv")
   
