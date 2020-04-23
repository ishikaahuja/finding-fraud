import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\ISHIKA\4th\Financial Fraud Dataset\set.csv")
dummy=pd.get_dummies(dataset["type"])
dataset=pd.concat([dummy,dataset],axis=1)
dataset.drop(["type"],axis=1,inplace=True)
print(dataset.head())

x=dataset.drop(['isFraud','nameOrig','nameDest'],axis=1)
y=dataset[['isFraud']]
print(x.columns)
print(y)
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7,random_state=100)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)
from sklearn.svm import SVC
classifier=SVC(kernel="poly",degree=3,random_state=100)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
from sklearn import metrics
from sklearn.metrics  import confusion_matrix
cutoff=pd.DataFrame(y_pred)
cm1=confusion_matrix(y_test, y_pred)
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
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
X1 ,X2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01), np.arange(start = x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
X1.shape
X2.shape
y_pred.shape
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).reshape(X1.shape)),alpha =0.75 ,cmap =ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

def getop(path):
    cv=pd.read_csv(path)
    if
'''