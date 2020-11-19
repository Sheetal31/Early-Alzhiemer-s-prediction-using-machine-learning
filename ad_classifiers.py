


import numpy as np
import pandas as pd
import seaborn as sns
import keras
import glob3
import time
import timeit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


cross1=pd.read_csv('oasis_longitudinal.csv') 
cross1 = cross1.fillna(method='ffill')
cross2=pd.read_csv('oasis_cross-sectional.csv')
cross2 = cross2.fillna(method='ffill')

#graphics from first dataset
from pylab import rcParams
rcParams['figure.figsize'] = 4,6
cols =  ['Age','MR Delay', 'EDUC', 'SES', 'MMSE', 'CDR','eTIV','nWBV','ASF']
x = cross1.fillna('')
sns_plot = sns.pairplot(x[cols])
plt.show()

#ploting correleation matrix
corr_matrix =cross1.corr()
rcParams['figure.figsize'] = 6,4
sns.heatmap(corr_matrix)
plt.show()


cross1.drop(['MRI ID'], axis=1, inplace=True)
cross1.drop(['Visit'], axis=1, inplace=True)
#cdr=cross1["CDR"]
cross1['CDR'].replace(to_replace=0.0, value='A', inplace=True)
cross1['CDR'].replace(to_replace=0.5, value='B', inplace=True)
cross1['CDR'].replace(to_replace=1.0, value='C', inplace=True)
cross1['CDR'].replace(to_replace=2.0, value='D', inplace=True)

chart.Correlation(select(Data, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF), histogram = TRUE, main = "Correlation between Variables")
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
for x in cross1.columns:
    f = LabelEncoder()
    cross1[x] = f.fit_transform(cross1[x])

cross1.head()


#CLASSIFIERS

train, test  = train_test_split(cross1,test_size = 0.3)
X_train = train[['M/F', 'Age', 'EDUC', 'SES',  'eTIV', 'ASF']]
y_train = train.CDR
X_test = test[['M/F', 'Age', 'EDUC', 'SES',  'eTIV',  'ASF']]
y_test = test.CDR

from sklearn.preprocessing import StandardScaler
#define the scalar
scalar = StandardScaler().fit(X_train)
#scale the train set
X_train = scalar.transform(X_train)
#scale the test set
X_test = scalar.transform(X_test)


y_train=np.ravel(y_train)
X_train=np.asarray(X_train)

y_test=np.ravel(y_test)
X_test=np.asarray(X_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression()
classifierLR.fit(X_train, y_train)
predictionLR = classifierLR.predict(X_test)
m = classifierLR.score(X_train, y_train)
print('Training accuracy of LR = ', m)
n = classifierLR.score(X_test, y_test)
print ('Test accuracyof LR = ',n)


#Descicion Trees classifier
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(max_depth = 12)
classifierDT.fit(X_train,y_train)
predictionDT = classifierDT.predict(X_test)
o = classifierDT.score(X_train, y_train)
print('Training accuracy of DT = ', o)
p = classifierDT.score(X_test, y_test)
print ('Test accuracyof DT = ',p)


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
predictionKnn = knn.predict(X_test)
q = knn.score(X_train, y_train)
print('Training accuracy of Knn = ', q)
r = knn.score(X_test, y_test)
print ('Test accuracyof Knn = ',r)


#support vector machine
from sklearn.svm import SVC
svc = SVC(kernel = "linear", C = 0.01)
svc.fit(X_train,y_train)
predictionSvc = svc.predict(X_test)
s = svc.score(X_train, y_train)
print('Training accuracy of Svc = ', s)
t = svc.score(X_test, y_test)
print ('Test accuracyof Svc = ',t)

#CONCAT THE DATASET

#lets encode second dataset
for x in cross2.columns:
    f = LabelEncoder()
    cross2[x] = f.fit_transform(cross2[x])

#concanting both datasets
df = pd.concat([cross1,cross2])
df = df.fillna(method='ffill')
df.head()

train, test = train_test_split(cross1, test_size=0.3)
X_train1 = train[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_train1 = train.CDR
X_test1 = test[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_test1 = test.CDR

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train1)

# Scale the train set
X_train1 = scaler.transform(X_train1)

# Scale the test set
X_test1 = scaler.transform(X_test1)

y_train1=np.ravel(y_train1)
X_train1=np.asarray(X_train1)

y_test1=np.ravel(y_test1)
X_test1=np.asarray(X_test1)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR1 = LogisticRegression()
classifierLR1.fit(X_train1, y_train1)
predictionLR1 = classifierLR1.predict(X_test1)
m1 = classifierLR1.score(X_train1, y_train1)
print('Training accuracy of LR after concat = ', m1)
n1 = classifierLR1.score(X_test1, y_test1)
print ('Test accuracyof LR after concat = ',n1)


#Descicion Trees classifier
from sklearn.tree import DecisionTreeClassifier
classifierDT1 = DecisionTreeClassifier(max_depth = 12)
classifierDT1.fit(X_train1,y_train1)
predictionDT1 = classifierDT1.predict(X_test1)
o1 = classifierDT1.score(X_train1, y_train1)
print('Training accuracy of DT after concat = ', o1)
p1 = classifierDT1.score(X_test1, y_test1)
print ('Test accuracyof DT after concat = ',p1)


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors = 2)
knn1.fit(X_train1,y_train1)
predictionKnn1 = knn1.predict(X_test1)
q1 = knn1.score(X_train1, y_train1)
print('Training accuracy of Knn after concat = ', q1)
r1 = knn1.score(X_test1, y_test1)
print ('Test accuracyof Knn after concat = ',r1)


#support vector machine
from sklearn.svm import SVC
svc1 = SVC(kernel = "linear", C = 0.01)
svc1.fit(X_train1,y_train1)
predictionSvc1 = svc1.predict(X_test1)
s1 = svc1.score(X_train1, y_train1)
print('Training accuracy of Svc after concat = ', s1)
t1 = svc1.score(X_test1, y_test1)
print ('Test accuracyof Svc after concat = ',t1)