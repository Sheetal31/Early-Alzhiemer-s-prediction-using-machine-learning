


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

sns.set()

cross1 = pd.read_csv('oasis_longitudinal.csv') 
cross1 = cross1.fillna(method='ffill')
cross2 = pd.read_csv('oasis_cross-sectional.csv')
cross2 = cross2.fillna(method='ffill')


#lil data cleaning
cross1.drop(['Subject ID','Visit','Hand','MR Delay'],axis=1,inplace=True)
cross1.columns = ["id","group", "gender","age","education","socio_economic_status","mini_mental_state_examination","clinical_dementia_rating",
                  "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]
cross2.drop(['Hand','Delay'],axis=1,inplace=True)
cross2.columns = ["id","gender","age","education","socio_economic_status","mini_mental_state_examination","clinical_dementia_rating",
                  "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]
group = ["Nondemented" if (each > 0.8) & (each < 1.2) else "Demented" for each in  cross2.atlas_scaling_factor]
cross2['group'] = group 

#concanting both datasets
data = pd.concat([cross1,cross2])

#lil more data processing
data.columns = ["id","group", "gender","age","education","socio_economic_status","mini_mental_state_examination","clinical_dementia_rating",
                "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]
data.gender = [0 if each == "F" else 1 for each in  data.gender]
data.group = [0 if each == "Nondemented" else 1 for each in  data.group]

data.isnull().sum()
def impute_median(series):
	return series.fillna(series.median())

data.education = data['education'].transform(impute_median)
data.socio_economic_status = data['socio_economic_status'].transform(impute_median)
data.mini_mental_state_examination = data['mini_mental_state_examination'].transform(impute_median)
data.clinical_dementia_rating = data['clinical_dementia_rating'].transform(impute_median)

#correlation matrix plot
plt.figure(figsize = (8,6))
sns.heatmap(data.iloc[:,0:11].corr(), annot = True, fmt = ".0%")
plt.show()

#bar plot
def bar_chart(feature):
	Demented = data[data['group']==1][feature].value_counts()
	Nondemented = data[data['group']==0][feature].value_counts()
	data_bar = pd.DataFrame([Demented,Nondemented])
	data_bar.index = ['Demented','Nondemented']
	data_bar.plot(kind='bar',stacked=True, figsize=(8,6))

#bar plot to see variation in genders
bar_chart('gender')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')
plt.show()

#facet plots
facet= sns.FacetGrid(data,hue="group", aspect=3)
facet.map(sns.kdeplot,'estimated_total_intracranial_volume',shade= True)
facet.set(xlim=(0, data['estimated_total_intracranial_volume'].max()))
facet.add_legend()
plt.xlim(400, 2500)
plt.show()

facet= sns.FacetGrid(data,hue="group", aspect=3)
facet.map(sns.kdeplot,'normalize_whole_brain_volume',shade= True)
facet.set(xlim=(0, data['normalize_whole_brain_volume'].max()))
facet.add_legend()
plt.xlim(0.6,0.9)
plt.show()
#The chart indicates that Nondemented group has 
#higher brain volume ratio than Demented group. 
#This is assumed to be because the diseases affect 
#the brain to be shrinking its tissue.

#AGE. Nondemented =0, Demented =1
facet= sns.FacetGrid(data,hue="group", aspect=3)
facet.map(sns.kdeplot,'age',shade= True)
facet.set(xlim=(0, data['age'].max()))
facet.add_legend()
plt.xlim(0,100)
plt.show()
#There is a higher concentration of 70-80 years old in the 
#Demented patient group than those in the nondemented patients. 
#We guess patients who suffered from that kind of disease has 
#lower survival rate so that there are a few of 90 years old.


#'EDUC' = Years of Education
# Nondemented = 0, Demented =1
facet= sns.FacetGrid(data,hue="group", aspect=3)
facet.map(sns.kdeplot,'education',shade= True)
facet.set(xlim=(data['education'].min(), data['education'].max()))
facet.add_legend()
plt.ylim(0, 0.16)
plt.show()


#lets encode the dataset
for x in data.columns:
    f = LabelEncoder()
    data[x] = f.fit_transform(data[x])


#CLASSIFIERS
from sklearn.model_selection import train_test_split
train, test  = train_test_split(data, test_size = 0.3)
X_train = train[['gender', 'age', 'education', 'socio_economic_status',  'estimated_total_intracranial_volume', 'atlas_scaling_factor']]
y_train = train.clinical_dementia_rating
X_test = test[['gender', 'age', 'education', 'socio_economic_status',  'estimated_total_intracranial_volume', 'atlas_scaling_factor']]
y_test = test.clinical_dementia_rating

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









