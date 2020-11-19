


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('oasis_cross-sectional.csv')
data.drop(["Hand","Delay"], axis=1,inplace=True)
data.columns = ["id","gender","age","education","socio_economic_status","mini_mental_state_examination","clinical_dementia_rating",
           "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]
data.gender = [1 if each == "F" else 0 for each in  data.gender]
data.info()

#data imputation
data.isnull().sum()   ## knowing number of missing values for each variable
def impute_median(series):
	return series.fillna(series.median())	#find the advantage of median over mean

data.education = data["education"].transform(impute_median)		#transform returns a transformed dataframe with new values 
data.socio_economic_status = data["socio_economic_status"].transform(impute_median)
data.mini_mental_state_examination = data["mini_mental_state_examination"].transform(impute_median)
data.clinical_dementia_rating = data["clinical_dementia_rating"].transform(impute_median)

#correlation matrix plot
plt.figure(figsize = (8,6))
sns.heatmap(data.iloc[:,0:10].corr(), annot = True, fmt = ".0%")
plt.show()

#scatter plot
plt.figure(figsize = (8,6))
plt.scatter(data['age'],data['atlas_scaling_factor'],c = 'black', label = 'Bad')
plt.xlabel('age')
plt.ylabel('atlas_scaling_factor')
plt.show()



#bar plot
def bar_chart(feature):
	Demented = data[data['Group']==1][feature].value_counts()
	Nondemented = data[data['Group']==0][feature].value_counts()
	data_bar = pd.DataFrame([Demented,Nondemented])
	data_bar.index = ['Demented','Nondemented']
	data_bar.plot(kind='bar',stacked=True, figsize=(8,5))


bar_chart('gender')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')
plt.show()


#distribution with facet plot
facet= sns.FacetGrid(data,hue="Group", aspect=3)
facet.map(sns.kdeplot,'atlas_scaling_factor',shade= True)
facet.set(xlim=(0, data.atlas_scaling_factor.max()))
facet.add_legend()
plt.xlim(0.5, 2)
plt.show()


#Kmeans clustering | REMEBER loc is label based indexing and iloc is location based indexing
data2 = data.loc[:,['age','atlas_scaling_factor']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.figure(figsize =(8,6))
plt.scatter(data['age'],data['atlas_scaling_factor'], c = labels)
plt.xlabel('age')
plt.ylabel('atlas_scaling_factor')
plt.show()



data["class"] = ["Normal" if (each > 0.8) & (each < 1.2) else "Abnormal" for each in  data.atlas_scaling_factor]
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.figure(figsize =(5,5))
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


#dendogram
data3 = data.drop('class',axis = 1)
data3 = pd.get_dummies(data3)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[0:50,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()


# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)


# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)
plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()
# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()


















