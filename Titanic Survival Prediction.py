#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('D:/mlp-course-material/MODULE 9 - Mini Projects/1. Titanic Kaggle Challenge/titanic data.csv')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


heatmap = sns.heatmap(data[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True)


# In[10]:


data['SibSp'].unique()


# In[11]:


sibsp_plot = sns.catplot(x = 'SibSp', y = 'Survived', data = data, kind = 'bar', height = 8)


# In[12]:


age_visual = sns.FacetGrid(data, col = 'Survived', height = 7)
age_visual = age_visual.map(sns.histplot, 'Age')
age_visual = age_visual.set_ylabels('Survival probability')


# In[13]:


age_plot = sns.barplot(x = 'Sex', y = 'Survived', data = data, )
plt.figure(figsize = (12, 5))


# In[14]:


data[['Sex', 'Survived']].groupby('Sex').mean()


# In[15]:


Pclass = sns.catplot(x = 'Pclass', y = 'Survived', data = data, kind = 'bar', height = 5)


# In[16]:


data[['Pclass', 'Survived']].groupby('Pclass').mean()


# In[17]:


a = sns.catplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = data,  height = 7, kind = 'bar')


# In[18]:


data['Embarked'].isnull().sum()


# In[19]:


data['Embarked'].value_counts()


# In[20]:


data['Embarked'] = data['Embarked'].fillna('S')


# In[21]:


data['Embarked'].isnull().sum()


# In[22]:


sns.catplot(x= 'Embarked', y = 'Survived', data = data, kind = 'bar')


# In[23]:


sns.catplot(x= 'Pclass', col = 'Embarked', data = data, height = 7, kind = 'count')


# In[24]:


sns.catplot(x ='Sex', col = 'Embarked', data = data, kind = 'count', height = 7)


# In[25]:


data = pd.read_csv('D:/mlp-course-material/MODULE 9 - Mini Projects/1. Titanic Kaggle Challenge/titanic data.csv')


# In[26]:


data.head()


# In[27]:


null = data['Age'].isnull().sum()


# In[28]:


null


# In[29]:


mean = data['Age'].mean()
std = data['Age'].std()


# In[30]:


mean


# In[31]:


std


# In[32]:


rand_age = np.random.randint(mean-std, mean+std, size = null)
age_slice = data['Age'].copy()

age_slice[np.isnan(age_slice)] = rand_age
data['Age'] = age_slice


# In[33]:


data['Age'].isnull().sum()


# In[34]:


data['Embarked'] = data['Embarked'].fillna('S')


# In[35]:


col_to_drop = ['PassengerId',  'Cabin','Name', 'Ticket']
data.drop(col_to_drop, axis = 1, inplace = True)


# In[36]:


data.head()


# In[37]:


genders = {'male':0, 'female': 1}
data['Sex'] = data['Sex'].map(genders)


# In[38]:


ports = {'S':0, 'C':1, 'Q':2}
data['Embarked'] = data['Embarked'].map(ports)


# In[39]:


data.head()


# In[40]:


x = data.drop(data.columns[[0]],  axis = 1 )
y = data['Survived']


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25,random_state = 0)


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


sc = StandardScaler()

x = sc.fit_transform(x)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR


# In[46]:


logreg = LogisticRegression()
svc_classifier = SVR()
tree_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(5)
rf_classifier  = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)


# In[53]:


logreg.fit(xtrain, ytrain)
svc_classifier.fit(xtrain, ytrain)
tree_classifier.fit(xtrain, ytrain)
knn_classifier.fit(xtrain, ytrain)
rf_classifier.fit(xtrain, ytrain)


# In[54]:


logreg_pred = logreg.predict(xtest)
svc_classifier_pred = svc_classifier.predict(xtest)
tree_classifier_pred = tree_classifier.predict(xtest)
knn_classifier_pred = knn_classifier.predict(xtest)
rf_classifeir_pred = rf_classifier.predict(xtest)


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


accuracy_score(ytest, logreg_pred)


# In[57]:


accuracy_score(ytest, tree_classifier_pred )


# In[58]:


accuracy_score(ytest, knn_classifier_pred)


# In[59]:


accuracy_score(ytest, rf_classifeir_pred)


# In[ ]:




