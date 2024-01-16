#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Import dataset

# In[3]:


df = pd.read_csv(r'C:\Users\paruc\Desktop\DS\3. Jan\2nd,3rd\Project\adult.csv')


# In[4]:


df


# # INFO

# In[57]:


df.shape


# In[58]:


df.info()


# In[59]:


df.describe()


# In[60]:


df.columns


# In[ ]:





# # Exploratory Data Analysis

# #### Repalce ? to NaN

# In[7]:


df[df == '?'] = np.nan


# #### Fill null values with mode

# In[8]:


for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)


# ### Check again for missing values

# In[9]:


df.isnull().sum()


# ## Divide dependent and independent

# In[10]:


X = df.drop(['income'], axis=1)

y = df['income']


# ## Split the data into Train and Test

# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# # Labeling the data 

# In[12]:


from sklearn import preprocessing

cat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in cat:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# ## Scaling the data

# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# ## Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ## PCA

# In[27]:


from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
pca.explained_variance_ratio_


# # Logistic Regression with first 13 features

# In[28]:


X = df.drop(['income','native.country'], axis=1)
y = df['income']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[30]:


categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[31]:


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[32]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 13 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# ## Logistic Regression with first 12 features

# In[33]:


X = df.drop(['income','native.country', 'hours.per.week'], axis=1)
y = df['income']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[35]:


cat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']


# In[36]:


for feature in cat:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[37]:


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[38]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[39]:


print('Logistic Regression accuracy score with the first 12 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # Logistic Regression with first 11 features

# In[40]:


X = df.drop(['income','native.country', 'hours.per.week', 'capital.loss'], axis=1)
y = df['income']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[42]:


cat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in cat:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[43]:


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[44]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[45]:


print('Logistic Regression accuracy score with the first 11 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # Select right number of dimensions

# A better approach is to compute the number of dimensions that can explain significantly large portion of the variance.

# In[46]:


X = df.drop(['income'], axis=1)
y = df['income']


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[48]:


cat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']


# In[49]:


for feature in cat:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[50]:


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)


# In[55]:


pca= PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90)+1
print('The number of dimensions required to preserve 90% of variance is',dim)


# # Plot explained variance ratio with number of dimensions

# An alternative option is to plot the explained variance as a function of the number of dimensions.

# In[56]:


plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,14,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# # Conclusion

# The above plot shows that almost 90% of variance is explained by the first 12 components.

# In[ ]:




