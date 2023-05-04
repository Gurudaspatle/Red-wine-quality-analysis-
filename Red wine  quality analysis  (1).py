#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')
import seaborn as sns



#  # Data Analysation and Visualisation  Part #   

# In[ ]:





# In[2]:


wine=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\1788410-1767134-1729261-1613779-Red_wine__(1).csv")
wine.head()


# In[3]:


wine.shape


# In[4]:


plt.figure(figsize = (12,6))
sns.countplot(wine['quality'])
plt.show()


# In[5]:


plt.figure(figsize = (12,6))
sns.barplot(x='quality', y = 'alcohol', data = wine, palette = 'inferno')
plt.show()


# In[6]:


plt.figure(figsize = (12,6))
sns.scatterplot(x='citric acid', y = 'pH', data = wine)
plt.show()


# In[7]:


plt.figure(figsize = (12,6))
sns.pairplot(wine)
plt.show()


# In[8]:


plt.figure(figsize = (12,6))
sns.heatmap(wine.corr())
plt.show()


# In[9]:


x=wine.drop(['quality'], axis=1)
y=wine['quality']


# # Data Preprocessing

# In[24]:


from imblearn.over_sampling import SMOTE
os=SMOTE()
x_res,y_res=os.fit_sample(x, y)


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res,y_res,test_size=0.2, random_state=0)


# In[19]:


from sklearn.preprocessing import StandardScaler

stdscale = StandardScaler().fit(x_train)
x_train_std = stdscale.transform(x_train)
x_test_std = stdscale.transform(x_test)


# In[20]:


from sklearn.metrics import accuracy_score


# # # Logistic Regression

# In[16]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_std, y_train)
predictions = lr.predict(x_test_std)
accuracy_score(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




