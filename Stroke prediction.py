#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[58]:


data = pd.read_csv(r'E:\Stroke prediction\archive\healthcare-dataset-stroke-data.csv')


# In[59]:


data.head()


# In[60]:


data.isnull()


# In[61]:


data.sum()


# In[62]:


data.isna().sum()


# In[63]:


data = data.interpolate()
data


# In[64]:


data.isna().sum()


# In[65]:


data = data.drop(columns=['id'])


# In[66]:


# Label encode 'gender':
data['gender'] = data['gender'].astype('category')
data['gender'] = data['gender'].cat.codes

# Label encode 'ever_married':
data['ever_married'] = data['ever_married'].astype('category')
data['ever_married'] = data['ever_married'].cat.codes

# Label encode 'work_type':
data['work_type'] = data['work_type'].astype('category')
data['work_type'] = data['work_type'].cat.codes

# Label encode 'Residence_type'
data['Residence_type'] = data['Residence_type'].astype('category')
data['Residence_type'] = data['Residence_type'].cat.codes

# Label encode 'smoking_status':
data['smoking_status'] = data['smoking_status'].astype('category')
data['smoking_status'] = data['smoking_status'].cat.codes


# In[67]:


data


# In[68]:


data.dtypes


# In[69]:


x = data.iloc[:, :-1].values


# In[70]:


print(x)


# In[71]:


y = data.iloc[:,10] .values 


# In[72]:


print(y)


# In[73]:


from sklearn.model_selection import train_test_split

# Split our data into test and train sets
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=17)


# In[74]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =  sc_x.transform(x_test)


# In[76]:


from sklearn.linear_model import LogisticRegression


# In[77]:


classifier = LogisticRegression()


# In[78]:


classifier.fit(x_train, y_train)


# In[79]:


y_pred = classifier.predict(x_test)


# In[80]:


y_pred


# In[81]:


from sklearn.metrics import accuracy_score


# In[82]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:




