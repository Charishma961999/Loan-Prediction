#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[20]:


train = pd.read_csv(r"C:\Users\CHARISHMA\Downloads\train_u6lujuX_CVtuZ9i.csv")
train.head()


# In[21]:


test = pd.read_csv(r"C:\Users\CHARISHMA\Downloads\test_Y3wMUE5_7gLdaTN.csv")
test.head()


# In[23]:


train.columns


# In[24]:


test.columns


# In[25]:


train.dtypes


# In[26]:


train.shape


# In[27]:


test.shape


# In[31]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[33]:


train['Loan_Amount_Term'].value_counts()


# In[35]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[36]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[37]:


train.isnull().sum()


# In[38]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[39]:


train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log']=np.log(test['LoanAmount'])


# In[40]:


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


# In[41]:


X = train.drop('Loan_Status',1)
y = train.Loan_Status


# In[42]:


X = pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


# In[43]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()


# In[45]:


pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
0.7891891891891892


# In[46]:


pred_test = model.predict(test)


# In[49]:


from sklearn.model_selection import StratifiedKFold


# In[53]:


i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1)
for train_index,test_index in kf.split(X,y):
 print ('\n{} of kfold {}'.format(i,kf.n_splits))
 xtr,xvl = X.loc[train_index],X.loc[test_index]
 ytr,yvl = y[train_index],y[test_index]
 model = LogisticRegression(random_state=1)
 model.fit(xtr,ytr)
 pred_test=model.predict(xvl)
 score=accuracy_score(yvl,pred_test)
 mean += score
 print ('accuracy_score',score)
 i+=1
 pred_test = model.predict(test)
 pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))


# In[56]:


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label='validation, auc='+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[59]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']


# In[61]:


sns.distplot(train['Total_Income'])


# In[63]:


train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log'])
test['Total_Income_log'] = np.log(test['Total_Income'])


# In[64]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[65]:


sns.distplot(train['EMI'])


# In[ ]:




