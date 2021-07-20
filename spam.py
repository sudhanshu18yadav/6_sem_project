#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv')
df.head()


# In[4]:


df.groupby('Category').describe()


# In[5]:


df['spam'] = df['Category'].apply(lambda x: 1 if x== 'spam' else 0)


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size = 0.2)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()[:2]


# In[9]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count, y_train)


# In[11]:


email = [
    'Hey mohan, can we get together to eatch football game tommorow?',
    'Upto 20% discount on parking , exclusive offer just for you. Dont ,miss this is reward!'
]

email_count = v.transform(email)
model.predict(email_count)


# In[12]:


x_test_count = v.transform(x_test)
print(model.score(x_test_count, y_test))

# In[ ]:




