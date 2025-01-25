#!/usr/bin/env python
# coding: utf-8

# # Build the Lookalike Model

# In[4]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ## 1. Data Preparation:
# - Aggregate Transaction Data to derive new features
# - Merge this data products dataset also

# In[7]:


customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


# In[8]:


final_data = transactions.merge(customers , on="CustomerID").merge(products , on='ProductID')


# In[9]:


final_data.head()


# <h3 style="color:green; font-weight:bolder; text-align:center;">Extracting the required features like total spent value ,unique products bought , average product price , numbers of transactions</h3>

# In[12]:


customer_features = final_data.groupby('CustomerID').agg(
    total_spend=("TotalValue" ,'sum'),
    unique_products = ("ProductID" , "nunique"),
    avg_product_price = ("Price_x" , "mean"),
    transaction_count = ("TransactionID" ,'count')
).reset_index()


# In[13]:


customer_features.head()


# <h3 style="color:green; font-weight:bolder; text-align:center;">Merging Region with customer features to get the similarity between customers from a specific region</h1>

# In[19]:


customer_features= customer_features.merge(customers[["CustomerID" , "Region"]] , on="CustomerID")


# In[20]:


customer_features.head()


# In[21]:


customer_features = pd.get_dummies(customer_features , columns=['Region'] , drop_first=True)


# In[22]:


scaler = StandardScaler()
numerical_cols = ["total_spend" , "unique_products" , "avg_product_price" , "transaction_count"]
customer_features[numerical_cols] = scaler.fit_transform(customer_features[numerical_cols])


# In[23]:


similarity_matrix =cosine_similarity(customer_features.drop('CustomerID' , axis=1))
similarity_df = pd.DataFrame(similarity_matrix , index=customer_features['CustomerID'] , columns=customer_features["CustomerID"])


# In[24]:


def get_top_similar_three(customers ,similarity_data):
    lookalikes = {}
    for customer_id in customers:
        similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:4]
        lookalikes[customer_id] = list(zip(similar_customers.index , similar_customers.values))
    return lookalikes


# In[26]:


customer_ids = customer_features['CustomerID'].head(20)


# In[34]:


lookalikes_df = get_top_similar_three(customer_ids , similarity_df)


# In[42]:


lookalikes_df = pd.DataFrame({"cust_id" : lookalikes.keys() , "lookalikes" : lookalikes.values()})

lookalikes_df.head()