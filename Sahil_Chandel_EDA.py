#!/usr/bin/env python
# coding: utf-8

# <h1 style='color:black;background:yellow;font-weigth:bolder;font-size:32px;text-align:center;'>E-commerce EDA</h1>

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


# <h3 style='color:black;background:yellow;font-weigth:bolder;font-size:24px;text-align:center;'>Getting basic Information of the shape and nature of the data</h3>

# In[5]:


customers.info()
customers.describe()


# In[6]:


products.info()
print()
products.describe()


# In[7]:


transactions.info()
print()
transactions.describe()


# In[8]:


customers.isnull().sum()


# In[9]:


transactions.isnull().sum()


# In[10]:


products.isnull().sum()


# In[11]:


customers[customers.duplicated()]


# In[12]:


transactions[transactions.duplicated()]


# In[13]:


products[products.duplicated()]


# <h1 style='color:black;background:yellow;font-weigth:bolder;font-size:24px;text-align:center;'>Converting Datetime to supported format</h1>

# In[14]:


customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers.head()


# In[15]:


transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate']);
transactions.head()


# ### Merging Transaction and Customers

# In[16]:


transaction_customers = pd.merge(transactions , customers , on='CustomerID',how='inner')
transaction_customers.info()


# In[17]:


transaction_customers.describe()


# In[18]:


transaction_customers.head()


# ### Merging Products With The Transaction & Customers

# In[19]:


product_bought = pd.merge(transaction_customers  , products , on="ProductID" , how="inner")
product_bought.info()


# In[20]:


product_bought.describe()


# In[21]:


product_bought.head()


# In[22]:


product_bought.isnull().sum()


# In[23]:


product_bought.to_csv("FinalData.csv",index=False)


# In[24]:


product_bought['Year_Month'] = pd.to_datetime(product_bought['TransactionDate']).dt.to_period('M')


# In[25]:


product_bought.head()


# In[26]:


revenue_trends = product_bought.groupby('Year_Month')['TotalValue'].sum().reset_index()


# In[27]:


revenue_trends['Year_Month'] = revenue_trends['Year_Month'].dt.to_timestamp()


# In[28]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=revenue_trends, x="Year_Month", y='TotalValue')
plt.title("Revenue Trends Over Time")
plt.xlabel("Year Month")
plt.ylabel("Total Value in USD")
plt.xticks(rotation=45)
plt.show()


# In[29]:


average_order_value = product_bought['TotalValue'].mean()
f'{average_order_value:.2f}'


# <h1 style='color:white;background:purple;font-weigth:bolder;font-size:32px;text-align:center;'>Business Insights</h1>

# <h1 style='color:black;background:#ceddbb;font-weigth:bolder;font-size:24px;text-align:center;'>Top Customer(Revenue)</h1>

# In[30]:


n = 10 
top_customers = product_bought.groupby('CustomerName')['TotalValue'].sum().sort_values(ascending=False).head(n)
top_customers


# <h1 style='color:black;background:#ceddbb;font-weigth:bolder;font-size:24px;text-align:center;'>Best Selling Products</h1>

# In[31]:


best_selling_products = product_bought.groupby('ProductName')['TotalValue'].sum().sort_values(ascending=False).head(10)
best_selling_products


# In[40]:


plt.figure(figsize=(10,10))
best_selling_products.plot(kind='pie' , autopct='%1.1f%%' , startangle=90 , colormap='coolwarm' , textprops={'fontsize':10} , wedgeprops={'edgecolor' : '#fff'})
plt.title('Revenue by Categories' , fontsize=16)
plt.show()


# <h1 style='color:black;background:#ceddbb;font-weigth:bolder;font-size:24px;text-align:center;'>Most Selling Categories</h1>

# In[33]:


categories = product_bought.Category.unique()
category_revenue = product_bought.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
category_revenue.head()


# In[34]:


plt.figure(figsize=(10,10))
category_revenue.plot(kind='pie' , autopct='%1.1f%%' , startangle=90 , colormap='coolwarm' , textprops={'fontsize':14} , wedgeprops={'edgecolor' : '#fff'})
plt.title('Revenue by Categories' , fontsize=16)
plt.show()


# <h1 style='color:black;background:#ceddbb;font-weigth:bolder;font-size:24px;text-align:center;'>Frequent Customers</h1>

# In[35]:


customer_frequency = product_bought.groupby('CustomerID')['TransactionDate'].nunique()
customer_frequency.describe()


# In[36]:


customer_frequency.head()


# In[37]:


plt.figure(figsize=(12,6))
sns.histplot(customer_frequency , kde =  True , bins=30 , color = 'red')
plt.title('Transactions by Customers Over Time')
plt.xlabel("Number of Transactions")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# <h1 style='color:black;background:#ceddbb;font-weigth:bolder;font-size:24px;text-align:center;'>Sales during Particular Time (Season)</h1>

# In[38]:


product_bought['Month'] = product_bought['TransactionDate'].dt.month
monthly_revenue = product_bought.groupby('Month')['TotalValue'].sum()


# In[39]:


plt.figure(figsize=(10,6))
sns.barplot(x=monthly_revenue.index , y = monthly_revenue.values , palette='husl')
plt.title('Monthly Revenue Trends')
plt.xlabel('Months')
plt.ylabel('Total Revenue (USD)')
plt.xticks(range(12) , ['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May'  , 'Jun' ,'Jul' , 'Aug' , 'Sept' , 'Oct' , 'Nov' , 'Dec'])
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




