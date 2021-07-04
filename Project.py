#!/usr/bin/env python
# coding: utf-8

# # House Prices Prediction

# In[1]:


#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing data
data = pd.read_csv('Raw_Housing_Prices.csv')
data.head()


# In[3]:


#descriptives of Sale_Price
data['Sale Price'].describe()


# In[4]:


#distribution of target variable (Sale Price)
plt.figure(figsize = (13,6) , dpi = 70)
plt.subplot(1,2,1)
sns.distplot(data['Sale Price'])
plt.subplot(1,2,2)
data['Sale Price'].plot.hist()


# It can be observed target variable is left skewed. And is not uniformally Distributed

# # Treating Outliers

# We will be using Quantiles to find upper_limit and lower_limit and replace the outliers with limits.
# 

# In[5]:


# checking quantiles
q1 = data['Sale Price'].quantile(0.25)
q3 = data['Sale Price'].quantile(0.75)
q1, q3


# In[6]:


#calculating iqr(inter quartile range)
iqr = q3 - q1
iqr


# In[7]:


upper_limit = q3 + 1.5*iqr
lower_limit = q1 - 1.5*iqr
upper_limit, lower_limit


# In[8]:


# imputing outliers
def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value


# In[9]:


data['Sale Price'] = data['Sale Price'].apply(limit_imputer)


# In[10]:


#distribution of target variable (Sale Price) after treating outliers
plt.figure(figsize = (13,6) , dpi = 70)
plt.subplot(1,2,1)
sns.distplot(data['Sale Price'])
plt.subplot(1,2,2)
data['Sale Price'].plot.hist()


# Now ,Target Variable (Sale Price) is uniformly distributed.

# # Treating Missing Values

# In[11]:


#checking missing values
data.isnull().sum()


# We can clearly see there are many missing values and we have to treat them. We will be treating numerical variables and categorical variables separately.

# In[12]:


#we will be deleting all rows in which Sale Price is missing.
data.dropna(inplace=True , axis = 0 , subset=['Sale Price'])
data["Sale Price"].isnull().sum()


# In[13]:


data.info()


# In[14]:


#separating numerical variables
numerical_columns = ['No of Bathrooms', 'Flat Area (in Sqft)','Lot Area (in Sqft)',
                     'Area of the House from Basement (in Sqft)','Latitude',
                     'Longitude','Living Area after Renovation (in Sqft)']


# In[15]:


#using mean to fill the missing values we can also use median for the same.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])


# In[16]:


#data information
data.info()


# All numerical variables have been treated.

# Only zipcode is left as it is more of a categorical variable and has to be treat (mean value doesn't make sense for a pincode).

# In[17]:


data['Zipcode'].shape


# In[18]:


column = data["Zipcode"].values.reshape(-1,1)
column.shape


# In[19]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(column)


# In[20]:


data.info()


# # Other Transformations 

# In[21]:


data['No of Times Visited'].unique()


# In[22]:


# converting from string to categorical
mapping = {'None' : "0",
           'Once' : '1',
           'Twice' : '2',
           'Thrice' : '3',
           'Four' : '4'}

data['No of Times Visited'] = data['No of Times Visited'].map(mapping)


# In[23]:


data['No of Times Visited'].unique()


# In[24]:


# new variable creation that makes more sense
data['Ever Renovated'] = np.where(data['Renovated Year'] == 0, 'No', 'Yes')


# In[25]:


#manipulating datetime variable
data['Purchase Year'] = pd.DatetimeIndex(data['Date House was Sold']).year


# In[26]:


data['Years Since Renovation'] = np.where(data['Ever Renovated'] == 'Yes',
                                                     abs(data['Purchase Year'] - 
                                                        data['Renovated Year']), 0)


# In[27]:


data.head()


# In[28]:


# dropping redundant variables
data.drop( columns = ['Purchase Year', 'Date House was Sold', 'Renovated Year'], inplace = True)


# In[29]:


data.head(30)


# In[30]:


#removing ID column
data.drop( columns = 'ID', inplace = True)


# In[31]:


data.head()


# # Analyzing all categorical variables

# In[32]:


data.groupby('Condition of the House')['Sale Price'].mean().plot(kind = 'bar' , color = 'red')


# In[33]:


data.groupby('Condition of the House')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[34]:


data.groupby('Waterfront View')['Sale Price'].mean().sort_values().plot(kind = 'bar' , color = 'orange')


# In[35]:


data.groupby('Ever Renovated')['Sale Price'].mean().sort_values().plot(kind = 'bar' , color = 'green')


# In[36]:


data.groupby('Zipcode')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[37]:


data.dropna(inplace=True)
X = data.drop(columns=['Sale Price'])
Y = data['Sale Price']


# # Numerical Variable Analysis

# In[38]:


#checking distribution of independent numerical variables
def distribution(data ,var):
  plt.figure(figsize = (len(var)*6,6), dpi = 120)
  for j,i in enumerate(var):
    plt.subplot(1,len(var),j+1)
    plt.hist(data[i])
    plt.title(i)


# In[39]:


numerical_columns = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
       'Age of House (in Years)', 'Latitude', 'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[40]:


for i in numerical_columns:
  X[i] = pd.to_numeric(X[i])


# In[41]:


distribution(X, numerical_columns)


# In[42]:


#removing right skew
def right_skew(x):
  return np.log(abs(x+500))

right_skew_variables = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
        'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[43]:


for i in right_skew_variables:
  X[i] = X[i].map(right_skew)

# removing infinite values
X = X.replace(np.inf, np.nan)
X.dropna(inplace=True)


# In[44]:


distribution(X, numerical_columns)


# mapping categorical data to numerical data

# In[45]:


X["Waterfront View"] = X["Waterfront View"].map({ 'No':0,
                                                 'Yes':1})


X['Condition of the House'] = X['Condition of the House'].map({'Bad':1,
                                                                'Okay':2,
                                                                 'Fair':3,
                                                                 'Good':4,
                                                                 'Excellent':5})

X['Ever Renovated'] = X['Ever Renovated'].map({'No':0,
                                               'Yes':1 })

X.head()


# # Train Test Split

# In[46]:


Y = data['Sale Price']


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[48]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[49]:


predictions = lr.predict(x_test)
predictions


# In[50]:


#score of our model
lr.score(x_test, y_test)


# # Training the Polynomial Regression model on the whole dataset

# In[51]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)


# In[54]:


lin_reg_2.score(X_poly , Y)


# Score of Polynomial Linear Regression is better than Linear Regression

# # Residuals

# In[55]:


residuals = predictions - y_test

residual_table = pd.DataFrame({'residuals':residuals,
                    'predictions':predictions})
residual_table = residual_table.sort_values( by = 'predictions')


# In[56]:


z = [i for i in range(int(residual_table['predictions'].max()))]
k = [0 for i in range(int(residual_table['predictions'].max()))]


# In[57]:


plt.figure(dpi = 130, figsize = (17,7))

plt.scatter( residual_table['predictions'], residual_table['residuals'], color = 'red', s = 2)
plt.plot(z, k, color = 'green', linewidth = 3, label = 'regression line')
plt.ylim(-800000, 800000)
plt.xlabel('fitted points (ordered by predictions)')
plt.ylabel('residuals')
plt.title('residual plot')
plt.legend()
plt.show()


# # Distribution of errors

# In[58]:


plt.figure(dpi = 100, figsize = (10,7))
plt.hist(residual_table['residuals'], color = 'red', bins = 200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('distribution of residuals')
plt.show()


# # By Pratham Bist
# 

# In[ ]:




