#!/usr/bin/env python
# coding: utf-8

# # Tomas Costantino - A00042881
# 
# # Analysis on e-commerce logistics with EDA and SVM to predict target variable
# 
# dataset extracted from: https://www.kaggle.com/prachi13/customer-analytics

# In[1]:


import pandas as pd


# In[2]:


#Read dataset
df = pd.read_csv('Train.csv',header=0)
df.head()


# In[3]:


#Drop the column ID out the dataset.
#This column only contains numerical increasing entries that are not required for model building
df.drop('ID',axis=1,inplace=True)
df.head()


# In[4]:


#Rename the columns to make it short and easy to keep coding with 
df.rename({'Warehouse_block':'block',
           'Mode_of_Shipment':'shipment',
           'Customer_care_calls':'care_calls',
           'Customer_rating':'rating',
           'Cost_of_the_Product':'cost',
           'Prior_purchases':'prior_purchases',
           'Product_importance':'importance',
           'Gender':'gender',
           'Discount_offered':'discount',
           'Weight_in_gms':'weight',
           'Reached.on.Time_Y.N':'on_time'},
          axis='columns',
          inplace=True)
df.head()


# # EDA and Descriptive Statistics
# 
# import **matplotlib** to do the necessary plots 

# Use **describe** function to know about the dataset

# In[5]:


#this function provides a quick description of the dataset
df.describe()


# In[6]:


df.isnull().any()


# **PLOTS FOR ON TIME**

# In[7]:


#import libraries to do EDA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#holds only the column on_time
count_df = df['on_time']
#holds the values of the counting for the pie graph
pie_values = []

#Append the counting values 
pie_values.append(count_df[df['on_time'] == 1].count())
pie_values.append(count_df[df['on_time'] == 0].count())

#Print with two decimals
plt.pie(pie_values,autopct="%.2f%%")
plt.legend(['On time','Not on time'])
plt.tight_layout()
plt.show()


# In[8]:


sns.catplot('on_time',data=df,kind='count')


# **Categorical plots for qualitative variables (categorical)** 
# 
# Counting 

# In[9]:


sns.catplot('block', data=df,  kind='count')


# In[10]:


sns.catplot('shipment',data=df,kind='count')


# In[11]:


sns.catplot('rating',data=df,kind='count')


# In[12]:


sns.catplot('importance',data=df,kind='count')


# In[13]:


sns.catplot('gender',data=df,kind='count')


# Now let's see how of these variables are related to **on_time**

# In[14]:


sns.catplot('block',data=df,hue='on_time',kind='count')


# In[15]:


sns.catplot('shipment',data=df,hue='on_time',kind='count')


# In[16]:


sns.catplot('rating',data=df,hue='on_time',kind='count')


# In[17]:


sns.catplot('importance',data=df,hue='on_time',kind='count')


# In[18]:


sns.catplot('gender',data=df,hue='on_time',kind='count')


# **Histograms of quantitative variables**

# In[19]:



#Create a new DataFrame with only quantitaitve data to plot the histograms
#Qualitative plots are below
df_quantitative = df.drop(['block','shipment','rating','importance','gender','on_time'],axis=1).copy()

fig = df_quantitative.hist(figsize=(20,20))
[x.title.set_size(32) for x in fig.ravel()]

plt.tight_layout()
plt.show()


# In[20]:


df['cost'].mean()


# **Density plots for quantitative variables**

# In[21]:


df_density = df.drop(['block','shipment','importance','gender','rating'],axis=1).copy()
# show density plot
# create a subplot of 2 x 3
plt.subplots(2,3,figsize=(20,20))

# Plot a density plot for each variable
for idx, col in enumerate(df_density.columns):
    ax = plt.subplot(3,3,idx+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df_density.on_time == 0][col], hist=False, axlabel= False, kde_kws={'linestyle':'-', 'color':'black', 'label':'Not on Time'})
    sns.distplot(df.loc[df_density.on_time == 1][col], hist=False, axlabel= False, kde_kws={'linestyle':'--', 'color':'black', 'label':'On time'})
    ax.set_title(col)
    ax.legend()
    
# Hide the 7th, 8th and 9th subplot (bottom right) since there are only 6 plots
# This is done because apparently it does not allow a 2x3 matrix to plot only 6
plt.subplot(2,3,6).set_visible(False)
plt.tight_layout()
plt.show()


# # Discount and weight graphs show that they are predictors of whether a product is on time or not, so let's plot some whisker boxes

# In[22]:


sns.boxplot(x='on_time',y='weight',data=df)


# In[23]:


sns.boxplot(x='on_time',y='weight',hue='importance',data=df)


# In[24]:


sns.boxplot(x='on_time',y='discount',data=df)


# In[25]:


sns.boxplot(x='on_time',y='discount',hue='care_calls',data=df)


# In[26]:


sns.boxplot(x='on_time',y='discount',hue='prior_purchases',data=df)


# # Data Preprocessing and Data Modelling

# In[27]:


#Check for missing values on the data
print(df.isnull().any())


# Looks like there is **no missing value**, are we sure?
# Let's see how the dataset looks

# In[28]:


df.describe()


# In[29]:


#Now check type of variables to pass to the model as some of them may need to be casted or encoded
df.dtypes


# In[30]:


#Encode 'gender' as it only contains two categories
df['gender'].replace({'M':1,'F':0},inplace=True)


# In[32]:


#Now encode the rest of the categorical variables with pandas get_dummies
df_encoded = pd.get_dummies(df,columns=['shipment',
                                      'importance',
                                      'block',
                                      'rating'])
df_encoded.head()


# In[38]:


#import library and create a new dataframe

from sklearn import preprocessing
X_scaled = preprocessing.scale(df_encoded.drop('on_time',axis=1).copy())


# In[31]:


#X contains the features to train the model
X = df.drop('on_time',axis=1).copy()
X.head()


# In[32]:


#One-hot encoding of categorical variables
X_encoded = pd.get_dummies(X,columns=['shipment',
                                      'importance',
                                      'block',
                                      'rating'])
X_encoded.head()


# In[46]:


#variable we want to predict
y = df['on_time'].copy()
y.head(10)


# # DATA STANDARDISATION AND SPLIT
# Standardization of variables as a preprocessing step is a requirement for many machine learning algorithms. Another positive effect of data standardization is that it shrinks the magnitude of the variables, transforming them to a scale that is more proportional.
# 

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y ,random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)


# In[39]:


#Create the model and train it
from sklearn.svm import SVC
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)


# In[40]:


#Plot confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=['On time','Not on time'])


# In[ ]:


#Library for cross-validation
from sklearn.model_selection import GridSearchCV


# In[45]:


#Search for the best parameters for the model
param_grid = [
    {'C':[1,5,10,15,20,25,30], #The values for C must be >0
     'gamma':[1,0.1,0.05,0.01,0.005,0.0025,0.001],
     'kernel':['rbf']},
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=7,
    scoring='accuracy',
    verbose=0
)
optimal_params.fit(X_train_scaled,y_train)
print(optimal_params.best_params_)


# In[47]:


#create a new model with the parameters obtained from the search
clf_svm = SVC(random_state=42,C=1,gamma=0.001)
clf_svm.fit(X_train_scaled,y_train)


# In[48]:


#plot new confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=['On time','Not on time'])


# In[ ]:




