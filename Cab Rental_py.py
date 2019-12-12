#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from scipy import stats
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Setting working directory
os.chdir(r"E:\edWisor\Project2\Cab Rental\train_cab")
os.getcwd()


# In[3]:


#loading data
cab_train = pd.read_csv('train_cab.csv')
cab_test = pd.read_csv('test.csv')


# ## Understanding the data and Exploratory Data Analysis
# To know the basic understanding of the dataset such as shape, data types, uniques values, missing value analysis, to understand the basic statistics of each variable and all the pre-processing techniques.

# In[4]:


#Shape of the data
print(cab_train.shape) #(16067,7)
print(cab_test.shape) #(9914,6)


# In[5]:


#First five rows of our train data
cab_train.head()


# In[6]:


#First five rows of our test data
cab_test.head()


# In[7]:


#Number of Unique values in train data
cab_train.nunique()


# In[8]:


#To find the missing values in our dataset
cab_train.isna().sum()


# In[9]:


#To know the data types in train dataset
cab_train.dtypes

#Fare_amount should be float
#pickup_datetime should be a date type
#passenger_count should be an integer type


# In[10]:


#To know the data types in test dataset
cab_test.dtypes

#pickup_datetime should be a date type


# A few observations from the datasets
#    * `pickup_datetime` should be converted to date type using pandas
#    * `passenger_count` should be an int type and any data point less than 1 and greater than 6 can be removed/imputed
#    * `fare_amount` should be a float type and any data point less than 0 can be removed/imputed
#    * `pickup_latitude` and `dropoff_latitude` should have values in between -90 to +90 degrees and data point beyond these     values can be removed
#    * `pickup_longitude` and `dropoff_longitude` should have values in between -180 to +180 degrees and data point beyond these     values can be removed
#    * By using the co-ordinates of latitude and longitude, we can find the distance between pickup and drop locations
#    * After the above steps, we'll try to drop a few variables and data types are to properly converted

# In[11]:


#Convert the data types
cab_train['fare_amount'] = pd.to_numeric(cab_train['fare_amount'] , errors = 'coerce')
#By using errors parameter with corece value, we can replace non-numeric values with NaN values


# To convert the `pickup_datetime` to datetime format and to separate year,month and date etc.
# While trying to convert `pickup_datetime` it was found that value at index# 1327 is 43, which is to be dropped.

# In[12]:


np.where(cab_train['pickup_datetime'] == '43')
cab_train.iloc[1327,:]
cab_train = cab_train.drop(cab_train.index[1327])


# In[13]:


#To convert the pickup_datetime to datetime format and separating year,month and date etc.
cab_train['pickup_datetime'] = pd.to_datetime(cab_train['pickup_datetime'], format = "%Y-%m-%d %H:%M:%S UTC")


# In[14]:


#To check the data types
cab_train.dtypes


# In[15]:


#Creating new features such as year, month, date etc. based on the timestamp 
cab_train['year'] = cab_train['pickup_datetime'].dt.year
cab_train['Month'] = cab_train['pickup_datetime'].dt.month
cab_train['Date'] = cab_train['pickup_datetime'].dt.day
cab_train['Day'] = cab_train['pickup_datetime'].dt.dayofweek
cab_train['Hour'] = cab_train['pickup_datetime'].dt.hour


# In[16]:


#To check top 5 rows of the data
cab_train.head()


# In[17]:


#To convert the pickup_datetime for test data to datetime format and separating year,month and date etc.
cab_test['pickup_datetime'] = pd.to_datetime(cab_test['pickup_datetime'], format = "%Y-%m-%d %H:%M:%S UTC")


# In[18]:


#Creating new features such as year, month and date etc based on datetime for test data
cab_test['year'] = cab_test['pickup_datetime'].dt.year
cab_test['Month'] = cab_test['pickup_datetime'].dt.month
cab_test['Date'] = cab_test['pickup_datetime'].dt.day
cab_test['Day'] = cab_test['pickup_datetime'].dt.dayofweek
cab_test['Hour'] = cab_test['pickup_datetime'].dt.hour


# * As of now `pickup_datetime` in `cab_train` dataset is cleaned and now let's check with the `passenger_count`
# * Any data point with values < 1 and > 6 in `passenger_count` are to be removed

# In[19]:


#Let's remove the values in passenger_count variable with the values < 1 and > 6
cab_train = cab_train.drop(cab_train[cab_train['passenger_count'] < 1].index , axis = 0)
cab_train = cab_train.drop(cab_train[cab_train['passenger_count'] > 6].index , axis = 0)


# In[20]:


#To check if any missing values in passenger_count and delete them if they are less in number(55 we found)
cab_train['passenger_count'].isnull().sum()
#To remove missing values or null values from passenger_count variable
cab_train = cab_train.drop(cab_train[cab_train['passenger_count'].isnull()].index , axis = 0)
cab_train['passenger_count'].isnull().sum()


# In[21]:


#Let's remove the values in passenger_count variable with the values < 1 and > 6 in test data also and found no null values
cab_test = cab_test.drop(cab_test[cab_test['passenger_count'] < 1].index , axis = 0)
cab_test = cab_test.drop(cab_test[cab_test['passenger_count'] > 6].index , axis = 0)


# In[22]:


#Let's check for the fair_amount variable and any negative values should be removed/imputed
cab_train = cab_train.drop(cab_train[cab_train['fare_amount'] < 0].index, axis = 0)


# In[23]:


#We found that we have 24 missing values in fare_amount variable. As this is a less number, we can remove them
cab_train.isnull().sum()
cab_train = cab_train.drop(cab_train[cab_train['fare_amount'].isnull()].index, axis = 0)


# By using the the four variables, `pickup_longitude`, `pickup_latitude`,	`dropoff_longitude`, `dropoff_latitude` let's try to find the distance travelled. The usual procedure is to find with Haversine's formula, but let us try with different methods.

# In[24]:


# We have found one value in pickup_latitude > 90, (401...) so let's drop that observation

cab_train = cab_train.drop(cab_train[cab_train['pickup_latitude'] > 90].index, axis = 0)


# In[25]:


#To find the distance travelled using latitudes and longitudes
from geopy.distance import geodesic

def distance_conversion(x):
    origin_lat = x[0]
    origin_long = x[1]
    dest_lat = x[2]
    dest_long = x[3]
    origin = [origin_lat,origin_long]
    dest = [dest_lat,dest_long]
    distance = geodesic(origin, dest).kilometers
    return distance

#distance_conversion(40.721319,-73.844311,40.712278,-73.841610)
#distance_conversion(40.711303,-74.016048,40.782004,-73.979268)

#Let's create a variable "distance_travelled" and try to find it's values using the above function in both the datasets

cab_train['distance_travelled'] = cab_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(distance_conversion,axis=1)
cab_test['distance_travelled'] = cab_test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(distance_conversion,axis=1)


# In[26]:


#To check a few observations after removing the variables
cab_train.head()
cab_test.head()


# In[27]:


#To check the variable distance_travelled whose values are zero and found that 454 observations have value as 0
#This will list out the indexes with distance_travelled = 0
#cab_train[cab_train['distance_travelled'] == 0].index

#Let's replace zeros with NAN values and try to impute them
cab_train.loc[cab_train['distance_travelled'] == 0,'distance_travelled'] = np.nan
cab_test.loc[cab_test['distance_travelled'] == 0,'distance_travelled'] = np.nan
cab_train.isna().sum().sum()


# Let us drop a few variables for which we don't need them as the necessary information from those variables has been extracted

# In[28]:


#Dropping the 5 variables as mentioned above
var_to_drop = ['pickup_datetime','pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude']
cab_train = cab_train.drop(var_to_drop, axis = 1)
cab_test = cab_test.drop(var_to_drop, axis = 1)


# In[29]:


#Let's convert a few variables to the required data types in both train and test data
cab_train['passenger_count'] = cab_train['passenger_count'].astype('int64')
cab_train['Month'] = cab_train['Month'].astype('category')
cab_train['Day'] = cab_train['Day'].astype('category')

#Convert in test data
cab_test['Month'] = cab_test['Month'].astype('category')
cab_test['Day'] = cab_test['Day'].astype('category')


# In[30]:


#Imputing the missing values
#Actual value = 5.037
#Mean Value = 15.12
#Median Value = 2.196
#KNN value = 4.790
#cab_data['distance_travelled'].iloc[52] = np.nan
#cab_data['distance_travelled'].iloc[52] = cab_data['distance_travelled'].mean()
#cab_data['distance_travelled'].iloc[52] = cab_data['distance_travelled'].median()

#Imputing with KNN method
cab_train = pd.DataFrame(KNN(k = 3).fit_transform(cab_train), columns = cab_train.columns)
cab_test = pd.DataFrame(KNN(k = 3).fit_transform(cab_test), columns = cab_test.columns)


# In[31]:


#To verify if we have any null values after KNN imputation
print(cab_train.isnull().sum().sum())
print(cab_train.shape)


# In[32]:


cab_train.dtypes

#Dividing categorical and continuous variables

cont_var = ['passenger_count','year','Date','Hour','fare_amount','distance_travelled']
cat_var = ['Month','Day']

#To take a copy of cab_train dataset
cab_data = cab_train.copy()


# Now our `cab_train` dataset is a cleaned one with no missing values. Let's take a copy of `cab_train` as `cab_data` with dimensions (15905,8) and let's work on it for further steps.

# ## Univariate and Bivariate Analysis
# * Let us visualize the distribution of variables in our train dataset
# * We will use histograms for continuous variables and barplots for the categorical variables

# In[33]:


#For continuous variables
for i in cont_var[:4]:
    plt.hist(cab_train[i].dropna(),bins = 'auto')
    plt.title("Distribution of " + str(i))
    plt.show()


# In[34]:


#Check the distribution of the Categorical variables
sns.set_style("whitegrid")
sns.factorplot(data=cab_data, x='Month', kind= 'count',size=4,aspect=2)
sns.factorplot(data=cab_data, x='Day', kind= 'count',size=4,aspect=2)


# From the above plots, we can have a few quick insights.
# * Demand of cabs is high on 6th and 5th days in a week and least on 1st day of the week
# * Demand of cabs is high in the month of May, March and June respectively and least demand in August
# * Cabs are high in demand during evening hours and least demand during early hours of the day
# * Single travelled passenger's prefer cabs than with a group of 4/5    

# In[35]:


# Grouping the data using Day against our target variable and plotting bar plot
cab_data.groupby('Day').mean()['fare_amount'].plot.bar()
plt.ylabel('Fare Amount')


# In[36]:


# Grouping the data using Month against our target variable and plotting bar plot
cab_data.groupby('Month').mean()['fare_amount'].plot.bar()
plt.ylabel('Fare Amount')


# In[37]:


# Grouping the data using Hour against our target variable and plotting bar plot
cab_data.groupby('Hour').mean()['fare_amount'].plot.bar()
plt.ylabel('Fare Amount')


# In[38]:


# Grouping the data using Date against our target variable and plotting bar plot
cab_data.groupby('Date').mean()['fare_amount'].plot.bar()
plt.ylabel('Fare Amount')


# From the above plots, we can have a few quick insights.
# * The average fare amount is higher on the 5th day of the week
# * The average fare amount is higher in the month of February
# * Fair amounts are higher between 6 P.M.- 7 P.M. and least at5 A.M.
# * Average fare price is highest on 27th of every month
# * The average fare amount is higher at 5 P.M.

# ## Feature Scaling
# Let's scale our variables distance_travelled and passenger_count in both train and test datasets

# In[39]:


#The data is right skewed in distance_travelled and hence we'll apply log on that variable
cab_data['distance_travelled'] = np.log1p(cab_data['distance_travelled'])
#To apply normalisation on passenger count
norm_var = ['passenger_count']

for i in norm_var:
    cab_data[i] = (cab_data[i] - cab_data[i].min()) / (cab_data[i].max() - cab_data[i].min())#Normalization formula

#To apply normalisation on passenger count for Test dataset
cab_test['distance_travelled'] = np.log1p(cab_test['distance_travelled'])
for i in norm_var:
    cab_test[i] = (cab_test[i] - cab_test[i].min()) / (cab_test[i].max() - cab_test[i].min())#Normalization formula


# In[40]:


#Distribution of distance_travelled
sns.distplot(cab_data['distance_travelled'],bins='auto',color='green')
plt.title("Distribution for Distance Travelled")
plt.ylabel("Density")
plt.show()


# In[41]:


#To check the distribution of distance_travelled in the test data
sns.distplot(cab_test['distance_travelled'],bins='auto',color='blue')
plt.title("Distribution for Distance Travelled")
plt.ylabel("Density")
plt.show()


# In[42]:


#To know the shapes of the train and test data
print(cab_data.shape)
print(cab_test.shape)


# ## Outlier Analysis
# Outliers are to be detected by using box plot and those values are to be replaced or imputed by using various techniques such as mean, median or KNN method.

# In[43]:


#Box plot for Fare Amount
plt.boxplot(cab_data['fare_amount'])
plt.xlabel("Fare Amount")
plt.ylabel("Values")
plt.title("Box Plot for Fare Amount")
plt.show()


# In[44]:


#Box plot for Fare Amount
plt.boxplot(cab_data['distance_travelled'])
plt.xlabel("Distance Travelled")
plt.ylabel("Values")
plt.title("Box Plot for Distance travelled")
plt.show()


# In[45]:


#Box plot for Fare Amount
plt.boxplot(cab_data['passenger_count'])
plt.xlabel("Passenger Count")
plt.ylabel("Values")
plt.title("Box Plot for Passenger Count")
plt.show()


# It is found that we have a few outliers in passenger_count, fare_amount and distance_travelled

# In[46]:


#List with variables with outliers
outliers = ['passenger_count', 'distance_travelled', 'fare_amount']

#Loop through the above list of variables
for i in outliers:
    q75,q25 = np.percentile(cab_data[i], [75,25]) #To get 75 and 25 percentile values
    iqr = q75 - q25 #Interquartile region
    #Calculating outerfence and innerfence
    outer = q75 + (iqr*1.5)
    inner = q25 - (iqr*1.5)
# Replacing all the outliers value to NA
cab_data.loc[cab_data[i]< inner,i] = np.nan
cab_data.loc[cab_data[i]> outer,i] = np.nan


# In[47]:


# Imputing missing values with KNN
cab_data = pd.DataFrame(KNN(k = 3).fit_transform(cab_data), columns = cab_data.columns)
# Checking if there is any missing value
cab_data.isnull().sum().sum()


# In[48]:


#To ensure if the outliers are removed
cab_data.describe()
cab_data.shape


# In[49]:


#Relationship between distance_travelled and fare_amount 
plt.figure(figsize=(15,7))
plt.scatter(x = cab_data['distance_travelled'],y = cab_data['fare_amount'],c = "g")
plt.xlabel('Distance_travelled')
plt.ylabel('Fare_amount')
plt.show()


# Now `cab_data` is free from missing values, outliers with shape as (15905, 8). Let's proceed for Feature Selection, Feature Scaling and developing ML alogorithms on our training dataset.

# ## Feature Selection
# To check for the multicollinearity for continuous variables by plotting correlation plot and remove the
# variables with r > 0.8

# In[50]:


#Correlation analysis for continuous variables
#Let's store all the numeric data into an object
numeric_data = cab_data.loc[:,cont_var]
#Set the measurements of the plot, let's say width = 10 and height = 10
a , k = plt.subplots(figsize=(10,10))
#Correlation matrix
corr_matrix = numeric_data.corr()
#Plotting a correlation graph
ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(10, 220, n=200),
square=True, annot = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')


# In[51]:


#Let's create dummy variables for categorical variables
#Get dummy variables for categorical variables
cab_data = pd.get_dummies(cab_data, columns = cat_var)
cab_test = pd.get_dummies(cab_test, columns = cat_var)
print(cab_data.shape)
print(cab_test.shape)


# ## Model development
# We've performed all the Preprocessing techniques for our data. Our next step is to divide the data into train
# and test, build a model upon the train data and evaluate on the test data. Then finally choose one ML model to validate on our actual test data and predict the values of `fare_amount`

# In[52]:


#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(cab_data.iloc[:,cab_data.columns != 'fare_amount'],
cab_data.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[53]:


#Building the model using linear regression
#Importing the necessary libraries for Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#Build a model on our training dataset
lr_model = LinearRegression().fit(X_train,y_train)
#Predict for the test cases
lr_predictions = lr_model.predict(X_test)
#To create a dataframe for both actual and predicted values
cabdata_lrmodel = pd.DataFrame({"Actual" : y_test, "Predicted" : lr_predictions})

#Function to find RMSE
def RMSE(x,y):
    rmse = np.sqrt(mean_squared_error(x,y))
    return rmse

#Function to find MAPE
def MAPE(true,predict):
    mape = np.mean(np.abs((true - predict) / true)) * 100
    return mape

#Calculate RMSE, MAPE and R-Squared value for this model

print("Root Mean Squared error :- " + str(RMSE(y_test,lr_predictions)))
print("R-Squared value :- " + str(r2_score(y_test,lr_predictions)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test,lr_predictions)))


# In[54]:


#Building the model using Decisison Tree
#Importing necessary libraries for Decision tree
from sklearn.tree import DecisionTreeRegressor
#Build Decision tree model on the train data
dt_model = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)
#Predict for the test cases
dt_predict = dt_model.predict(X_test)
#Create a dataframe for actual and predicted values
df_dtmodel = pd.DataFrame({"Actual" : y_test, "Predicted" : dt_predict})
#Calculate RMSE, MAPE and R_squared value for this model

print("RMSE: " + str(RMSE(y_test,dt_predict)))
print("R_Square score: " + str(r2_score(y_test,dt_predict)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test,dt_predict)))


# In[55]:


#Building the model using Randomforest
#Import library for RandomForest
from sklearn.ensemble import RandomForestRegressor
#Build random forest using RandomForestRegressor
ranfor_model = RandomForestRegressor(n_estimators = 300, random_state = 1).fit(X_train,y_train)
#Perdict for test cases
rf_predictions = ranfor_model.predict(X_test)
#Create data frame for actual and predicted values
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_predictions})
#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test, rf_predictions)))
print("R_square Score: "+str(r2_score(y_test, rf_predictions)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test,rf_predictions)))


# In[56]:


#Building the model using GradientBoosting
#Import necessary libraries for this ML algorithm
from sklearn.ensemble import GradientBoostingRegressor
#Build GB model on the train data
gb_model = GradientBoostingRegressor().fit(X_train,y_train)
#Predict the test cases
gb_predict = gb_model.predict(X_test)
#Create a dataframe for actual and predicted values
df_gbmodel = pd.DataFrame({"Actual" : y_test, "Predicted" : gb_predict})
#Calculate RMSE and R_squared values
print("RMSE: " + str(RMSE(y_test,dt_predict)))
print("R_Square score: " + str(r2_score(y_test,gb_predict)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test,gb_predict)))


# ## Dimension Reduction using Pricipal Component Analysis
# Principal Component Analysis (PCA) is a dimension-reduction tool that can be used to reduce a large set of
# variables to a small set that still contains most of the information in the large set.

# In[57]:


#Get the target variable
target_var = cab_data['fare_amount']
#Get the shape of our cleaned dataset
cab_data.shape #(15451, 25)
#Importing the library for PCA
from sklearn.decomposition import PCA
#Dropping the target variable
cab_data.drop(['fare_amount'], inplace = True, axis =1)
#To check the shape of the data after dropping the target variable
cab_data.shape# (15451, 25)


# In[58]:


#Converting our data to numpy array
numpy_data = cab_data.values
#Our data without target variable has 133 variables, so number of components = 24
pca = PCA(n_components = 24)
pca.fit(numpy_data)
#To check the variance that each PC explains
var = pca.explained_variance_ratio_
#Cumulative variance
var_cum = np.cumsum(np.round(var, decimals = 4) * 100)
plt.plot(var_cum)
plt.show()


# From the above graph, it is clear that approximately after 7 components, there is no variance even if all the rest
# of the components are considered. So let's select these 7 components as it explains almost 95 percent of data
# variance.

# In[59]:


#Selecting the 7 components
pca = PCA(n_components = 7)
#To fit the selected components to the data
pca.fit(numpy_data)
#Splitting into train and test data using train_test_split
X_train1,X_test1,y_train1,y_test1 = train_test_split(numpy_data,target_var,test_size = 0.2)


# Now by using the above data let's develop the model on our train data

# In[60]:


#Building the model using linear regression
#Importing the necessary libraries for Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#Build a model on our training dataset
lr_model = LinearRegression().fit(X_train1,y_train1)
#Predict for the test cases
lr_predictions = lr_model.predict(X_test1)
#To create a dataframe for both actual and predicted values
cabdata_lrmodel = pd.DataFrame({"Actual" : y_test1, "Predicted" : lr_predictions})

#Function to find RMSE
def RMSE(x,y):
    rmse = np.sqrt(mean_squared_error(x,y))
    return rmse

#Function to find MAPE
def MAPE(true,predict):
    mape = np.mean(np.abs((true - predict) / true)) * 100
    return mape

#Calculate RMSE, MAPE and R-Squared value for this model

print("Root Mean Squared error :- " + str(RMSE(y_test1,lr_predictions)))
print("R-Squared score :- " + str(r2_score(y_test1,lr_predictions)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test1,lr_predictions)))


# In[61]:


#Building the model using Decisison Tree
#Importing necessary libraries for Decision tree
from sklearn.tree import DecisionTreeRegressor
#Build Decision tree model on the train data
dt_model = DecisionTreeRegressor(max_depth = 2).fit(X_train1,y_train1)
#Predict for the test cases
dt_predict = dt_model.predict(X_test1)
#Create a dataframe for actual and predicted values
df_dtmodel = pd.DataFrame({"Actual" : y_test1, "Predicted" : dt_predict})
#Calculate RMSE, MAPE and R_squared value for this model

print("RMSE: " + str(RMSE(y_test1,dt_predict)))
print("R_Square score: " + str(r2_score(y_test1,dt_predict)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test1,dt_predict)))


# In[62]:


#Building the model using Randomforest
#Import library for RandomForest
from sklearn.ensemble import RandomForestRegressor
#Build random forest using RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 300, random_state = 1).fit(X_train1,y_train1)
#Perdict for test cases
rf_predictions = rf_model.predict(X_test1)
#Create data frame for actual and predicted values
df_rf = pd.DataFrame({'Actual': y_test1, 'Predicted': rf_predictions})
#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test1, rf_predictions)))
print("R_square Score: "+str(r2_score(y_test1, rf_predictions)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test1,rf_predictions)))


# In[63]:


#Building the model using GradientBoosting
#Import necessary libraries for this ML algorithm
from sklearn.ensemble import GradientBoostingRegressor
#Build GB model on the train data
gb_model = GradientBoostingRegressor().fit(X_train1,y_train1)
#Predict the test cases
gb_predict = gb_model.predict(X_test1)
#Create a dataframe for actual and predicted values
df_gbmodel = pd.DataFrame({"Actual" : y_test1, "Predicted" : gb_predict})
#Calculate RMSE and R_squared values
print("RMSE: " + str(RMSE(y_test1,dt_predict)))
print("R_Square score: " + str(r2_score(y_test1,gb_predict)))
print("Mean Absolute Percentage Error :- " + str(MAPE(y_test1,gb_predict)))


# So we have finally decided that RandomForest has predicted the least RMSE indicating the best fit. So let's predict our cleaned test data using Randomforest Regressor.

# In[64]:


#Building the model using Randomforest
#Import library for RandomForest
from sklearn.ensemble import RandomForestRegressor
#Build random forest using RandomForestRegressor
rf_predictions_test = rf_model.predict(cab_test)
#Create a new variable to the test dataset
cab_test['Predicted_Fare'] = rf_predictions_test


# In[65]:


cab_test.head()


# In[66]:


cab_test.isna().sum()


# In[67]:


cab_test.to_csv("Predicted_testdata.csv")

