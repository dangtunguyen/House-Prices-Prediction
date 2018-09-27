
# coding: utf-8

# In[1]:

###
#CS235 Predict house price    Shasha Li & Tu Nguyen
#This code has refered several Scripts from Kaggle Kernel.
#References
#1. XGBoost: https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
#2. Ridge, Lasso& Random Forest: https://www.kaggle.com/tedpetrou/house-prices-advanced-regression-techniques/dummies-for-dummies
#3. Feature engineering: https://www.kaggle.com/yadavsarthak/house-prices-advanced-regression-techniques/you-got-this-feature-engineering-and-lasso
#4. Feature selection: https://www.kaggle.com/mshih2/house-prices-advanced-regression-techniques/using-xgboost-for-feature-selection
###


# In[2]:

import numpy as np # array calculation
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import skew, skewtest
# Input data files are available in the "../input/" directory.
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:

#Read in data
#Make sure we put the Id column in the index of the pandas data frame. It is of no value for model building


# In[4]:

train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)


# In[5]:

#Drop some outlinears


# In[6]:

o=[30,462,523,632,968,970, 1298, 1324]
train_df=train_df.drop(o,axis=0)


# In[7]:

#Combine Test and Train into a Single DataFrame
#Next, we will combine train and test into a single pandas DataFrame so that we can apply all our transformations to both the train and test set in one step and that both the train and the test set will have identical column names. 
#We must first store the SalePrice variable as our y and drop it from train_df. 
#Evaluation will be done on the log of SalePrice, we go ahead and transform the final home price. We need to remember to transform it back in order to submit.


# In[8]:

y_train = np.log(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)


# In[9]:

#One thing that we notice is that the first variable MSSubClass is actually a categorical variable that is recorded as numeric. Lets change this variable to a string.


# In[10]:

all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)


# In[11]:

#ADD two new features


# In[12]:

#feat_trial = (all_df['1stFlrSF'] + all_df['2ndFlrSF']).copy()
#print("Skewness of the original intended feature:",skew(feat_trial))
#print("Skewness of transformed feature", skew(np.log1p(feat_trial)))

# hence, we'll use the transformed feature
# lets create the feature then
all_df['1stFlr_2ndFlr_Sf'] = np.log1p(all_df['1stFlrSF'] + all_df['2ndFlrSF'])


# In[13]:

#feat_trial = (all_df['1stFlr_2ndFlr_Sf'] + all_df['LowQualFinSF'] + all_df['GrLivArea']).copy()
#print("Skewness of the original intended feature:",skew(feat_trial))
#print("Skewness of transformed feature", skew(np.log1p(feat_trial)))
all_df['All_Liv_SF'] = np.log1p(all_df['1stFlr_2ndFlr_Sf'] + all_df['LowQualFinSF'] + all_df['GrLivArea'])


# In[14]:

#transform log for some numerical features


# In[15]:

numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
#print(skewed_feats)
all_df[skewed_feats] = np.log1p(all_df[skewed_feats])


# In[16]:

# Indicator (dummy) Variables
#Since sklearn doesn't natively handle categorical predictor variables we must 0/1 encode them as a different column for each unique category for each variable. As you can see above, MSSubClass has about a dozen unique values. Each of these values will turn into a column.


# In[17]:

# Using pd.get_dummies on an entire DataFrame.
#get_dummies can work on an entire dataframe. This is very nice and works by ignoring all numeric features and making 0/1 columns for all categorical features (those that have 'object') as its data type.


# In[18]:

all_dummy_df = pd.get_dummies(all_df)
#all_dummy_df.head()


# In[19]:

#all_df['MSSubClass'].dtypes


# In[20]:

# Missing Values
#Let's check the missing values first to see how many we have to deal with.


# In[21]:

#all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)


# In[22]:

# Replacing missing values
# There are many valid ways to replace missing values. Many times, a missing value might mean a very specific thing. Here we will bypass further introspection and simply replace each missing value with the mean. This could potentially be a very stupid thing to do, but to proceed to model building we will just go ahead with the mean.


# In[23]:

mean_cols = all_dummy_df.mean()
#mean_cols.head(10)


# In[24]:

all_dummy_df = all_dummy_df.fillna(mean_cols)


# In[25]:

# Check that missing values are no more


# In[26]:

#all_dummy_df.isnull().sum().sum()


# In[27]:

# More Data Prep
#Since we will be using penalized (ridge) regression - Ridge its best practice to standardize our inputs by subtracting the mean and dividing by the standard deviation so that they are all scaled similarly. We only want to do this to our non 0/1 variables. In pandas these were our original numeric variables. We will apply this standardization to both train and test data at the same time, which is technically data snooping but will proceed again for simplicity.


# In[28]:

numeric_cols = all_df.columns[all_df.dtypes != 'object']
#numeric_cols


# In[29]:

numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


# ### Check a histogram of a variable to see that the scaling worked
# Checking the variable **GrLivArea** we see that the scaling has centered it to 0. We also see some outliers here > 3 standard deviations. We could apply a log transformation (something you can think about) or investigate those large values.

# In[30]:

#all_dummy_df['GrLivArea'].hist();


# In[31]:

# Model Building
# Next we will explore a few popular models and use cross validation to get an estimate for what we are likely to see as our score for a submission.


# In[32]:

# Splitting Data back to Train/test
#At the beginning of the notebook we combined all the train and test data. We will no separate it back out.


# In[33]:

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]


# In[34]:

#dummy_train_df.shape, dummy_test_df.shape


# In[35]:

### Ridge Regression with Cross Validation


# In[36]:

from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score


# In[37]:

# Not completely necessary, just converts dataframe to numpy array
X_train = dummy_train_df.values
X_test = dummy_test_df.values


# In[38]:

# Using cross_val_score 
#Sklearn has a nice function that computes the cross validation score for any chosen model. The code below loops through an array of alphas (the penalty term for ridge regression) and outputs the results of 10-fold cross validation. Sklearn uses the negative mean squared error as its scoring method so we must take the negative and the square root to get the same metric that kaggle is using.


# In[39]:

alphas = np.logspace(-3, 2, 50)
#print(alphas)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    #print(clf)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='mean_squared_error'))
    test_scores.append(np.mean(test_score))
#print(test_scores)


# In[40]:

#test_scores=np.array(test_scores)
#print(test_scores=test_scores.min())
val, idx = min((val, idx) for (idx, val) in enumerate(test_scores))
#print(alphas[idx])
ridge = Ridge(alpha=alphas[idx])
ridge.fit(X_train, y_train)
y_ridge = np.exp(ridge.predict(X_test))


# In[41]:

###
#Using Lasso
#from sklearn.linear_model import LassoCV
#alphas = np.logspace(-5, 0, 50)
#print(alphas)
#test_scores = []
#for alpha in alphas:
#    clf = LassoCV(alpha, selection='random', max_iter=25000)
#    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='mean_squared_error'))
#    test_scores.append(np.mean(test_score))
#    #print(test_scores)
#print(test_scores)

#val, idx = min((val, idx) for (idx, val) in enumerate(test_scores))
#print(alphas[idx])
#model_lasso = LassoCV(alphas[idx], selection='random', max_iter=25000).fit(X_train, y_train)
#y_final = np.expm1(model_lasso.predict(X_test))
###


# In[42]:

###
# Using random forest
#Lets do the same procedure for fitting a random forest to the data. The training is done on 200 trees for 5-fold cv. This will take quite some time.


# In[43]:

#from sklearn.ensemble import RandomForestRegressor


# In[44]:

#max_features = [.1, .3, .5, .7, .9, .99]
#max_features=[.3]
#test_scores = []
#for max_feat in max_features:
#    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='mean_squared_error'))
#    test_scores.append(np.mean(test_score))
#    #print(test_scores)
#print(test_scores)


# In[45]:

#plt.plot(max_features, test_scores)
#plt.title("Max Features vs CV Error");
#rf = RandomForestRegressor(n_estimators=500, max_features=.3)
#rf.fit(X_train, y_train)
#y_rf = np.exp(rf.predict(X_test))
###


# In[46]:

#Using xboost


# In[47]:

import xgboost as xgb


# In[48]:

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test)
params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[49]:

#print(model)
#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[50]:

model_xgb = xgb.XGBRegressor(n_estimators=450, max_depth=4, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y_train)


# In[51]:

y_xgb = np.expm1(model_xgb.predict(X_test))


# In[52]:

#Make a prediction based on the combination of Ridge and XGBoost


# In[53]:

y_final = 0.8*y_ridge+0.2*y_xgb


# In[54]:

submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})


# In[55]:

#submission_df.head(10)


# In[56]:

# Make Submisison


# In[57]:

submission_df.to_csv('HousePrice.csv', index=False)


# In[ ]:



