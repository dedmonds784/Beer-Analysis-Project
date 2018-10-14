
# coding: utf-8

# # Beer Modeling

# In this part of the analysis we will be using our exploratory analysis we did in the last part to model. This model will help us see what features in the data set have the biggest effect on making a prediciton. We will be using a gradient boosting regressor to predict the score of a beer.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import sklearn.preprocessing as prp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error


# In[2]:


beer = pd.read_csv('~/documents/data/beer/data/final_beer_data.csv')
beer[['ABV', 'Score']] = beer.replace('?', 0)[['ABV', 'Score']].astype(np.float64)
print(beer.keys())


# In the cell above we can see the features we will be using to predict. The independent variables will be the beer style, alcohol percentage, number ratings, brewery name, state, city, latitude, and longitude. The dependent variable will be the beer score.

# In[3]:


beer.head()


# If we look at the output above we can see that the values are all integers. We will need to preprocess this data so that it will work in the model. The columns we will have to encode are 'Style', 'brewery', 'state', 'city', 'brewery_new'.
colnames = ['Style', 'brewery', 'state', 'city', 'brewery_new']
for col in colnames: 
    mask = ~beer[col].isnull()
    beer[col][mask] = prp.LabelEncoder().fit_transform(beer[col][mask])
    beer[col].fillna(0, inplace = True)
# In[4]:


y = beer['Score']
x = pd.get_dummies(beer[['ABV', 'Style', 'Ratings', 'brewery', 'state', 'city', 'latitude', 'longitude']])


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)

X_train_scale = MinMaxScaler().fit_transform(X_train)
X_test_scale = MinMaxScaler().fit_transform(X_test)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scale, y_train, random_state = 0 )


# With our training and testing data set up lets run the model and return a score to see how accurate the model can be.

# In[9]:


clf = GradientBoostingRegressor()
clf.fit(X_train_scale, y_train)
clf.score(X_test_scale, y_test).round(3)


# With the default parameters we get an accuracy of about 79% lets use a grid search to optimize the model

# In[10]:


learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
num_estimators = [80, 100, 120, 150, 170]
max_depths = [2, 3, 4, 5, 6, 7, 8]


# In[ ]:


best_score = 0
best_abs = 1
for rate in learning_rates:
    for num in num_estimators:
        for depth in max_depths:
            clf = GradientBoostingRegressor(learning_rate = rate, n_estimators = num, max_depth = depth, random_state = 0)
            clf.fit(X_train_scale, y_train)
            score = clf.score(X_train_scale, y_train)
            score_val = clf.score(X_val, y_val)
            score_abs = abs(score - score_val)
            if (score_val > best_score) and (score_abs < best_abs):
                best_score = score_val
                best_abs = score_abs
                best_rate = rate
                best_num = num
                best_depth = depth
                  
            print('learning_rate: ' + str(rate))
            print('num_estimators: ' + str(num))
            print('max_depth: ' + str(depth))
            print('score(training): ' + str(score))
            print('score(validation): ' + str(score_val))
            print()


# In the cell above we can see that we get a model that predicts with 99% accuracy. After checking for over and under fitting though we can see that the most consistent model was one that had 97% accuracy as you can see below.

# In[ ]:


clf = GradientBoostingRegressor(learning_rate = best_rate, n_estimators = best_num, max_depth = best_depth, random_state = 0)
clf.fit(X_train_scale, y_train)
score = clf.score(X_train_scale, y_train)
score_val = clf.score(X_val, y_val)


# In[ ]:


print('best_model: ', 'learning_rate = ',best_rate ,';','n_estimators = ', best_num,';', 'max_depth = ', best_depth )
print('best score: ', best_score)
print('least under/over fitting model: ', best_abs)


# In[ ]:


mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


# When we see what features had the greatest effect on the model it appears that the number of ratings a beer has, as well as the alcohol percentage and style, are the greatest predictors for a score.

# In[ ]:


feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance/feature_importance.max())


# In[ ]:


relative_importance_df = pd.DataFrame(list(zip(X_train.keys(), feature_importance)), columns = ['Feature_name', 'Relative_importance'])
relative_importance_df.sort_values('Relative_importance', ascending = False, inplace = True)


# In[ ]:


relative_importance_df.set_index(relative_importance_df['Feature_name']).plot.barh()
mpl.title('Variable Importance')

