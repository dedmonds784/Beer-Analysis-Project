
# coding: utf-8

# # Beer Exploratory Analysis

# In this analysis we will be doing a basic overview of the data we previously manipulated in the beer_etl.csv file. 
# The main question we will be looking to answer is exactly what we can find while looking at this data set.

# First lets start off by loading the packages and reading in the data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import squarify
import seaborn as sns
from scipy import stats


# In[2]:


import sys
sys.version


# In[3]:


beer = pd.read_csv('~/documents/data/beer/data/final_beer_data.csv')
beer[['ABV', 'Score']] = beer.replace('?', 0)[['ABV', 'Score']].astype(np.float64)
print(beer.keys())
print(beer.shape)


# As we can see in the above output we have a 45672 individual beers with 15 features. Now that we see what we have lets look at the distribution of the data.

# In[4]:


styles = pd.DataFrame(beer['Style'].value_counts()).reset_index()


# In[5]:


squarify.plot(sizes = styles.iloc[:15]['Style'], label = styles.iloc[:15]['index'], alpha = .3)


# In[6]:


cities = beer['city'].value_counts().reset_index()
squarify.plot(sizes = cities.iloc[:20]['city'], label = cities.iloc[:20]['index'], alpha = .3)


# In[7]:


sns.regplot(pd.to_numeric(beer['Ratings'].replace('?', None)), beer['Score'])


# In[8]:


pd.to_numeric(beer['Score'].replace('?', None)).plot.hist(bins = 50)


# In[9]:


sns.pairplot(beer, diag_kind = 'kde')
mpl.title('Pair Plot')


# In[10]:


print('Alcohol Percentage Stats: ',stats.describe(beer['ABV']))
print()
print('Ratings Stats: ',stats.describe(beer['Ratings']))
print()
print('Score Stats: ',stats.describe(beer['Score']))

