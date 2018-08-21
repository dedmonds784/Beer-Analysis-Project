
# Beer Exploratory Analysis

In this analysis we will be doing a basic overview of the data we previously manipulated in the beer_etl.csv file. 
The main question we will be looking to answer is exactly what we can find while looking at this data set.

First lets start off by loading the packages and reading in the data.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
%matplotlib inline
import squarify
import seaborn as sns
from scipy import stats

```


```python
import sys
sys.version
```




    '3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24) \n[Clang 6.0 (clang-600.0.57)]'




```python
beer = pd.read_csv('~/documents/data/beer/data/final_beer_data.csv')
beer[['ABV', 'Score']] = beer.replace('?', 0)[['ABV', 'Score']].astype(np.float64)
print(beer.keys())
print(beer.shape)
```

    Index(['Name', 'Style', 'ABV', 'Ratings', 'Score', 'brewery', 'website',
           'country', 'state', 'city', 'brewery_new', 'latitude', 'longitude',
           'accuracy', 'location'],
          dtype='object')
    (45672, 15)


As we can see in the above output we have a 45672 individual beers with 15 features. Now that we see what we have lets look at the distribution of the data.


```python
styles = pd.DataFrame(beer['Style'].value_counts()).reset_index()
```


```python
squarify.plot(sizes = styles.iloc[:15]['Style'], label = styles.iloc[:15]['index'], alpha = .3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11112ac50>




![png](output_8_1.png)



```python
cities = beer['city'].value_counts().reset_index()
squarify.plot(sizes = cities.iloc[:20]['city'], label = cities.iloc[:20]['index'], alpha = .3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11120dcf8>




![png](output_9_1.png)



```python
sns.regplot(pd.to_numeric(beer['Ratings'].replace('?', None)), beer['Score'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1119bb588>




![png](output_10_1.png)



```python
pd.to_numeric(beer['Score'].replace('?', None)).plot.hist(bins = 50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x110e2a0f0>




![png](output_11_1.png)



```python
sns.pairplot(beer, diag_kind = 'kde')
mpl.title('Pair Plot')
```




    Text(0.5,1,'Pair Plot')




![png](output_12_1.png)



```python
print('Alcohol Percentage Stats: ',stats.describe(beer['ABV']))
print()
print('Ratings Stats: ',stats.describe(beer['Ratings']))
print()
print('Score Stats: ',stats.describe(beer['Score']))
```

    Alcohol Percentage Stats:  DescribeResult(nobs=45672, minmax=(0.0, 32.0), mean=5.8632063408653, variance=8.701994956612374, skewness=0.18321063672095847, kurtosis=1.9962484469889468)
    
    Ratings Stats:  DescribeResult(nobs=45672, minmax=(0, 17091), mean=139.27071290944124, variance=409633.76901985036, skewness=10.951896669798415, kurtosis=169.186905056022)
    
    Score Stats:  DescribeResult(nobs=45672, minmax=(0.0, 5.0), mean=3.5365162462778073, variance=0.7568928379046936, skewness=-2.9466060380397097, kurtosis=9.278109156484867)

