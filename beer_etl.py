
# # Craft Beer Analysis - ETL
# _____________________________________
# -------------------------------------
# 
# In this analysis we will be looking at how craft beer varies by location and ranking. Imagine we are a brewery and we want to figure out what makes a quality beer.


import pandas as pd
import matplotlib as mpl
import numpy as np
from IPython.display import clear_output
from requests import get
from bs4 import BeautifulSoup as soup
import re
from timeit import default_timer as timer
import random as rnd




import sys
sys.version


# ### Read in primary data
# 
# Our primary data consists of a beer data set followed by a series of other data sets that can be matched to each beer such as the brewery, beer style, categorie, and geo location.



beer = pd.read_csv('beers.csv')
breweries = pd.read_csv('breweries.csv')
styles = pd.read_csv('styles.csv')
cats = pd.read_csv('categories.csv')
geo = pd.read_csv('breweries_geocode.csv')


# ### do general exploratory anlysis of primary data



print('beer.csv shape: ',beer.shape)
print('breweries.csv shapee: ', breweries.shape)
print('styles.csv shape: ', styles.shape)
print('categories.csv shape: ', cats.shape)
print('geo.csv shape: ', geo.shape)


# In the output below we can the general composition of the dataset. The data provides us with columns such as 'cat_id' and 'style_id' that helps us match to each categorie and style.



beer.keys()
beer.head()


# Where this data is a good starting point, when you think about what our business does and the problem we are interested in solving it becomes obvious the information we have isn't particularly helpful. Therefore, we should expand on it and make it cleaner and bigger.

# # Scrape Data
# 
# Beeradvocate is social media website where beer drinkers are provided with a platform to track the beers they have tried as well as give a review of the beer. Since we want to know what make a beer a favorite it is a good idea to look here since it provides us with what their users have rated different beers. 
# 
# To retrieve this information we will:
# 1. we will be getting all the 'beeradvocate.com' links related to the breweries we are interested in by search engine scraping
# 2. use those links to get relevant data from beeradovcate 
# _________________________
# -------------------------

# *Remove punctuation and strip extra white space from brewery names*
# We do this since strings can be case sensitive and symbols such as '\' can complicate them. We will also remove any excess white space.


breweries["brewery_new"] = breweries['name'].str.replace('[^\w\s]','')
breweries['brewery_new'] = breweries['brewery_new'].map(str.strip)


# Create links to search engines using newly cleaned brewery names. We are using three major search engines incase a call limit is reached.
# 
# **Note**: *Timers will be used in loops to slow down gets, this is simply a precaution.*


brewery_link = breweries['brewery_new'].replace(' ', '+', regex = True).unique()#.replace('Co.', 'Company', regex = True)


brewery_link_rep = []
for i in range(brewery_link.shape[0]):
    brewery_link_rep.append(brewery_link[i].replace('Co.', 'Company')+'+beeradvocate')



links_google = []
for i in range(len(brewery_link_rep)):
    links_google.append('https://www.google.com/search?source=hp&ei=HBFaW57mFIjusQW0jaKwDg&q={0}&oq={1}&gs_l=psy-ab.3...14043.22551.0.22711.0.0.0.0.0.0.0.0..0.0....0...1.1.64.psy-ab..0.0.0....0.76ocgYyyiG4'.format(brewery_link_rep[i], brewery_link_rep[i]))



links_bing = []
for i in range(len(brewery_link_rep)):
    links_bing.append('https://www.bing.com/search?q={}&qs=n&form=QBLH&sp=-1&pq={}&sc=6-19&sk=&cvid=71F3CE631E12440793705F4E62BAEAB2'.format(brewery_link_rep[i], brewery_link_rep[i]))



links_yahoo = []
for i in range(len(brewery_link_rep)):
    links_yahoo.append('https://search.yahoo.com/search;_ylt=A0geKeVgnVhbbC4AhIZXNyoA;_ylc=X1MDMjc2NjY3OQRfcgMyBGZyA3lmcC10BGdwcmlkAzVTS2pUVlFFUXJ5VmhydlVKb1dVR0EEbl9yc2x0AzAEbl9zdWdnAzQEb3JpZ2luA3NlYXJjaC55YWhvby5jb20EcG9zAzAEcHFzdHIDBHBxc3RybAMwBHFzdHJsAzQxBHF1ZXJ5AzUxMiUyMGJyZXdpbmclMjBjb21wYW55JTIwYmVlciUyMGFkdm9jYXRlBHRfc3RtcAMxNTMyNTM0MTIx?p={}&fr2=sb-top&fr=yfp-t&fp=1'.format(brewery_link_rep[i]))


# Put new links into pandas data frame


links_df = pd.DataFrame(columns = ['Company', 'link_yahoo', 'link_google'])
links_df['Company'] = brewery_link_rep
links_df['link_yahoo'] = links_yahoo
links_df['link_google'] = links_google
links_df['link_bing'] = links_bing


links_df['Company'] = links_df['Company'].str.replace('+', ' ')
links_df['Company'] = links_df['Company'].str.replace('beeradvocate', '')


# ## Scraping
# 1. First get Beer Advocate links with the search engine links created above
# 2. Second use Beer Advocate links to get data for each brewery
# -------------------

# ### Get Beer Advocate links


len(links_df)-1


beer_links = []
for i in range(len(links_df)):
    ## Implement timer to resist reaching call limit
    rand_num = rnd.uniform(12, 16)
    start = timer()
    end = timer()
    while end-start < rand_num:
        end = timer()
    ## wrap get in try-except to catch companies with no link
    try:
        ## use get to retrieve the html of the search engine when a brewery is searched up
        source = get(links_df.iloc[i]['link_google'])
        ## Use regex to extract all beeradvocate links for that company;
        # follow up by taking the first occurence
        # we append a tuple to a list that matches each link to a brewery to help keep organized
        beer_links.append((links_df.iloc[i]['Company'], re.findall('https://www.beeradvocate.com/beer/profile/\d*/', str(source.content))[0]))
        clear_output()
        # print after each loop whether a get has succeded or failed so we can see where we are in the loop
        print(str(i)+'.','Success:', links_df.iloc[i]['Company'])
    except:
        ## if a company is found to not have a link append None and the company name to keep track of missing information
        beer_links.append((links_df.iloc[i]['Company'], None))
        clear_output()
        print(str(i)+'.', 'Failure:', links_df.iloc[i]['Company'])



beer_links_df = pd.DataFrame(beer_links)
beer_links_df.rename(columns = {0 : 'Brewery', 1 : 'link_beeradvocate'}, inplace = True)
## Optional: save to a csv incase you have to leave or change projects; this allows you to take a break
#beer_links_df.to_csv('~/desktop/beer_advocate_links.csv', index = False)
beer_links_df = pd.read_csv('~/desktop/beer_advocate_links1.csv')



beer_links_all =pd.read_csv("~/desktop/beer_advocate_links_all.csv").drop('Unnamed: 0', axis = 1 )


# ### Extract brewery data
# Now we will be looping over all of those links and appending the scraped df of each brewery to a final data set.


big = pd.DataFrame()

# this is our target value. The loop will update us where we are each itteration
beer_links_all.shape


for i in range(len(beer_links_all)):
    #rand_num = rnd.uniform(2, 8)
    #start = timer()
    #end = timer()
    #while end-start < rand_num:
    #    end = timer()
    try:
        ## Lastly we use the links we extracted from search engine previously to retrieve
        # the information we are interested in from beeradvocate. 
        # In this case we are retrieving the name, score, and abv of each beer from a brewey.
        r = get(beer_links_all['link_beeradvocate'][i])
        brewery = pd.read_html(r.text)[2]
        brewery['brewery'] = beer_links_all.iloc[i]['Brewery']
        big = big.append(brewery)
        clear_output()
        print(str(i)+'.','Success:', beer_links_all.iloc[i]['Brewery'])
    except:
        clear_output()
        print(str(i)+'.','Failure:', beer_links_all.iloc[i]['Brewery'])
        pass



#big.to_csv('~/desktop/scraped_beeradvocate_data.csv', index = False)




big.reset_index().drop(['index', 'Yours'], axis = 1, inplace = True)


merged_breweries= breweries[['id','website', 'country', 'state', 'city', 'brewery_new']].merge(geo, left_on = 'id', right_on = 'brewery_id')


beer_final = big.merge(merged_breweries, left_on ='brewery', right_on = 'brewery_new', how = 'inner').drop(['id_x', 'id_y', 'brewery_id', 'Yours'], axis = 1)


#beer_final.to_csv('~/desktop/final_beer_data.csv', index = False)

