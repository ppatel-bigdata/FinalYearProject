

"""
-------------------------------------------------------------------------------------------------------------------
|                                                                                                                 |
|                                   EXPLORATORY ANALYSIS FOR YELP BUSSINESS DATASET                               |
|                                                                                                                 |
-------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords



# importing Yelp Business dataset
yelp_business = pd.read_csv ( "yelp_business.csv " )
print ( yelp_business )
print ( yelp_business.head ( n=5 ) )
print ( yelp_business.describe () )

# importing Yelp reviews dataset
yelp_reviews = pd.read_csv( "yelp_reviews.csv " )
print(yelp_reviews )
print(yelp_reviews.head(n=5))
print(yelp_reviews.describe())

print(yelp_business.columns)
print(yelp_reviews.columns)

#reviewing the total number of rows and columns in the dataset

print(yelp_business.shape)
print(yelp_reviews.shape)

#Dealing with missing values in the category attribute

yelp_business['categories'].isna().mean()
yelp_business = yelp_business[yelp_business['categories'].notna()]
print(yelp_business.shape)

# Making an overall count of each and every kind of category
category_temp1 = ';'.join(yelp_business['categories'])
category_temp2 = re.split('[;|,]', category_temp1)
business_cat_modified = [item.lstrip() for item in category_temp2 ]
business_category = pd.DataFrame(business_cat_modified ,columns=['category'])
print(business_category)

bussiness_cat_count = business_category.category.value_counts()
bussiness_cat_count = bussiness_cat_count.sort_values(ascending = False)
bussiness_cat_count = bussiness_cat_count.iloc[0:10]

# plotting the graph
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(bussiness_cat_count.index, bussiness_cat_count.values, palette='Paired',edgecolor = 'white')
plt.title("Top Business Categories",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 60)
plt.ylabel('Number of Businesses', fontsize = 12)
plt.xlabel('Category of Business', fontsize = 12)

#text labels
r = ax.patches
labels = bussiness_cat_count.values
for rect, label in zip(r, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')
plt.show()

#Using Shopping Stores
yelp_bussiness_shp = yelp_business.loc[[i for i in yelp_business['categories'].index if re.search('Shopping', yelp_business['categories'][i])]]
#Using Rstaurants
yelp_bussiness_res = yelp_business.loc[[i for i in yelp_business['categories'].index if re.search('Restaurants', yelp_business['categories'][i])]]
#Top 10 cities with highest number of shopping stores:
yelp_bussiness_shp['city_state'] = yelp_bussiness_shp['city'] + ',' + yelp_bussiness_shp['state']
yelp_bussiness_res['city_state'] = yelp_bussiness_res['city'] + ',' + yelp_bussiness_res['state']

#For Shopping Businsses
city_shp_count = yelp_bussiness_shp.city_state.value_counts()
city_shp_count = city_shp_count.sort_values(ascending = False)
city_shp_count = city_shp_count.iloc[0:10]

#For Restaurants
city_res_count = yelp_bussiness_res.city_state.value_counts()
city_res_count = city_res_count.sort_values(ascending = False)
city_res_count = city_res_count.iloc[0:10]

# graph-1(Shopping)

sns.set_palette("pastel")
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(city_shp_count.index, city_shp_count.values,edgecolor = 'black')
plt.title("Top Cities with highest number of Shopping Businesses",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 60)
plt.ylabel('Number of Shopping Businesses', fontsize = 12)
plt.xlabel('City,State', fontsize = 12)

#text labels
r = ax.patches
labels = city_res_count.values
for rect, label in zip(r, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')

plt.show()

# graph-1(Restaurants)

sns.set_palette("pastel")
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(city_res_count.index, city_res_count.values,edgecolor = 'black')
plt.title("Top Cities with highest number of Restaurants",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 60)
plt.ylabel('Number of Restaurants', fontsize = 12)
plt.xlabel('City,State', fontsize = 12)

#text labels
r = ax.patches
labels = city_res_count.values
for rect, label in zip(r, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')

plt.show()

yelp_bus_shp = yelp_business.loc[[i for i in yelp_business['categories'].index if re.search('Shopping', yelp_business['categories'][i])]]

yelp_bus_shp.loc[yelp_bus_shp.name == 'Walmart', 'name'] = 'Walmart Supercenter'

res_count = yelp_bus_shp.name.value_counts()
res_count = res_count.sort_values(ascending = False)
res_count = res_count.iloc[0:15]

# plot

fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(res_count.index, res_count.values, palette = "plasma")
plt.title("Shopping Stores with High Occurences",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 60)
plt.ylabel('Number of Stores', fontsize = 12)
plt.xlabel('Store', fontsize = 12)

#text labels
r = ax.patches
labels = res_count.values
for rect, label in zip(r, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')


plt.show()

# Import Data
sub_wc = yelp_bussiness_shp.loc[(yelp_bus_shp.name == 'Walgreens') | (yelp_bus_shp.name == 'CVS Pharmacy')]
print(sub_wc)

# Plot
sns.set_style("white")
gridobj = sns.lmplot(x="stars", y="review_count", hue="name", data=sub_wc,
                     height=7, aspect=1.6, robust=True, palette='tab10',
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'), legend=False)
gridobj.set(xlim=(0.5, 5.5), ylim=(0, 60))
plt.title("Scatterplot with line of best fit grouped by Top Pharmacy Stores", fontsize=18, weight= 'bold')
plt.legend(title='Pharmacy Stores', loc='upper right', labels=['CVS Pharmacy', 'Walgreens'], fancybox=True, fontsize='10')
plt.show()

