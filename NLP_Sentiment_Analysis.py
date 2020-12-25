import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import collections
import sys
print(sys.executable)
import time
import warnings
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from textblob import TextBlob
from textblob import Word
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from PIL import Image


#import reviews file here into load it into dataframe
yelp_reviews = pd.read_csv ( "yelp_reviews.csv " )

#Basic information about the data like number of rows , columns , type of data etc. 
print(yelp_reviews.head(n=5))
print(yelp_reviews.info())
print(yelp_reviews.describe())


#Creating a new column that is customer feedback from the star rating
Cust = []
for i in yelp_reviews['stars']:
    if (i == 1) | (i == 2):
        Cust.append('Negative')
    elif (i == 3) :
        Cust.append('Mixed')
    else:
        Cust.append('Positive')
        
yelp_reviews['Customer FEEDBACK'] = Cust
yelp_reviews['Customer FEEDBACK'].value_counts()

#A new column text length is created to store the number of characters in each review
yelp_reviews['Text length'] = yelp_reviews['text'].apply(lambda x:len(x.split()))

print(yelp_reviews.head(n=5))

'''
-----------------------------------------------------------------------------------------------------------

|
|                          EXPLORATORY DATA ANALYSIS FOR REVIEWS DATASET
|
-----------------------------------------------------------------------------------------------------------
'''
#Text lengths of the reviews made by customers
df = sns.FacetGrid(data = yelp_reviews, col = 'Customer FEEDBACK', hue = 'Customer FEEDBACK', palette='terrain', height=5)
df.map(sns.distplot, "Text length")
print(yelp_reviews.groupby('Customer FEEDBACK').mean()['Text length'])
#plt.show()

#joint plot for stars vs text length
yelp_reviews["date"]= pd.to_datetime(yelp_reviews["date"]).dt.date
yelp_reviews.set_index('date').head(1)

yelp_reviews["month"] = pd.to_datetime(yelp_reviews["date"]).dt.month
yelp_reviews["Year"] = pd.to_datetime(yelp_reviews["date"]).dt.year

yelp_reviews["length"] = yelp_reviews["text"].apply(len)
sns.jointplot(x=yelp_reviews["length"],
              y=yelp_reviews["stars"],
              data=yelp_reviews, kind='reg')
#plt.show()

#boxplot for stars vs textlength
plt.figure(figsize = (10,7))
sns.boxplot(x = 'stars', y = 'Text length', data = yelp_reviews)
#plt.show()

#countplot
plt.figure(figsize = (7,5))
sns.countplot('stars', data = yelp_reviews, palette="Accent")
#plt.show()

#countplot for customer feedback
plt.figure(figsize = (7,5))
sns.countplot('Customer FEEDBACK', data = yelp_reviews, palette="copper")
#plt.show()

yelp_reviews.groupby('Customer FEEDBACK').mean().corr()

#heatmap for correlation between cool,useful,funny,textlength and stars
plt.figure(figsize = (8,6))
sns.heatmap(yelp_reviews.groupby('Customer FEEDBACK').mean().corr(), cmap = "YlGnBu", annot=True)
#plt.show()



"""
-------------------------------------------------------------------------------------------------------------------
|                                                                                                                 |
|                                           SENTIMENT ANALYSIS                                                    |
|                                                                                                                 |
-------------------------------------------------------------------------------------------------------------------
"""

# Convert every word to lower case
yelp_reviews['review_lower'] = yelp_reviews['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(yelp_reviews)

# Removing the Punctuation
yelp_reviews['review_nopunc'] = yelp_reviews['review_lower'].str.replace('[^\w\s]', '')
print(yelp_reviews)

stop_words = stopwords.words('english')

# Removing of the Stopwords
yelp_reviews['cleaned_reviews'] = yelp_reviews['review_nopunc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
print(yelp_reviews)

# Lemmatize final review format
yelp_reviews['cleaned_reviews'] = yelp_reviews['cleaned_reviews']\
.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Calculate polarity
yelp_reviews['polarity'] = yelp_reviews['cleaned_reviews'].apply(lambda x: TextBlob(x).sentiment[0])
print(yelp_reviews)

# Calculate subjectivity
yelp_reviews['subjectivity'] = yelp_reviews['cleaned_reviews'].apply(lambda x: TextBlob(x).sentiment[1])
print(yelp_reviews)

count = 0
k = 0
for i in yelp_reviews['polarity']:
    if (i > 0 ) & (i < 1):
        count = count + 1
    else:
        k = k + 1

print("Overall Positive reviews based on the calculated polarity in the Sentiment analysis is ", count)
print("Overall Negative reviews based on the calculated polarity in the Sentiment Analysis is ", k)

count = 0
k = 0
for i in yelp_reviews['subjectivity']:
    if (i > 0.5 ):
        count = count + 1
    else:
        k = k + 1

print("The number reviews that tend to be more close to the factual informations base on the Sentiment Analysis are ", count)
print("The number reviews that does not tend to be close to the factual informations base on the Sentiment Analysis are ", k)


'''
-----------------------------------------------------------------------------------------
|                                                                                       |
|                        CLASSIFICATION ALGORITHM FOR OUR PREDICTION                    |
|                                                                                       |
------------------------------------------------------------------------------------------

FEATURE CREATION

'''

#first split the dataset into training and test dataset
x = yelp_reviews['text']
y = yelp_reviews['Customer FEEDBACK']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)

# examine the object shapes
print(x_train.shape)
print(x_test.shape)

'''
TEXT ANALYSIS
'''

#Text cleaning process for remvoing punctuations and stopwords from the data
from nltk.corpus import stopwords
def text_clean(message):
    nopunc = [i for i in message if i not in string.punctuation]
    nn = "".join(nopunc)
    nn = nn.lower().split()
    nostop = [words for words in nn if words not in stopwords.words('english')]
    return(nostop)

Positive = yelp_reviews[yelp_reviews['Customer FEEDBACK'] == 'Positive']
Negative = yelp_reviews[yelp_reviews['Customer FEEDBACK'] == 'Negative']
Mixed = yelp_reviews[yelp_reviews['Customer FEEDBACK'] == 'Mixed']

#cleaning the reviews for positive , negative and mixed by removing stopwords and punctuations
positive_bow = text_clean(Positive['text'])
negative_bow = text_clean(Negative['text'])
mixed_bow = text_clean(Mixed['text'])

positive_para = ' '.join(positive_bow)
negative_para = ' '.join(negative_bow)
mixed_para = ' '.join(mixed_bow)

'''
#Word cloud to display the most common words in the Reviews where customer experience was GOOD

stopwords = set(STOPWORDS)
stopwords.add('one')
stopwords.add('also')
mask_image = np.array(Image.open("thumb_up.png"))
wordcloud_good = WordCloud(colormap = "Paired",mask = mask_image, width = 300, height = 200, scale=2,max_words=1000, stopwords=stopwords).generate(positive_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_good, interpolation="bilinear", cmap = plt.cm.autumn)
plt.axis('off')
plt.figure(figsize = (10,6))
plt.show()
wordcloud_good.to_file("positive.png")



#Word cloud to display the most common words in the Reviews where customer experience was Negative
stopwords = set(STOPWORDS)
stopwords.add('one')
stopwords.add('also')
stopwords.add('good')
mask_image1 = np.array(Image.open("thumb_down.png"))
wordcloud_bad = WordCloud(colormap = 'tab10', mask = mask_image1, font_path = "C:\Windows\Fonts\Verdana.ttf", width = 1100, height = 700, scale=2,max_words=1000, stopwords=stopwords).generate(negative_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_bad,cmap = plt.cm.autumn)
plt.axis('off')
plt.show()
wordcloud_bad.to_file('negative.png')


#Word cloud to display the most common words in the Reviews where customer experience was mixed
stopwords = set(STOPWORDS)
wordcloud_mixed = WordCloud(colormap = "plasma",font_path = "C:\Windows\Fonts\Verdana.ttf", width = 1100, height = 700, scale=2,max_words=1000, stopwords=stopwords).generate(mixed_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_mixed,cmap = plt.cm.autumn)
plt.axis('off')
plt.show()
wordcloud_mixed.to_file('mixed.png')
'''

'''
-----------------------------------------------------------------------------------------------------------------------
|
|                                  MODEL BUILDING AND VALIDATION
|
-----------------------------------------------------------------------------------------------------------------------
'''

#Naive bayes classfier for classification
#from sklearn.feature_extraction.text import CountVectorizer

def vectorization():
    """
    Converts the text collection into a matrix of tokens
    and transforms the dataframe into a sparse matrix
    """
    cv_transformer = CountVectorizer(analyzer = text_clean)
    #print(len(cv_transformer.vocabulary_))

    x_trans = cv_transformer.fit_transform(x)
    print(x_trans)

    print('Shape of Sparse Matrix: \n', x_trans.shape)
    print('Amount of Non-Zero occurences:  \n', x_trans.nnz)

    sparsity = (100.0 * x_trans.nnz / (x_trans.shape[0] * x_trans.shape[1]))
    print('sparsity: {} \n'.format(sparsity))

    return x_trans

x = vectorization()

#Training Naive bayes model
#from sklearn.naive_bayes import MultinomialNB

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

def training_MNB():
    """
    Multinomial Naive Bayes is a specialised version of Naive Bayes designed more for text documents.
    Multinomial Naive Bayes model is built and fit it to our training set (x_train and y_train).
    """

    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    return nb

mnb = training_MNB()

def test_MNB():
     """
     Tests the MNB trained model and prints the stats
     """
     
     pred = mnb.predict(x_test) 
     accuracy_MNB = metrics.accuracy_score(y_test, pred)*100
     f1_MNB = f1_score(y_test, pred, average="weighted")*100

     print("Accuracy score from Naive Bayes Classifier \n", accuracy_MNB)
     print("f1-score from Naive Bayes Classifier \n", f1_MNB)

     print("Confusion Matrix after Naive Bayes\n", confusion_matrix(y_test, pred))
     print('\n')
     print("Classification report after Naive Bayes\n", classification_report(y_test, pred))
     return accuracy_MNB, f1_MNB

MNB_accuracy, MNB_f1_score = test_MNB()


#Fitting a Random forest classifier to predict the Customer Experience

   
rf = RandomForestClassifier(criterion='gini')
rf.fit(x_train, y_train)


def test_Random():
    """
    Tests the Random Forest model and prints the stats
    """

    #Predictions for Random forest Classification model
    pred = rf.predict(x_test)
    accuracy_ran = metrics.accuracy_score(y_test, pred)*100
    f1_ran = f1_score(y_test, pred, average="weighted")*100

    print("Accuracy Score after Random Forest classifier \n", accuracy_ran)
    print("f1-score after Random Forest classifier \n", f1_ran)

    print("Confusion Matrix after Random Forest \n", confusion_matrix(y_test, pred))
    print('\n')
    print("Classification report after Random Forest \n", classification_report(y_test, pred))
    return accuracy_ran, f1_ran

RAN_accuracy, RAN_f1_score = test_Random()


# K Nearest Neighbour classifer 

def training_KNN():
    """
    k-nearest neighbors model is built and fit it to our training set (x_train and y_train).
    """

    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(x_train, y_train)

    return neigh

knn = training_KNN()

def test_KNN():
    """
    Tests the KNN trained model and prints the stats
    """
    pred = knn.predict(x_test)
    accuracy_knn = metrics.accuracy_score(y_test, pred)*100
    f1_knn = f1_score(y_test, pred, average="weighted")*100

    print("Accuracy Score after KNN classifier \n", accuracy_knn)
    print("f1-score after KNN classifier \n", f1_knn)

    print("Confusion Matrix after KNN : \n", confusion_matrix(y_test, pred))
    print('\n')
    print("Classification Report after KNN \n ", classification_report(y_test, pred))
    return accuracy_knn, f1_knn

KNN_accuracy, KNN_f1_score = test_KNN()

def training_svm():
    """
    Support Vector Machine model is built and fit it to our training set (x_train and y_train).
    """

    s_v_m = svm.SVC()
    s_v_m.fit(x_train, y_train)

    return s_v_m

su_vm = training_svm()

def test_svm():
    """
    Tests the SVM trained model and prints the stats
    """
    pred = su_vm.predict(x_test)
    accuracy_svm = metrics.accuracy_score(y_test, pred)*100
    f1_svm = f1_score(y_test, pred, average="weighted")*100

    print("Accuracy Score after SVM : \n ", accuracy_svm)
    print("f1-score after SVM : \n", f1_svm)

    print("Confusion matrix after SVM : \n ", confusion_matrix(y_test, pred))
    print('\n')
    print("Classification report after SVM : \n", classification_report(y_test, pred))
    return accuracy_svm, f1_svm

svm_accuracy, svm_f1_score = test_svm()

def print_accuracy():
    print("The accuracy using Multinomial Naive Bayes classifier is: ", MNB_accuracy)
    print("The accuracy using Random Forest classifier is: ", RAN_accuracy)
    print("The accuracy using k-nearest neighbor classifier is: ", KNN_accuracy)
    print("The accuracy using support vector machine classifier is: ", svm_accuracy)

    if MNB_accuracy > KNN_accuracy and MNB_accuracy > svm_accuracy and MNB_accuracy > RAN_accuracy:
        print("\nThus, Sentiment analysis using Multinomial Naive Bayes classifier is more accurate.")
    elif KNN_accuracy > MNB_accuracy and KNN_accuracy > svm_accuracy and KNN_accuracy > RAN_accuracy:
        print("\nThus, Sentiment analysis using k-nearest neighbor classifier is more accurate.")
    elif RAN_accuracy > MNB_accuracy and RAN_accuracy > svm_accuracy and RAN_accuracy > KNN_accuracy:
        print("\nThus, Sentiment analysis using Random Forest classifier is more accurate.")
    else:
        print("\nThus, Sentiment analysis using support vector machine classifier is more accurate.")

print_accuracy()

###NULL Accuracy : No of correct classifications when we predict all record to be star rating 5####

print("Baseline Model \n", y_test.value_counts().head(1)/y_test.shape)

'''
#####Filter out a sample of false positives#####
print(x_test[y_test < pred].sample(10, random_state=6))

######Filter out a sample of false negatives#####
print(x_test[y_test < pred].sample(10, random_state=6))

'''








