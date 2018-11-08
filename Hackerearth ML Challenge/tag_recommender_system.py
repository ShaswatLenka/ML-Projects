import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import string
import os
import nltk #NLP library
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to perform stemming
#from bs4 import BeautifulSoup #uncomment if you like to use BeautifulSoup instead
from sklearn.feature_extraction.text import TfidfVectorizer


#define chunk size for big dataset
chunk_size = 50000

#reading in CSV's from file path
dataset= pd.read_csv('./new_dataset/train.csv', chunksize = chunk_size, iterator = True )

for chunk in dataset:
   df = chunk
   break


#stemmer object
#NOTE: I think stemming in this case is not important as we only deal with the technical terms so the words/tags
#themselves are the root words
ps = PorterStemmer()

#Another way of doing it but I found it not accurate enough
#soup = BeautifulSoup(df['article'][0], 'html.parser')
#article = soup.get_text()

#creating the corpus for all the articles
corpus = []

for i in range(0,1000):
    article = re.sub('[^a-zA-Z]', ' ', df['article'][i])
    article = article.lower()
    article = article.split()
    article = [ps.stem(word) for word in article if not word in set(stopwords.words('english')) and len(word) != 1] 
    article = ' '.join(article)
    corpus.append(article)
#Create the bag of words model
