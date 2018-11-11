#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk #NLP library
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to perform stemming
#from bs4 import BeautifulSoup #uncomment if you like to use BeautifulSoup instead
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer #to compute the IDF


#define chunk size for big dataset
chunk_size = 100

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

#Preprocessing
for i in range(0,100):
    article = re.sub('[^a-zA-Z]', ' ', df['article'][i])
    title = re.sub('[^a-zA-Z]', ' ', df['title'][i])
    article = article.lower()
    title = title.lower()
    article = article.split()
    title = title.split()
    article = [ps.stem(word) for word in article if not word in set(stopwords.words('english')) and len(word) != 1] 
    title = [ps.stem(word) for word in title if not word in set(stopwords.words('english')) and len(word) != 1] 
    article = ' '.join(article)
    title = ' '.join(title)
    paragraph = article + title
    corpus.append(paragraph)
    
#Using the TF.IDF technique to extract the keywords from the corpus
#create a vocabulary of words
#ignore words that appear in more than 85% of the document
cv = CountVectorizer(max_df = 0.85, max_features = 10000)    
word_count_vector = cv.fit_transform(corpus)

#to see the words in the vocabulary use: list(cv.vocabulary_.keys())[:10]

#calculate the IDF
#WARNING: ALWAYS USE IDF ON A LARGE CORPUS
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# do this once, this is a mapping of index 
feature_names=cv.get_feature_names()
 
# get the document that we want to extract keywords from
doc=corpus[2]
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#helper functions
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
 
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 5
keywords=extract_topn_from_vector(feature_names,sorted_items,5)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])

#write our results into a csv file
import csv
# store first five tags in a list
top_tags = ""
for x in keywords.keys():
    top_tags = top_tags + (x + "|")
    
#change 1 to the number of rows in the submission file
with open('submission.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['id','tags'])
    for x in range(1):
        row = [chunk['id'][x],top_tags]
        writer.writerow(row)
csvFile.close()

        
        
        
    