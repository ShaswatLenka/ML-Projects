import pandas as pd 
import re
import nltk
import csv
import time
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from bs4 import BeautifulSoup #uncomment if you like to use BeautifulSoup instead
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer #to compute the IDF

start_time = time.time()

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
def Preprocessing(df):
    for i in range(0,chunk_size):
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

#Using the TF.IDF technique to extract the keywords from the corpus
#create a vocabulary of words
#ignore words that appear in more than 85% of the document
def TfIdf(x):
    cv = CountVectorizer(max_df = 0.85, max_features = 1000)    
    word_count_vector = cv.fit_transform(corpus)

    #to see the words in the vocabulary use: list(cv.vocabulary_.keys())[:10]

    #calculate the IDF
    #WARNING: ALWAYS USE IDF ON A LARGE CORPUS
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)


    # do this once, this is a mapping of index 
    feature_names=cv.get_feature_names()

    # get the document that we want to extract keywords from
    doc=corpus[x]
     
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    return [feature_names, tf_idf_vector,doc]

#define chunk size for big dataset
chunk_size = 1000

#reading in CSV's from file path
dataset = pd.read_csv('./new_dataset/train.csv', chunksize = chunk_size, iterator = True )
with open('submission.csv','a',newline='') as csvFile:
    writer = csv.writer(csvFile)
    for chunk in dataset:
        df = chunk
        Preprocessing(df)
        for x in range(chunk_size):
            
            #perform TF.IDF
            tfidf = TfIdf(x)
            
            #sort the tf-idf vectors by descending order of scores
            sorted_items=sort_coo(tfidf[1].tocoo())
     
            #extract only the top n; n here is 5
            keywords=extract_topn_from_vector(tfidf[0],sorted_items,5)
            top_tags = ""
            for y in keywords.keys():
                top_tags = top_tags + (y + "|")                                
            row = [df['id'][x],top_tags]
            writer.writerow(row)
        csvFile.close()
        break
print(' time taken = %s '%(time.time() - start_time))
 
    
    
    

        
        
        
    