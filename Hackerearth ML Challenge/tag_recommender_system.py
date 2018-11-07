import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import nltk #NLP library
nltk.download('stopwords')
from nltk.corpus import stopwords

#define chunk size for big dataset
chunk_size = 50000

#reading in CSV's from file path
dataset= pd.read_csv('./new_dataset/train.csv', chunksize = chunk_size, iterator = True )

for chunk in dataset:
   df = chunk
   break


article = re.sub('[^a-zA-Z]', ' ', df['article'][0])
article = article.lower()
article = article.split()
article = [word for word in article if not word in set(stopwords.words('english')) and len(word) != 1] 

