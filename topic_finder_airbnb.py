from langdetect import detect_langs
import pymongo
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
import gensim
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import time

USERNAME='manos'
PASSWORD='11Ian19891989'
HOST='d0002332'
PORT='27017'
MONGO_DATABASE='large_papers'

client2 = pymongo.MongoClient('mongodb://'+USERNAME+':'+PASSWORD+'@'+HOST+':'+PORT+'/'+MONGO_DATABASE)
db2 = client2['airbnb']
results=db2[collection].aggregate([{"$group":{"_id": "$id","last_scraped": { "$first": "$last_scraped" },"clean_text":{"$first":"$clean_text"} }}],allowDiskUse=True)

# client = pymongo.MongoClient('mongodb://localhost:27017/')
# db = client['airbnb']
collection="detailed_results"
tokenizer = RegexpTokenizer(r'\w+')

# town='london'
counter=0
# results=db2[collection].aggregate([{"$match":{"Town":"london"}},{"$group":{"_id": "$id","last_scraped": { "$first": "$last_scraped" },"clean_text":{"$first":"$clean_text"} }}],allowDiskUse=True)
text_list=[]
for r in results:
	tokens = tokenizer.tokenize(r['clean_text'])
	text_list.append(tokens)
	counter+=1
	if counter%100==0:
		pass

print(counter)
time.sleep(160)
quit()
dictionary = corpora.Dictionary(text_list)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]


lda = models.LdaModel(doc_term_matrix, id2word=dictionary, num_topics=8, per_word_topics=False,minimum_probability=0)
# print(lda.show_topics())
print(lda.num_topics)
corpus_lda=lda[doc_term_matrix]
count=0
for i in range(0,len(corpus_lda)):
	dict_write={'row':i,'distance':[]}
	for j in range(i-1,0,-1):
		sim = gensim.matutils.cossim(corpus_lda[i], corpus_lda[j])
		distance=1-sim
		# distance="%.4f" % distance
		distance_dict={'column':j,'distance':distance}
		dict_write['distance'].append(distance_dict)
	db2['lda_distance'].insert(dict_write)

