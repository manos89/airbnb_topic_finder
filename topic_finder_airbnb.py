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
import queue
import threading
from queue import Empty

USERNAME='manos'
PASSWORD='11Ian19891989'
HOST='d0002332'
PORT='27017'
MONGO_DATABASE='large_papers'
client2 = pymongo.MongoClient('mongodb://'+USERNAME+':'+PASSWORD+'@'+HOST+':'+PORT+'/'+MONGO_DATABASE)
db2 = client2['airbnb']
collection="detailed_results"

def get_info(queue):
	global count
	queue_full = True
	while queue_full:
		try:
			d= queue.get(False)
			try:
				sim = gensim.matutils.cossim(d['distance']['i'], d['distance']['j'])
				distance=1-sim
				distance=float("%.6f" % distance)
				dict_to_db={'id':d['id'],'distance':[{'id':d['distance']['id'],'distance':distance}]}
				if d['counter']>0:
					while True:
						try:
							push=db2['lda_distance'].update_one({'id':dict_to_db['id']},{'$push':{'distance':dict_to_db['distance'][0]}})
							break
						except:
							pass
				else:
					insert=db2['lda_distance'].insert_one(dict_to_db)
			except Exception as E:
				print(str(E))
			q.task_done()
		except Empty:
			queue_full = False



results=db2[collection].aggregate([{"$group":{"_id": "$id","last_scraped": { "$first": "$last_scraped" },"clean_text":{"$first":"$clean_text"} }}],allowDiskUse=True)
results=list(results)
# client = pymongo.MongoClient('mongodb://localhost:27017/')
# db = client['airbnb']

tokenizer = RegexpTokenizer(r'\w+')

# town='london'
counter=0
# results=db2[collection].aggregate([{"$match":{"Town":"london"}},{"$group":{"_id": "$id","last_scraped": { "$first": "$last_scraped" },"clean_text":{"$first":"$clean_text"} }}],allowDiskUse=True)
text_list=[]
for r in results:

	tokens = tokenizer.tokenize(r['clean_text'])
	text_list.append(tokens)
	counter+=1
	if counter%1000==0:
		pass


dictionary = corpora.Dictionary(text_list)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]


lda = models.LdaModel(doc_term_matrix, id2word=dictionary, num_topics=8, per_word_topics=False,minimum_probability=0)
# print(lda.show_topics())
print(lda.num_topics)
corpus_lda=lda[doc_term_matrix]

q = queue.Queue()
count=0
dict_list=[]
for i in range(15000,len(corpus_lda)):
	print(i)
	counter=0
	corpus_list=[]
	for j in range(0,len(corpus_lda[i])):
		corpus_list.append({'topic':corpus_lda[i][j][0],'lda':float(corpus_lda[i][j][1])})
	dict_write={'id':results[i]['_id'],'corpus':corpus_list}
	insert=db2['corpus'].insert_one(dict_write)
	# for j in range(i-1,0,-1):
	# 	dict_write={}
	# 	# dict_write={'id':results[i]['_id'],'counter':counter,'distance':{'id':results[j]['_id'],'i':corpus_lda[i],'j':corpus_lda[j]}}
	# 	dict_write['id']=results[i]['_id']
	# 	dict_write['counter']=counter
	# 	dict_write['distance']={'id':results[j]['_id'],'i':corpus_lda[i],'j':corpus_lda[j]}
	# 	# list_write=[results[i]['_id'],counter,results[j]['_id'],corpus_lda[i],corpus_lda[j]]
	# 	# dict_write.update({'id':results[i]['_id'],'counter':counter,'distance':{'id':results[j]['_id'],'i':corpus_lda[i],'j':corpus_lda[j]}})
	# 	q.put(dict_write)
	# 	counter+=1
		# sim = gensim.matutils.cossim(corpus_lda[i], corpus_lda[j])
	# 	distance=1-sim
	# 	# distance="%.4f" % distance
	# 	distance_dict={'column':j,'distance':distance}
	# 	dict_write['distance'].append(distance_dict)
	# db2['lda_distance'].insert(dict_write)
quit()
print('entering multithreading')
thread_count = 192


for i in range(thread_count):
	t = threading.Thread(target=get_info, args = (q,))
	t.start()

q.join()