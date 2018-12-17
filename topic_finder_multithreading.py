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


def get_info(queue):
	queue_full = True
	while queue_full:
		try:
			b= queue.get(False)
			try:
				below_corpus=list([(0,b['corpus'][0]['lda']),(1,b['corpus'][1]['lda']),(2,b['corpus'][2]['lda']),(3,b['corpus'][3]['lda']),
							(4,b['corpus'][4]['lda']),(5,b['corpus'][5]['lda']),(6,b['corpus'][6]['lda']),(7,b['corpus'][7]['lda'])])
				sim = gensim.matutils.cossim(root_corpus, below_corpus)
				distance=1-sim
				distance=float("%.6f" % distance)
				dict_to_db={'id':a['id'],'distance':[{'id':b['id'],'distance':distance}]}
				if b['counter']>0:
					while True:
						try:
							push=db2['lda_distance_multithreading'].update_one({'id':dict_to_db['id']},{'$push':{'distance':dict_to_db['distance'][0]}})
							break
						except:
							pass
				else:
					insert=db2['lda_distance_multithreading'].insert_one(dict_to_db)
			except Exception as E:
				print(str(E))
			q.task_done()
		except Empty:
			queue_full = False



USERNAME='manos'
PASSWORD='11Ian19891989'
HOST='d0002332'
PORT='27017'
MONGO_DATABASE='large_papers'
client2 = pymongo.MongoClient('mongodb://'+USERNAME+':'+PASSWORD+'@'+HOST+':'+PORT+'/'+MONGO_DATABASE)
db2 = client2['airbnb']
collection="corpus"

all_results=db2[collection].find({})
for a in all_results:
	print(a['id'])
	below_ids=db2[collection].find({'id':{'$lt':a['id']}})
	counter=0
	root_corpus=list([(0,a['corpus'][0]['lda']),(1,a['corpus'][1]['lda']),(2,a['corpus'][2]['lda']),(3,a['corpus'][3]['lda']),
				(4,a['corpus'][4]['lda']),(5,a['corpus'][5]['lda']),(6,a['corpus'][6]['lda']),(7,a['corpus'][7]['lda'])])
	q = queue.Queue()
	for b in below_ids:
		b['counter']=counter		
		q.put(b)
		counter+=1
	thread_count = 48


	for i in range(thread_count):
		t = threading.Thread(target=get_info, args = (q,))
		t.start()

	q.join()
