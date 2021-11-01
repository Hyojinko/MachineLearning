import math
import numpy as np
import string
from nltk.corpus import stopwords
import collections

tokenizedQuery = {}
abstract_words = {}
query = []
queryScore = []
simList = []
stopset = [w for w in stopwords.words('english')] + list(string.punctuation)

#query index
qi=0
#abstract index
ai = -1

# process query
def get_query (current, qi):
	#i is query index
	wordArray = []
	for w in current:
		if w in stopset:
			continue
		if w not in wordArray:
			wordArray.append(w)

		if w not in tokenizedQuery:
			tokenizedQuery[w] ={}
		tokenizedQuery[w][qi] = tokenizedQuery[w].get(qi, 0.0) + 1

	for w in wordArray:
		tokenizedQuery[w]['SUM'] = tokenizedQuery[w].get('SUM', 0.0) + 1

	query.append(wordArray)
# process abstract
def get_abstract (current, ai):
	absArray = [] # i is index for abstract
	for w in current:
		if w in stopset:
			continue
		if w not in absArray:
			absArray.append(w)

		if w not in abstract_words:
			abstract_words[w] = {}
		abstract_words[w][ai] = abstract_words[w].get(ai, 0.0) + 1

	for w in absArray:
		abstract_words[w]['SUM'] = abstract_words[w].get('SUM', 0.0) + 1


f = open('/Users/kohyojin/Desktop/SoftWare/software2021/2021-2/machine learning/week9/cran/cran.qry.txt','r')
current = []
for line in f:
	if ".I" in line:
		if current != []:
			get_query(current, qi)
			current = []
			qi += 1
	elif ".W" in line:
		continue
	else:
		current += line

get_query(current, qi)
for i in range(225):
	qscore = []
	for w in query[i]:
		tf = tokenizedQuery[w][i]
		idf = math.log(225.0/tokenizedQuery[w]['SUM'])
		qscore.append(tf*idf)
	queryScore.append(qscore)
f.close()

f = open('/Users/kohyojin/Desktop/SoftWare/software2021/2021-2/machine learning/week9/cran/cran.all.1400.txt','r')
ai = -1
current = []
for line in f:
	if ".I" in line:
		first = True
		if current != []:
			get_abstract(current, ai)
		ai += 1
	elif ".W" in line:
		if(first == True):
			current = []
			first = False
	else:
		current += line
get_abstract(current, ai)
f.close()
f = open('output.txt','w')
for i in range(225):
	abs_sim = []
	for j in range (1400):
		absScore = []
		for w in query[i]:
			if w not in abstract_words:
				tf = 0.0
				idf = 0.0
			else:
				tf = abstract_words[w].get(j,0.0)
				idf = math.log(1400.0/abstract_words[w]['SUM'])
			absScore.append(tf*idf)
		sim = np.dot(queryScore[i], absScore)
		if sim != 0:
			sim /= math.sqrt((np.dot(queryScore[i],queryScore[i]))*np.dot(absScore, absScore))
		abs_sim.append(sim)
	abs_sim_idx = sorted(range(len(abs_sim)),key = lambda k: abs_sim[k], reverse=True)
	abs_sim.sort(reverse = True)
	for j in range(1400):
		numOfQuery = str(i+1)
		numOfAbs = str(abs_sim_idx[j] +1)
		sim = '{0:.15f}'.format(abs_sim[j])
		#sim = abs_sim[j]
		if i ==0 and j == 0:
			f.write(numOfQuery + ' ' + numOfAbs+ ' ' + sim)
		else:
			f.write('\n' +numOfQuery+' '+numOfAbs+' '+ sim)

f.close()


