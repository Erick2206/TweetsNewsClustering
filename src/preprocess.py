import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import json
import gensim
import pickle

docFile='docFile.pkl'
labelFile='labelFile.pkl'
LabeledSentence = gensim.models.doc2vec.LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])


def preprocess(x):
    sentences=x.split('\n')
    tokenized=[]
    print sentences
    for sentence in sentences:
        tokenized.append([word_tokenize(word) for word in sentence])


def createDocList(x):
    x=x.replace('\n',' ').lower()
    docList.append(x)


path='/home/erick/Repo/IIITD/Doc2Vec/Dataset'
docList=[]
labelList=[]
for root,dirs,files in os.walk(path):
    for file in files:
        try:
            with open(os.path.join(root,file)) as content:
                data=json.load(content)

                createDocList(data['text'])
                labelList.append(file)

        except Exception as e:
            print "Error", e
            break

print "docList created!"
with open(docFile,'wb') as f:
    pickle.dump(docList,f)

with open(labelFile,'wb') as f:
    pickle.dump(labelList,f)

print labelList
exit(0)
print len(labelList)
print len(docList)
print labelList[0]

it = LabeledLineSentence(docList, labelList)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=4,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)

for epoch in range(10):
    model.train(it,total_examples=model.corpus_count,epochs = model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    model.train(it,total_examples=model.corpus_count,epochs = model.iter)

model.save("doc2vec1.model")
