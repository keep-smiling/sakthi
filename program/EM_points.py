# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 10:18:08 2018

@author: Girinathaprasad
"""

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxsize
import math
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
features_dictionary = {}
def dictionary_creation():	
	filepath = "features_example.txt"
	with open(filepath) as fp:  
	   line = fp.readline()
	   while line:
	        line = line.split();
	        val = line[0]+","+line[1]
	        features_dictionary[val]= line[2]
	        line = fp.readline()
	print(features_dictionary)

def cummulation(line):
    wordsList = nltk.word_tokenize(line)
    tagged = nltk.pos_tag(wordsList)
    words = []
    labels = []
    for i in tagged:
        words.append(i[0])
        labels.append(i[1])
    val = 0.0
    for i in range(1, len(labels)):
        lab = labels[i-1]+","+labels[i]
        if lab in features_dictionary:
            v = features_dictionary[lab]
        else:
            v = float(3.0)
        val += float(v)
    return val

if __name__ == '__main__':
    dictionary_creation()
    sid = SentimentIntensityAnalyzer()
    x1 = []
    y1 = []
    filepath = 'preprocessed_sarcastic.txt'
    with open(filepath) as fp:  
        line = fp.readline()
        while line:
            ss = sid.polarity_scores(line)
            val= abs(ss['neg']-ss['pos'])
            x1.append(val)
            y1.append(cummulation(line))
            line = fp.readline()
    x2 = []
    y2 = []
    filepath = 'preprocessed_nonsarcastic3.txt'
    with open(filepath) as fp:  
        line = fp.readline()
        while line:
            ss = sid.polarity_scores(line)
            val= abs(ss['neg']-ss['pos'])
            x2.append(val)
            y2.append(cummulation(line))
            line = fp.readline()   
            
    v1 = len(x1)
    v2 = len(x2)
    xs = np.concatenate((x1, x2))
    ys = np.concatenate((y1, y2))
    labels = ([1] * v1) + ([2] * v2)
    data = {'x': xs, 'y': ys, 'label': labels}
    df = pd.DataFrame(data=data)
    print(df)
    print(v1)
    print(v2)
    fig = plt.figure()
    plt.scatter(data['x'], data['y'], 5, c=data['label'])
    fig.set_size_inches(20,20)
    fig.savefig("true-values1.png")