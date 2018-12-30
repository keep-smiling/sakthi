# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 10:32:38 2018

@author: Girinathaprasad
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
sid = SentimentIntensityAnalyzer()
filepath = 'preprocessed_nonsarcastic.txt' 

count = 0
f1=open("preprocessed_nonsarcastic1.txt","a+")
f2=open("preprocessed_nonsarcastic2.txt","a+")
 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        #print(line)
        ss = sid.polarity_scores(line)
        val= abs(ss['neg']-ss['pos'])
        if val>0.3:
            f1.write(line)
            
        else:
            f2.write(line)
            
        line = fp.readline()
f1.close()
f2.close()
print("Total")
print(count)