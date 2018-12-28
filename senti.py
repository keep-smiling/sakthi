import nltk
import random
import math
import pandas as pd 

from nltk.sentiment.vader import SentimentIntensityAnalyzer 

#hotel_rev = ['Great place to be when you are in Bangalore.','The place was being renovated when I visited so the seating was limited.','Loved the ambience, loved the food','The food is delicious but not over the top.','Service - Little slow, probably because too many people.','The place is not easy to locate','Mushroom fried rice was tasty']

#hotel_rev = ["100% gonna have to go out for tea tonight because I'll either burn the house down or give myself food poisoning if I make it myself ??"]
sid = SentimentIntensityAnalyzer()


data = pd.read_csv('exampletweets.csv', error_bad_lines=False);
data_text = data[['data']]
data_text['index'] = data_text.index
documents = data_text
count = 0
for i in range(899):
    txt = documents[documents['index'] == i].values[0][0]        
    #print(txt)
    ss = sid.polarity_scores(txt)
    if (ss['neg']>0 and ss['pos']>0) :
         print(ss['neg'],ss['pos'])
         print (txt)
         count += 1
    
print("total")
print(count)

'''
for sentence in hotel_rev:
     ss = sid.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1},'.format(k, ss[k]), end='')
         #print(k,ss[k]) 
     #if (ss['neg']>0 and ss['pos']>0) :
     #    print(ss['neg'],ss['pos'])
     #    print (sentence)
     print()
'''