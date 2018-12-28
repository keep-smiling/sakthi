import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
sid = SentimentIntensityAnalyzer()
filepath = 'second.txt' 
count = 0
f=open("positive.txt","a+") 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        #print(line)
        ss = sid.polarity_scores(line)
        if (ss['neg']==0 and ss['pos']>0) :
        	print(ss['neg'],ss['pos'])
        	print(line)
        	f.write(line)
        	count+=1
        line = fp.readline()
f.close()
print("Total")
print(count)