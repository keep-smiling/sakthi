import re
def clean_data(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

f=open("preprocessed_sarcastic.txt","a+") 
filepath = "sarcastic.txt"
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        line = clean_data(line)
        f.write(line)
        f.write("\n")
        line = fp.readline()
f.close()