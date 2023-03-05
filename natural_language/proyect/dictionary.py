import re

def run(text):
    text=re.split("\n",text)
    text=[w for w in text if len(w)>2]
    
    dictionary={}
    for w in text:
        a,b=re.split(":",w)
        dictionary[a]=b
    print(dictionary)

if __name__=='__main__':
    run()