import re

def run(text):
    text=re.split("\n",text)
    text=[w for w in text if len(w)>2]
    
    dictionary={}
    for w in text:
        print(w)
        a,b=re.split(":",w)
        dictionary[a]=b

    return dictionary

if __name__=='__main__':
    run()