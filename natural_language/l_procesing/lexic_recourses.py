import nltk
nltk.download('book')
from nltk.book import *
from nltk.corpus import stopwords

def run():
    vocabulary=sorted(set(text1))
    print(stopwords_percentage(text1))

def stopwords_percentage(text):
    stopwd=stopwords.words('english')
    print(stopwd)
    content=[w for w in text if w.lower() not in stopwd]

    return round((len(content)/len(text))*100,2)

if __name__=='__main__':
    run()