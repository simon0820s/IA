import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from urllib import request


def run():
    #Get text
    url="http://www.gutenberg.org/files/2554/2554-0.txt"
    response=request.urlopen(url)
    raw=response.read().decode('utf8')

    #Tokenization
    print(f"logitud del texto {len(raw)} chars")
    tokens=word_tokenize(raw)
    tokens=[t for t in tokens if len(t)>2]

    #procesing
    text=nltk.Text(tokens)
    print(f"Collocaciones:  {text.collocations()}")

if __name__=='__main__':
    run()