import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer


def run():
    url='https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'

    r=requests.get(url)
    html=r.text
    soup=BeautifulSoup(html,'html.parser')
    text=soup.get_text()
    tokens=re.findall('\w+',text)
    print(tokens)
    
if __name__=='__main__':
    run()