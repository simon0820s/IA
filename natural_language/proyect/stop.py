from nltk.corpus import stopwords
import nltk
from nltk.probability import FreqDist

def run(text):
    stop=set(stopwords.words('spanish'))
    words_text=nltk.word_tokenize(text.lower())
    filtered_text=[w for w in words_text if w.isalnum() and w not in stop and len(w)>1]
    text=''.join(filtered_text)
    return text
