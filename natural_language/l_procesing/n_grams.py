import nltk
nltk.download('book')
from nltk.book import *
from nltk.util import ngrams
import os


def run():
    #bigrams
    extract_b=list(bigrams(text1))
    distribution_b=FreqDist(extract_b)
    filtered_bigrams_b=[b for b in distribution_b if len(b[0])>3 and len(b[1])>3]
    print(filtered_bigrams_b)
    
    #N_grams only use n_grams import and especify N


    print("")
    n=int(input('please enter your "N" number => '))
    os.system("cls")
    extract_n=list(ngrams(text1,n)) #N=3 so we have a threegrams
    distribution_n=FreqDist(extract_n)
    filtered_n_grams=[n for n in distribution_n if len(n[0])>3 and len(n[1])>3 and len(n[2])>3]
    print(filtered_n_grams)
    
if __name__=='__main__':
    run()