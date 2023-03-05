import nltk
nltk.download('book')
from nltk.book import *
import pandas as pd
import numpy as np

def run():
    text=list(bigrams(text1))

    #extract and filter text
    filtered_bigrams=[b for b in text if len(b[0])>2 and len(b[1])>2]
    dis_bigrams=FreqDist(filtered_bigrams)

    #Create Dataframe
    df=pd.DataFrame()
    df['bi_grams']=list(set(dis_bigrams))

    #Calculating PMI
    filtered_words=[word for word in text1 if len(word)>2]
    filtered_word_dis=FreqDist(filtered_words)

    #generate  columns
    df['word_1']= df['bi_grams'].apply(lambda x:x[0])
    df['word_2']= df['bi_grams'].apply(lambda x:x[1])
    df['bi_grams_freq']=df['bi_grams'].apply(lambda x: dis_bigrams[x])
    df['word_1_freq']=df['word_1'].apply(lambda x: filtered_word_dis[x])
    df['word_2_freq']=df['word_2'].apply(lambda x: filtered_word_dis[x])
    df['PMI']=df[['bi_grams_freq','word_1_freq','word_2_freq']].apply(lambda x:np.log2(x.values[0]/(x.values[1]*x.values[2])),axis=1)
    df.sort_values(by='PMI',ascending=False)
if __name__=='__main__':
    run()