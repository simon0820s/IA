import nltk
import matplotlib.pyplot as plt
from nltk.book import *
import numpy as np

nltk.download('book')

def run():
    vocabulary=sorted(set(text1)) #convert to set for unique values
    lexic_richness=lambda x,y : len(x)/len(y)
    text1_lr=lexic_richness(vocabulary,text1)
    print(text1_lr)
    
if __name__=='__main__':
    run()