from nltk.corpus import swadesh

def run():
    corpus=swadesh.entries(['es','en'])

    trad=dict(corpus)
    print(trad['palo'])

if __name__=='__main__':
    run()