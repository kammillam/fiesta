from sklearn.feature_extraction.text import CountVectorizer
def BoW (document = None, file = None):
    vectorizer = CountVectorizer()
    if file != None :    
        corpus = []
        file = open(file, "r")
        for line in file:
            corpus.append(line)
        file.close()
        print (corpus)
        print( vectorizer.fit_transform(corpus).todense() )
        print (vectorizer.vocabulary_)
    elif document != None:
        corpus = document
        print (document)
        print( vectorizer.fit_transform(document).todense() )
        print (vectorizer.vocabulary_)
