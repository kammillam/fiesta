from sklearn.feature_extraction.text import CountVectorizer

def BoW (document = None, file = None, indexOfDocument = None):
   
    vectorizer = CountVectorizer()
    if file != None :    
        corpus = []
        file = open(file, "r")
        for line in file:
            corpus.append(line)
        file.close()  
    
    elif document != None:
        corpus = document
    
    termDocumentMatrix = vectorizer.fit_transform(corpus).toarray()
    
    if indexOfDocument != None :
        return termDocumentMatrix[indexOfDocument]
    else : 
        return termDocumentMatrix
    
        
