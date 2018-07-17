from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile

def bag_of_words ( document, indexOfDocument = None, separate = None):
    """This is docstring"""   
    
    vectorizer = CountVectorizer()
    full_document = []
    
    if  type(document) == list:
        full_document = document
    
    elif isfile(document):    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  

    elif type(document) == str:
        full_document.append(document)

    termDocumentMatrix = vectorizer.fit_transform(full_document).toarray()

    if indexOfDocument != None :
        return termDocumentMatrix[indexOfDocument]
    else: 
        return termDocumentMatrix
    
def bag_of_words_vocabulary(document):
    """This is docstring"""

    vectorizer = CountVectorizer()
    full_document = []
    
    if type(document) == list:
        full_document = document        

    elif isfile(document) :    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  
    
    elif type(document) == str:
        full_document.append(document)
    
    vectorizer.fit_transform(full_document).toarray()
    vocabulary = vectorizer.vocabulary_ 

    index_vocabulary = {}
    vocabulary_sorted = sorted(vocabulary.keys())
    for word in vocabulary_sorted:
        index_vocabulary[word] = vocabulary.get(word)
    return index_vocabulary
    
        

     
