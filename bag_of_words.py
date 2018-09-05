from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile

def bag_of_words ( document, indexOfDocument = None, separate = None):
    """This is docstring"""   
    
    vectorizer = CountVectorizer()
    full_document = document_transformer(document)
    
    termDocumentMatrix = vectorizer.fit_transform(full_document).toarray()

    if indexOfDocument != None :
        return termDocumentMatrix[indexOfDocument]
    else: 
        return termDocumentMatrix
    
def bag_of_words_vocabulary(document):
    """This is docstring"""

    vectorizer = CountVectorizer()
    full_document = document_transformer(document)
        
    vectorizer.fit_transform(full_document).toarray()
    vocabulary = vectorizer.vocabulary_ 

    index_vocabulary = {}
    vocabulary_sorted = sorted(vocabulary.keys())
    
    for word in vocabulary_sorted:
        index_vocabulary[word] = vocabulary.get(word)
    
    return index_vocabulary

def words_counting (document):
    """This is docstring"""
    full_document = document_transformer(document)
    wordcount = {}

    for document_part in full_document:
        for word in document_part.split():
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    
    new_wordcount = sorted(wordcount.keys())
    sorted_wordcount = {}
    
    for word in new_wordcount:
        sorted_wordcount[word] = wordcount.get(word)
    
    return sorted_wordcount

def document_transformer  (document):
    """This method returns list of strings."""
    
    full_document = []

    if  type(document) == list:
        return document
    elif isfile(document):    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  
        return full_document
    elif type(document) == str:
        full_document.append(document)
        return full_document
