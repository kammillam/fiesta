from nltk import ngrams
from os.path import isfile

def n_grams (document, n):
    """This is docstring""" 
    all_n_grams = []
    if type(document) == list:
        
        for document_part in document:
            ngrams_document = []
            n_grams = ngrams(document_part.split(), n)
            for grams in n_grams:
                ngrams_document.append(grams)
            all_n_grams.append(ngrams_document)

        return all_n_grams
    
    elif isfile(document) :
        full_document = []    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  

        for document_part in full_document:
            ngrams_document = []
            n_grams = ngrams(document_part.split(), n)
            for grams in n_grams:
                ngrams_document.append(grams)
            all_n_grams.append(ngrams_document)

        return all_n_grams

    elif type(document) == str:

        n_grams = ngrams(document.split(), n)
        for grams in n_grams:
            all_n_grams.append(grams)

        return all_n_grams
