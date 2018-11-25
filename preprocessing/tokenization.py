from nltk import ngrams, word_tokenize
from fiesta.feature_extraction.bag_of_words import document_transformer

def tokenization (document, index_of_document = None):
    full_document = document_transformer(document)
    tokenized_document = []
    for document_part in full_document:
        tokenized_document.append(word_tokenize(document_part))
    if index_of_document!= None:
        return tokenized_document[index_of_document]
    return tokenized_document

def n_grams_tokenization (document, n, index_of_document = None):
    """This is docstring""" 
    all_n_grams = []
    full_document = document_transformer(document)

    for document_part in full_document:
        ngrams_document = []
        n_grams = ngrams(document_part.split(), n)
        for grams in n_grams:
            ngrams_document.append(grams)
        all_n_grams.append(ngrams_document)
    if index_of_document!= None:
        return all_n_grams[index_of_document]
    return all_n_grams
