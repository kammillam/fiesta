"""
This module divides a documents into individuals words or sequence of words by splitting on the blank spaces.
"""
from nltk import ngrams, word_tokenize
from fiesta.transformers.document_transformer import document_transformer

def tokenization (document_collection, index_of_document = None):
    """This method divides a documents into individual words (strings) by splitting on the blank spaces.
        Args:
            document_collection (str, list or file directory): document collection to be tokenized 
        Returns: 
            list: list of divided documents into individual words
    """
    full_document = document_transformer(document_collection)
    tokenized_document = []
    for document_part in full_document:
        tokenized_document.append(word_tokenize(document_part))
    if index_of_document!= None:
        return tokenized_document[index_of_document]
    return tokenized_document

def n_grams_tokenization (document_collection, n, index_of_document = None):
    """Divides a documents into sequence of n words (strings) by splitting on the blank spaces.
        Args:
            document_collection (str, list or file directory): document collection to be tokenized
 			n (int): length of the sequence of words
        Returns: 
            list: list of divided documents into sequences of words
    """
    all_n_grams = []
    full_document = document_transformer(document_collection)

    for document_part in full_document:
        ngrams_document = []
        n_grams = ngrams(document_part.split(), n)
        for grams in n_grams:
            ngrams_document.append(grams)
        all_n_grams.append(ngrams_document)
    if index_of_document!= None:
        return all_n_grams[index_of_document]
    return all_n_grams