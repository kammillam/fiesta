"""
This module enables a transformation of the document collection into vector space representation
and the counting of occurrences of the respective word in the document collection.
"""
from sklearn.feature_extraction.text import CountVectorizer
from fiesta.transformers.document_transformer import document_transformer
from os.path import isfile
import pandas as pd

def bag_of_words (document_collection, index_of_document = None, specific_word = None, array = False, binary_count = False):
    """This method converts a collection of text documents to a matrix of token counts
    Args:
        document_collection (str, list of strings or file directory): document collection 
		indexOfDocument (int): (default None) index of the document whose vector is to be returned
		specific_word (str): (default None) word whose vector is to be returned
        array (bool): (default False) True, if Bag of Words representation as ndarray is needed for other method
    Returns:
        pandas.core.frame.DataFrame: Vector representation for all documents
        pandas.core.series.Series: vector representation for selected document or word
        numpy.ndarray: (if optional = True) Vector representation for all documents for another methods
    """   
    vectorizer = CountVectorizer(binary=binary_count)
    full_document = document_transformer(document_collection)

    term_document_matrix = vectorizer.fit_transform(full_document).toarray()
    if array:
        return term_document_matrix
    vocabulary = vectorizer.get_feature_names() 
    bag_of_words = pd.DataFrame(term_document_matrix, columns=vocabulary)
    if index_of_document != None:
        return bag_of_words.iloc[index_of_document]
    elif specific_word != None:
        return bag_of_words.loc[:,specific_word]
    else:
        return bag_of_words
        
def words_counting (document_collection, specific_word = None, binary = False):
    """This method counts the number of words in the whole document collection
    Args:
        document_collection (str, list of strings or file directory): document collection
    	specific_word (str): (default None) word whose number is to be returned
    Returns:
        pandas.core.series.Series: an assignment of terms to number of terms in all documents	
    """
    full_document = document_transformer(document_collection)
    tf = bag_of_words(full_document, binary_count=binary)

    tf_sum = tf.sum()

    tf_sum_sorted= tf_sum.sort_values(ascending=False)
    return tf_sum_sorted

