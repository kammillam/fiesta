"""
This module allows transfromation of document collection into tf- and tfidf-representation and
calculation of idf-weights of each term in the document collection.
"""
from sklearn.feature_extraction.text import TfidfTransformer
from fiesta.transformers.document_transformer import document_transformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



def term_frequency (document_collection, index_of_document = None, specific_word = None, scaled = False): 
    """This method calculates term frequency in each document of the document collection 
    Args:
        document_collection (str, list of strings or file directory): document collection 						
        index_of_document(int):  (default None) index of the document whose vector is to be returned
		specific_word(str): (default None) word whose vector is to be returned
		scaled(bool):  (default False) True, if scaling relative to the frequency of words in the document is needed
    Returns:
        pandas.core.frame.DataFrame: tf representation of the document collection
        pandas.core.series.Series: tf representation of the selected document or word
    """   
    vectorizer = CountVectorizer()
    full_document = document_transformer(document_collection)

    termFrequency = vectorizer.fit_transform(full_document).toarray()
    vocabulary = vectorizer.get_feature_names() 
    if scaled == True:
        termFrequency = scale (termFrequency)
         
    term_frequency_df = pd.DataFrame(termFrequency, columns=vocabulary)
    if index_of_document != None:
        return term_frequency_df.iloc[index_of_document]
    elif specific_word != None:
        return term_frequency_df.loc[:,specific_word]
    else:
        return term_frequency_df

def inverse_document_frequency (document_collection , smooth = True, specific_word = None):
    """This method calculates idf-weights for each term in the document collection
    Args:
        document_collection (str, list or file directory): document collection					
        smooth(bool): (default True) add one to document frequencies and prevents zero divisions
		specific_word(str): (default None) word whose idf-weight is to be returned
    Returns: 
        pandas.core.frame.DataFrame: idf-weights of each feature in the document collection
		pandas.core.series.Series: idf-weight of selected word
    """   
    vectorizer = CountVectorizer()
    full_document = document_transformer(document_collection)

    termFrequency = vectorizer.fit_transform(full_document).toarray()
    transformer = TfidfTransformer(smooth_idf=smooth)
    
    transformer.fit_transform(termFrequency).toarray()
    vocabulary = vectorizer.get_feature_names()
    idf_values = transformer.idf_
    idf_values_df = pd.DataFrame(idf_values, index = vocabulary, columns = ["idf"])

    if specific_word != None:
        return idf_values_df.loc[specific_word]
    else:
        return idf_values_df


def tfidf (document_collection, smooth = True, index_of_document = None):
    """This method calculates tf-idf weights for each term in the document collection  
    Args:
        document_collection (str, list or file directory): document collection
		smooth(bool): (default True) add one to document frequencies and prevents zero divisions
		indexOfDocument(int): (default None) index of the document whose vector is to be returned
    Returns:
        pandas.core.frame.DataFrame: tf-idf representation of the document collection
		pandas.core.series.Series: tf-idf representation of selected document
    """   
    vectorizer = CountVectorizer()
    full_document = document_transformer(document_collection)

    termFrequency = vectorizer.fit_transform(full_document).toarray()
    transformer = TfidfTransformer(smooth_idf=smooth)
    tfidf = transformer.fit_transform(termFrequency).toarray()
    vocabulary = vectorizer.get_feature_names()
   
    tfidf_df = pd.DataFrame(tfidf, columns = vocabulary )
    if index_of_document != None:
        return tfidf_df.iloc[index_of_document]
    else: 
        return tfidf_df
    
def scale (termFrequency) :
    """This method scales tf-representation of the document colletion relative to the frequency of words in the document
    Args:
        termFrequency(numpy.ndarray): tf representation of the document colleciton
    Returns:
        list: scaled term frequency representation relative to the frequency of words in the document
    """   
    if len(termFrequency) == 1:
        scaled_term_frequency = termFrequency / max (termFrequency)
        return scaled_term_frequency
    else:
        scaled_term_frequency = []
        for term in termFrequency:
            scaled_term = term / max (term)
            scaled_term_frequency.append(scaled_term)
        return scaled_term_frequency