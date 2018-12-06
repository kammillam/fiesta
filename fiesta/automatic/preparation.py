"""
This module contains methods for automatic preprocessing of text, feature extraction and feature selection.
"""
from fiesta.preprocessing.normalizing import stop_words, lemmatizer, stemming
from fiesta.feature_extraction.bag_of_words import bag_of_words, words_counting
import fiesta.feature_selection.selection as fs
from fiesta.feature_extraction.tfidf import tfidf
import pandas as pd

def extract_features (document_collection, language_of_documents = "en", preprocessing_only = False):
    """This method preprocesses the documents and extracts features from the preprocessed text. 
        Args:
            document_collection (str, list or file directory): document collection to extract the features of
            language_of_documents (str): (default „en“) language of the document collection 
			preprocessing_only (bool): (default False) if True returns only preprocessed documents 
        Returns: 
            list: list of preprocessed documents
            pandas.core.frame.DataFrame: DataFrame with matrix of token counts, tf-idf representation of the document collection and preprocessed documents. 
    """
    documents_without_stop_words = stop_words(document_collection, language=language_of_documents)
    lemmatized_documents = lemmatizer (documents_without_stop_words, language=language_of_documents)
    stemmed_documents = stemming (lemmatized_documents, language=language_of_documents)
    if preprocessing_only:
        return stemmed_documents
    documents_df = pd.DataFrame (stemmed_documents)
    documents_bag_of_words = bag_of_words(stemmed_documents)
    documents_tfidf = tfidf(stemmed_documents)
    
    result_df = pd.concat([documents_bag_of_words, documents_tfidf, documents_df], keys=["bag of words",  "tfidf", "preprocessed documents"], join = "outer")
    return result_df

def select_features (document_collection_cat1, document_collection_cat2 = None):
    """This method selects relevant features based on the methods of the feature_selection module. 
        Args:
            document_collection_cat1 (str, list or file directory): document collection of the first category
            document_collection_cat2 (str, list or file directory): (default None) document collection of the second category
        Returns: 
            pandas.core.series.Series: relevant features for one document collection selected with two methods of the feature_selection module
            pandas.core.frame.DataFrame: relevant features for two document collections selected with all five methods 
            of the feature_selection module    
    """
    term_frequency_features = fs.term_frequency_selection(document_collection_cat1, document_collection_cat2)
    tfidf_features = fs.tfidf_selection(document_collection_cat1, document_collection_cat2)

    if document_collection_cat2 == None:
        
        selected_features_df = pd.concat([term_frequency_features, tfidf_features], keys= ["term frequency", "tfidf"], join = "outer")
        return selected_features_df

    information_gain_features = fs.information_gain(document_collection_cat1, document_collection_cat2)
    chi_square_features = fs.chi_square (document_collection_cat1, document_collection_cat2)
    lsa_features = fs.latent_semantic_analysis(document_collection_cat1, document_collection_cat2)

    selected_features_df = pd.concat([term_frequency_features, tfidf_features, information_gain_features, chi_square_features, lsa_features], keys= ["term frequency", "tfidf" ,"information gain", "chi square", "latent semantic analysis"], join = "outer")
    return selected_features_df

def extract_relevant_features(document_collection_cat1, document_collection_cat2 = None, language_of_document_collection = "en"):
    """Firstly  prepares this method the document collection, then it extracts features and at the end it selects relevant features
    selects relevant features based on the methods of the feature_selection module. 
         Args:
            document_collection_cat1 (str, list or file directory): document collection of the first category	
            document_collection_cat2 (str, list or file directory): (default None) document collection of the second category
            language_of_document_collection (str): (default „en“) language of the document collection 
        Returns: 
            pandas.core.series.Series: relevant features for one preprocessed document collection selected with two methods of the feature_selection module
            pandas.core.frame.DataFrame: relevant features for two preprocessed document collections selected with all five methods 
            of the feature_selection module    
    """
    print (type(document_collection_cat2))
    extracted_features_cat1 = extract_features(document_collection_cat1, language_of_documents=language_of_document_collection, preprocessing_only=True )
    extracted_features_cat2 = None
    if document_collection_cat2 != None:
        extracted_features_cat2 = extract_features(document_collection_cat2, language_of_documents=language_of_document_collection, preprocessing_only=True )
    
    relevant_features = select_features(extracted_features_cat1, extracted_features_cat2)
    return relevant_features