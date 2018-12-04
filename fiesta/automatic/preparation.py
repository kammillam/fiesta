from fiesta.preprocessing.normalizing import stop_words, lemmatizer, stemming
from fiesta.feature_extraction.bag_of_words import bag_of_words, words_counting
import fiesta.feature_selection.selection as fs
from fiesta.feature_extraction.tfidf import tfidf
import pandas as pd

import timeit

def extract_features (document, language_of_categories = "en", optional = False):
    start = timeit.default_timer()
    documents_without_stop_words = stop_words(document, language=language_of_categories)
    print('nach stop words: ', timeit.default_timer() - start) 
    start = timeit.default_timer()
    lemmatized_documents = lemmatizer (documents_without_stop_words, language=language_of_categories)
    print('nach lemmatizer: ', timeit.default_timer() - start)  
    start = timeit.default_timer()
    stemmed_documents = stemming (lemmatized_documents, language=language_of_categories)
    print('nach stemming: ', timeit.default_timer() - start)  
    start = timeit.default_timer()
    if optional:
        return stemmed_documents
    documents_df = pd.DataFrame (stemmed_documents)
    documents_bag_of_words = bag_of_words(stemmed_documents)
    print('nach bag of words: ', timeit.default_timer() - start)  
    start = timeit.default_timer()
    documents_tfidf = tfidf(stemmed_documents)
    print('nach tfidf: ', timeit.default_timer() - start)  
    start = timeit.default_timer()
    document_tokens_counter = words_counting(stemmed_documents)
    print('nach counting: ', timeit.default_timer() - start)  
    result_df = pd.concat([documents_bag_of_words, documents_tfidf, documents_df], keys=["bag of words", "counter", "tfidf", "preprocessed documents"], join = "outer")
  
    return result_df

def select_features (category1,cat2): #für eine Dokumetntensammlung anpassen 
    term_frequency_features = fs.term_frequency_selection(category1, category2 = cat2)
    tfidf_features = fs.tfidf_selection(category1, category2 = cat2)
    information_gain_features = fs.information_gain(category1, cat2)
    chi_square_features = fs.chi_square (category1, cat2)
    lsa_features = fs.latent_semantic_analysis(category1, cat2)

    selected_features_df = pd.concat([term_frequency_features, tfidf_features, information_gain_features, chi_square_features, lsa_features], keys= ["term frequency", "tfidf" ,"information gain", "chi square", "latent semantic analysis"], join = "outer")

    return selected_features_df

def extract_relevant_features(category1, category2, language_of_categorie = "en"):
    extracted_features_cat1 = extract_features(category1, language_of_categories=language_of_categorie, optional=True )
    extracted_features_cat2 = extract_features(category2, language_of_categories=language_of_categorie, optional=True )
    relevant_features = select_features(extracted_features_cat1, extracted_features_cat2)

    return relevant_features

