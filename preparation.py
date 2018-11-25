from fiesta.normalizing import stop_words, lemmatizer, stemming
from fiesta.bag_of_words import bag_of_words, words_counting
import fiesta.feature_selection as fs
from fiesta.tfidf import tfidf
import pandas as pd

def extract_features (document, language = "en", optional = False):
    documents_without_stop_words = stop_words(document, language=language)
    lemmatized_documents = lemmatizer (documents_without_stop_words, language=language)
    stemmed_documents = stemming (lemmatized_documents, language=language)
    documents_df = pd.DataFrame (stemmed_documents)
    documents_bag_of_words = bag_of_words(stemmed_documents)
    documents_tfidf = tfidf(stemmed_documents)
    document_tokens_counter = words_counting(stemmed_documents)
    result_df = pd.concat([documents_bag_of_words, document_tokens_counter, documents_tfidf, documents_df], keys=["bag of words", "counter", "tfidf", "preprocessed documents"], join = "outer")

    if optional:
        return stemmed_documents
      
    return result_df

def select_features (category1,category2):
    term_frequency_features = fs.term_frequency_selection(category1, category2)
    tfidf_features = fs.tfidf_selection(category1, category2)
    information_gain_features = fs.information_gain(category1, category2)
    chi_square_features = fs.chi_square (category1, category2)
    lsa_features = fs.latent_semantic_analysis(category1, category2)

    selected_features_df = pd.concat([term_frequency_features, tfidf_features, information_gain_features, chi_square_features, lsa_features], keys= ["term frequency", "tfidf" ,"information gain", "chi square", "latent semantic analysis"], join = "outer")

    return selected_features_df

def extract_relevant_features(category1, category2, language_of_categories = "en"):
    extracted_features_cat1 = extract_features(category1, language=language_of_categories, optional=True )
    extracted_features_cat2 = extract_features(category2, language=language_of_categories, optional=True )
    relevant_features = select_features(extracted_features_cat1, extracted_features_cat2)

    return relevant_features

