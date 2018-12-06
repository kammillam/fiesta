"""
This module implements feature selection methods.
"""
from nltk import word_tokenize
from math import log
import pandas as pd
from fiesta.transformers.document_transformer import document_transformer
from fiesta.feature_extraction import words_counting, bag_of_words
from fiesta.feature_extraction.tfidf import tfidf
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pylab import *
import numpy

def term_frequency_selection(document_collection_category1, document_collection_category2 = None, specific_word = None, list_size = 10):
    """This method selects relevant features from one or two document collections based on the frequency of occurrence of the word in the document collection.
        Args:
            document_collection_category1 (str, list or file directory):
			document_collection_category2 (str, list or file directory):
            specific_word(str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size (int): (default 10) number of features to be returned
        Returns: 
            pandas.core.series.Series: most relevant features and their frequency in all documents           
    """
    cat1 = document_transformer(document_collection_category1)
    if category2 != None:
        cat2 = document_transformer(category2)
        full_document = cat1+cat2

    else:
        full_document = cat1

    tf_sum = words_counting(full_document)

    if specific_word != None:
        return pd.Series({specific_word : tf_sum.loc[specific_word]})
    tf_sort=tf_sum.sort_values(ascending=False)
    return tf_sort[:list_size]

def tfidf_selection (document_collection_category1, document_collection_category2 = None, specific_word = None, list_size = 10):
    """This method selects relevant features from one or two document collections based on the tf-idf value of the word.
        Args:
            document_collection_category1(str, list or file directory): document collection of the first category
			document_collection_category2(str, list or file directory): (dafualt None) document collection of the second category
            specific_word(str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size(int): (default 10) number of features to be returned
        Returns:
            pandas.core.series.Series: most relevant features and sum of their Tf-idf weights in all documents            
    """
    cat1 = document_transformer(document_collection_category1)
    if category2 != None:
        cat2 = document_transformer(document_collection_category2)
        full_document = cat1+cat2

    else:
        full_document = cat1

    tf_idf = tfidf(full_document)

    tf_idf_sum = tf_idf.sum()
    
    if specific_word != None:
        return pd.Series({specific_word : tf_idf_sum.loc[specific_word]})
    
    tf_idf_sort=tf_idf_sum.sort_values(ascending=False)
    return tf_idf_sort[:list_size]


def information_gain(document_collection_category1, document_collection_category2, specific_word = None, list_size = 10, visualize = False):
    """This method selects relevant features from two document collections based on the information gain algorithm.
        Args:
            document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
            specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size (int): (default 10) number of features to be returned
            visualize (bool): (default False) if True it represents the features graphically
        Returns:
            pandas.core.series.Series: most relevant features and their their information gain values             
    """
    cat1 = document_transformer(document_collection_category1)
    cat2 = document_transformer(document_collection_category2)

    individual_words = bag_of_words(cat1+cat2).columns.tolist()
    ig = { }
    cat1_words_counting = words_counting(cat1, binary=True)
    cat2_words_counting = words_counting(cat2, binary=True)

    for word in individual_words:
        cat_a = 0 #kommt in der Kategorie 
        cat_b = 0 #kommt in der anderen Kategorie 
        not_cat_a = 0 # kommt nicht in der Kategorie   
        not_cat_b = 0 # kommt nicht in der anderer Kategorie   
        if word in cat1_words_counting.index.values:
            cat_a = cat1_words_counting.loc[word] 
        not_cat_a = len(cat1) - cat_a
    
        if word in cat2_words_counting.index.values:
            cat_b = cat2_words_counting.loc[word] 
        
        not_cat_b = len(cat2) - cat_b

        if cat_a*cat_b*not_cat_a*not_cat_b!=0:
            all_words = cat_a + cat_b + not_cat_a + not_cat_b
            h_word = (-( (cat_a + not_cat_a)/all_words * log ( (cat_a + not_cat_a)/all_words )/log (2) + (cat_b + not_cat_b)/all_words * log((cat_b + not_cat_b)/all_words )/log(2) ))
            h_word_pos = (- (cat_a/(cat_a + cat_b) * log (cat_a/(cat_a + cat_b)) / log(2) + cat_b/(cat_a + cat_b) * log (cat_b/(cat_a + cat_b))/ log(2) ) )
            h_word_neg = (- (not_cat_a/(not_cat_a + not_cat_b) * log (not_cat_a/(not_cat_a + not_cat_b))/ log(2) + not_cat_b/(not_cat_a + not_cat_b) * log (not_cat_b/(not_cat_a + not_cat_b))/ log(2) ) )
           
            h_word_over = (cat_a + cat_b)/all_words * h_word_pos + (not_cat_a + not_cat_b)/all_words * h_word_neg

            ig_cat = h_word - h_word_over
            ig[word] = ig_cat
            
    result_ig = { }
    if list_size != None:
        size = list_size
    else: 
        size = len(ig)    
    if specific_word != None:
       
        result_ig[specific_word] = ig[specific_word]
    
    else:
        while len(result_ig) < size:
            print ( len(result_ig))
            print (result_ig)
            max_key = max(ig, key=ig.get)
            max_value = ig[max_key]
            result_ig [max_key] = max_value
            del ig [max_key]
    result_ig_df = pd.Series(result_ig)
    if visualize == True:
        positions = arange(list_size) + .5 
        figure()
        barh(positions, list(result_ig.values()), align='center')
        yticks(positions, list(result_ig.keys()))
        xlabel('Weight')
        title('Strongest terms for categories' )
        show()  
        
    return result_ig_df


def chi_square (document_collection_category1, document_collection_category2, specific_word = None, list_size = 10, visualize = False):
    """This method selects relevant features from two document collections based on the chi square test.
        Args:
            document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
            specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size (int): (default 10) number of features to be returned
            visualize (bool): (default False) if True it represents the features graphically
        Returns:
            pandas.core.series.Series: most relevant features and their chi square test values             
    """
    cat1 = document_transformer(document_collection_category1)
    cat2 = document_transformer(document_collection_category2)

    documents = []
    categories = []

    for doc in cat1:
        documents.append(doc)
        categories.append(0)
    
    for doc in cat2:
        documents.append(doc)
        categories.append(1)
    
    vectorizer = CountVectorizer()
    count_doc = vectorizer.fit_transform(documents)

    chi2score = chi2(count_doc, categories)[0]

    wscores = zip(vectorizer.get_feature_names(),chi2score)
    wchi2 = sorted(wscores,key=lambda x:x[1]) 

    chi= { }
    for date in wchi2:
        chi [date[0]] = date [1]

    if list_size != None:
        size = list_size
    else: 
        size = len(chi)    

    result_chi_square = {}
    if specific_word != None:
       
        result_chi_square[specific_word] = chi[specific_word]
    
    else:
       while len(result_chi_square) < size:
            max_key = max(chi, key=chi.get)
            max_value = chi[max_key]
            result_chi_square [max_key] = max_value
            del chi [max_key]
    result_df = pd.Series(result_chi_square)
    if visualize == True:
        
        positions = arange(list_size) + .5 
        figure()
        barh(positions, list(result_chi_square.values()), align='center')
        yticks(positions, list(result_chi_square.keys()))
        xlabel('Weight')
        title('Strongest terms for categories' )
        show()
     
    return result_df

def latent_semantic_analysis (document_collection_category1, document_collection_category2, list_size = 10 , visualize = False ):
    """This method selects relevant features from two document collections based on the singular-value decomposition.
        Args:
            document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
			list_size (int): (default 10) number of features to be returned
            visualize (bool): (default False) if True it represents the features graphically
        Returns:
            pandas.core.frame.DataFrame: most relevant features for each category and their relevance values             
    """
    cat1 = document_transformer(document_collection_category1)
    cat2 = document_transformer(document_collection_category2)

    documents = cat1 + cat2
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)

    documents_tfidf = vectorizer.fit_transform(documents)


    feat_names = vectorizer.get_feature_names()

    #singular value decomposition 
    lsa = TruncatedSVD(100)
    documents_lsa = lsa.fit_transform(documents_tfidf)

    values_list = []
    for category in range(0, 2):

        cat = lsa.components_[category]
    
        indeces = numpy.argsort(cat).tolist()
    
        indeces.reverse()    
        word = [feat_names[weightIndex] for weightIndex in indeces[0:list_size]]    
        value = [cat[weightIndex] for weightIndex in indeces[0:list_size]]   
        
        if visualize == True:
            word.reverse()
            value.reverse()
            positions = arange(list_size) + .5 
            figure(category)
            barh(positions, value, align='center')
            yticks(positions, word)
            xlabel('Weight')
            title('Strongest terms for category %d' % (category))
            show()

        lsa_values = { }

        for i in range (list_size):
            lsa_values [word[i]] = value[i]
            
        values_list.append(lsa_values)
        values_list_df = pd.DataFrame(values_list)
    return values_list_df.T



