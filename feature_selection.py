from nltk import word_tokenize
from math import log
from fiesta.bag_of_words import document_transformer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pylab import *
import numpy
import pandas as pd
from fiesta.bag_of_words import bag_of_words
from fiesta.tfidf import tfidf

def term_frequency_selection(category1, category2, list_size = 10):

    cat1 = document_transformer(category1)
    cat2 = document_transformer(category2)
    full_document = cat1+cat2

    tf = bag_of_words(full_document)

    tf_sum = tf.sum()

    tf_sort=tf_sum.sort_values(ascending=False)
    return tf_sort[:list_size]

def tfidf_selection (category1, category2, list_size = 10):
    cat1 = document_transformer(category1)
    cat2 = document_transformer(category2)
    full_document = cat1+cat2

    tf_idf = tfidf(full_document)

    tf_idf_sum = tf_idf.sum()

    tf_idf_sort=tf_idf_sum.sort_values(ascending=False)
    return tf_idf_sort[:list_size]


def information_gain(category1, category2, min_ig = None, specific_word = None, list_size = 10, visualize = False):
    """This is docstring"""   

    cat1 = document_transformer(category1)
    cat2 = document_transformer(category2)

    individual_words = [] 
    for doc in cat1 + cat2:
        tokens = word_tokenize(doc)
        for token in tokens:
            if len(token) > 2:
                if token not in individual_words:
                    individual_words.append(token)
    ig = { }

    for word in individual_words:
        cat_a = 0 #kommt in der Kategorie 
        cat_b = 0 #kommt in der anderen Kategorie 
        not_cat_a = 0 # kommt nicht in der Kategorie   
        not_cat_b = 0 # kommt nicht in der anderer Kategorie   
        for doc in cat1:
         
            if word in doc:
                cat_a = cat_a + 1
            else:
                not_cat_a  = not_cat_a + 1

    # neg ist falsche Kategorie, Berechnung von B und D: B falls vorkommt und D falls nicht vorkommt 
        for doc in cat2:
            if word in doc:
                
                cat_b = cat_b + 1
            else:
                not_cat_b = not_cat_b + 1
        if cat_a*cat_b*not_cat_a*not_cat_b!=0:
            all_words = cat_a + cat_b + not_cat_a + not_cat_b
            h_word = (-( (cat_a + not_cat_a)/all_words * log ( (cat_a + not_cat_a)/all_words )/log (2) + (cat_b + not_cat_b)/all_words * log((cat_b + not_cat_b)/all_words )/log(2) ))
            h_word_pos = (- (cat_a/(cat_a + cat_b) * log (cat_a/(cat_a + cat_b)) / log(2) + cat_b/(cat_a + cat_b) * log (cat_b/(cat_a + cat_b))/ log(2) ) )
            h_word_neg = (- (not_cat_a/(not_cat_a + not_cat_b) * log (not_cat_a/(not_cat_a + not_cat_b))/ log(2) + not_cat_b/(not_cat_a + not_cat_b) * log (not_cat_b/(not_cat_a + not_cat_b))/ log(2) ) )
           
            h_word_over = (cat_a + cat_b)/all_words * h_word_pos + (not_cat_a + not_cat_b)/all_words * h_word_neg

            ig_cat = h_word - h_word_over
            if min_ig == None:
                ig[word] = ig_cat
            elif ig_cat > min_ig:
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
            max_key = max(ig, key=ig.get)
            max_value = ig[max_key]
            result_ig [max_key] = max_value
            del ig [max_key]
    result_ig_df = pd.Series(result_ig)
    if visualize == True:
        positions = arange(list_size) + .5 # the bar centers on the y axis
        figure()
        barh(positions, list(result_ig.values()), align='center')
        yticks(positions, list(result_ig.keys()))
        xlabel('Weight')
        title('Strongest terms for categories' )
        show()  
        
    return result_ig_df


def chi_square (category1, category2, specific_word = None, list_size = 10, visualize = False):
    """This is docstring"""   

    cat1 = document_transformer(category1)
    cat2 = document_transformer(category2)

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
        
        positions = arange(list_size) + .5 # the bar centers on the y axis
        figure()
        barh(positions, list(result_chi_square.values()), align='center')
        yticks(positions, list(result_chi_square.keys()))
        xlabel('Weight')
        title('Strongest terms for categories' )
        show()
     
    return result_df

def latent_semantic_analysis (category1, category2, list_size = 10 , visualize = False ):
    """This is docstring"""   

    cat1 = document_transformer(category1)
    cat2 = document_transformer(category2)

    documents = cat1 + cat2
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english",   use_idf=True)

    documents_tfidf = vectorizer.fit_transform(documents)


    feat_names = vectorizer.get_feature_names()


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
            positions = arange(list_size) + .5 # the bar centers on the y axis
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



