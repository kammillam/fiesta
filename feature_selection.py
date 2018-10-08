from fiesta.normalizing import pos_tagging, lemmatizer, stop_words
from nltk import word_tokenize
from math import log
from fiesta.bag_of_words import document_transformer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy
from sklearn.decomposition import TruncatedSVD

def information_gain(categorie1, categorie2, min_ig = None, specific_word = None, list_size = 10):

    cat1 = document_transformer(categorie1)
    cat2 = document_transformer(categorie2)

    individual_words = [] 

    for doc in cat1 + cat2:
        tokens = word_tokenize(doc)
        for token in tokens:
            if token not in individual_words:
                individual_words.append(token)

    ig = { }

    for word in individual_words:
        a = 0 #kommt in der Kategorie 
        b = 0 #kommt in der anderen Kategorie 
        c = 0 # kommt nicht in der Kategorie 
        d = 0 # kommt nicht in der anderer Kategorie 
        for doc in cat1:
         
            if word in doc:
                a= a + 1
            else:
                c  = c + 1

    # neg ist falsche Kategorie, Berechnung von B und D: B falls vorkommt und D falls nicht vorkommt 

        for doc in cat2:
            if word in doc:
                
                b = b +1
            else:
                d = d + 1
    
        if a*b*c*d!=0:
            n = a + b + c + d
            h_word = (-( (a+c)/n * log ( (a+c)/n ,2) + (b+d)/n * log((b+d)/n ,2) ))
            h_word_pos = (- (a/(a+b) * log (a/(a+b) , 2) + b/(a+b) * log (b/(a+b) , 2) ) )
            h_word_neg = (- (c/(c+d) * log (c/(c+d) , 2) + d/(c+d) * log (d/(c+d) , 2) ) )

            h_word_over = (a+b)/n * h_word_pos + (c+d)/n * h_word_neg

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

    return result_ig


def chi_square (categorie1, categorie2, specific_word = None, list_size = 10):

    cat1 = document_transformer(categorie1)
    cat2 = document_transformer(categorie2)

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

    return result_chi_square

def latent_semantic_analysis (categorie1, categorie2, list_size = 10):
    cat1 = document_transformer(categorie1)
    cat2 = document_transformer(categorie2)

    documents = cat1 + cat2
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english",   use_idf=True)

    documents_tfidf = vectorizer.fit_transform(documents)


    feat_names = vectorizer.get_feature_names()


    lsa = TruncatedSVD(100)

    documents_lsa = lsa.fit_transform(documents_tfidf)

    values_list = []
    for categorie in range(0, 2):

        cat = lsa.components_[categorie]
    
        indeces = numpy.argsort(cat).tolist()
    
        indeces.reverse()    
        word = [feat_names[weightIndex] for weightIndex in indeces[0:list_size]]    
        value = [cat[weightIndex] for weightIndex in indeces[0:list_size]]   
        lsa_values = { }

        for i in range (list_size):
            lsa_values [word[i]] = value[i]
        values_list.append(lsa_values)
    return values_list



