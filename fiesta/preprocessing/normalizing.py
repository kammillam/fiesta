"""
This module preprocesses a text before feature extraction.
"""
from nltk.corpus import stopwords
from os.path import isfile
from nltk import pos_tag, word_tokenize
from fiesta.transformers.document_transformer import document_transformer
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
import pickle
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from fiesta.external_packages.germalemma.germalemma import GermaLemma

def stop_words (document_collection, language = "en", user_definded_stop_word_list = None, punctuation = True):
    """This method removes stop words which do not contribute to any future operations.
        Args:
            document_collection (str, list or file directory): document collection where stop words are to be removed 
 			language (str): (default „en“)  if "en": a pre-defened set of english stop words will be used
                                            if "de": a pre-defened set of german stop words will be used
			user_defined_stop_word_list (list): (default None) a user defined set of stop words will be used
			punctuation (bool): (default True) special characters will be removed
        Returns: 
            str: String without stop words
            list: list of documents without stop words
    """
    stop_word_list = []
    if user_definded_stop_word_list != None:
        stop_word_list = user_definded_stop_word_list
    elif language == "en":
        stop_word_list = set (stopwords.words ('english'))     
    elif language == "de":
        stop_word_list = set (stopwords.words ('german'))     

    full_document = document_transformer(document_collection)   
    documents_without_stopwords = []

    for document_part in full_document:
        document_tokens = word_tokenize(document_part)
        document_without_stopwords = ""
        for word in document_tokens:
            if word not in stop_word_list :
                if punctuation:
                    if word.isalpha():
                        document_without_stopwords = document_without_stopwords + " " + word
                else: 
                    document_without_stopwords = document_without_stopwords + " " + word

        documents_without_stopwords.append(document_without_stopwords.strip())

    return documents_without_stopwords    

def pos_tagging(document_collection, language = "en"):
    """This method determines the part of speech of each word.
        Args:
            document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 
        Returns:   
            list: list of assigned part-of-speech-tags for each word in form (term, part-of-speech-tag)
    """
    transformed_document = document_transformer(document_collection)
    documents_pos_tag = []

    if language == "en":
        if type(transformed_document) == str:
            document_pos_tag = pos_tag(word_tokenize(transformed_document))
            return document_pos_tag

        else:
            for document_part in transformed_document:
                documents_pos_tag.append(pos_tag(word_tokenize(document_part)))
            return documents_pos_tag  

    elif language == "de": 
        with open('nltk_german_classifier_data.pickle', 'rb') as f:
            tagger = pickle.load(f)            
        
        if type(transformed_document) == str:
            document_pos_tag = tagger.tag(transformed_document.split())
            return document_pos_tag 
        else:
            for document_part in transformed_document:
                documents_pos_tag.append(tagger.tag(document_part.split()))
            return documents_pos_tag 

def lemmatizer (document_collection, language = "en"):
    """This method reduces each word to their word stem or dictionary form.
        Args:
            document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 	
        Returns:
            str: string with lemmatized words
            list: list of strings with lemmatized words
    """
    transformed_document = document_transformer (document_collection) #aus einem File/List/String wird List mit Strings gemacht 
    total_lemmatized_document = [] #neuer List mit lemmatizierten Dokumenten
    
    if language == "en": #für englisch
        wnl = WordNetLemmatizer()
        for document in transformed_document: # jedes Dokument ...
            document_tokens = document.split() # ...auf Wörter verteilen 
            lemmatized_document_part = "" #neuen lemmatizierter Dokument 
            for word in document_tokens: 
                pos = pos_tag(word_tokenize(word))[0][1]
                if  pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word, pos = "v")
                elif pos in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word, pos = "a")         
                else:         
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word)             
            total_lemmatized_document.append(lemmatized_document_part.strip())
        return total_lemmatized_document

    elif language == "de":   #für deutsch
        lem = GermaLemma ()
        for document in transformed_document: 
            document_tokens = document.split()
            lemmatized_document_part = ""
            for word in document_tokens:
                pos = pos_tagging(word, language="de")[0][0][1]

                if pos  in ['VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VAFIN', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "V") 
                elif pos in ['ADJA', 'ADJD', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPOSAT', 'PWAT']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "ADJ")         
                elif pos in ['ADV', 'PAV', 'PAVREL', 'PTKA', 'PWAV', 'PWAVREL']:         
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "ADV")
                elif pos in ['NA', 'NE', 'NN']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "N")
                else:
                    lemmatized_document_part = lemmatized_document_part + " " + word

            total_lemmatized_document.append(lemmatized_document_part.strip())

        return total_lemmatized_document
            
def stemming (document_collection, language = "en"):
    """This method reduces each word to their word stem or root form.
        Args:
            document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 	
        Returns:
            str: string with stemmed words
            list: list of strings with stemmed words
    """
    if language == "en":
        stemmer = SnowballStemmer("english")
    elif language == "de": 
        stemmer = SnowballStemmer("german")

    transformed_document = document_transformer(document_collection)
    stemmed_documents = []

    for document_part in transformed_document:
        document_tokens = word_tokenize(document_part)
        stemmed_document = ""
        for word in document_tokens:
            word = stemmer.stem(word)
            stemmed_document = stemmed_document + " " + word
        stemmed_documents.append(stemmed_document.strip())

    return stemmed_documents

  