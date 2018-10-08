from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile
from nltk import pos_tag, word_tokenize
from fiesta.bag_of_words import document_transformer
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
import pickle
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from fiesta.germalemma import GermaLemma

def stop_words (document, english = False, user_definded_stop_word_list = None):
    """This is docstring"""   

    if user_definded_stop_word_list != None:
        stop_word_list = user_definded_stop_word_list
    elif english == True:
        stop_word_list = set (stopwords.words ('english'))     #englische vordefinierte Stopwörter 
    else:
        stop_word_list = set (stopwords.words ('german'))     #deutsche vordefinierte Stopwörter

    if type(document) == list:

        documents_without_stopwords = []

        for document_part in document:
            document_tokens = word_tokenize(document_part)
            document_without_stopwords = ""
            for word in document_tokens:
                if word not in stop_word_list :
                    document_without_stopwords = document_without_stopwords + " " + word
            documents_without_stopwords.append(document_without_stopwords.strip())

        return documents_without_stopwords    

    elif isfile(document):
        full_document = []    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  
        documents_without_stopwords = []

        for document_part in full_document:
            document_tokens = word_tokenize(document_part)
            document_without_stopwords = ""
            for word in document_tokens:
                if word not in stop_word_list :
                    document_without_stopwords = document_without_stopwords + " " + word
            documents_without_stopwords.append(document_without_stopwords.strip())

        return documents_without_stopwords    

    elif type(document) == str:

        document_tokens = word_tokenize(document)
        document_without_stopwords = ""
        for word in document_tokens:
            if word not in stop_word_list :
                document_without_stopwords = document_without_stopwords + " " + word
                
        return document_without_stopwords.strip()


def pos_tagging(document, english = False):

    transformed_document = document_transformer(document)
    documents_pos_tag = []

    if english:
        if type(transformed_document) == str:
            document_pos_tag = pos_tag(word_tokenize(transformed_document))
            return document_pos_tag

        else:
            for document_part in transformed_document:
                documents_pos_tag.append(pos_tag(word_tokenize(document_part)))
            return documents_pos_tag   

    else: 
        with open('nltk_german_classifier_data.pickle', 'rb') as f:
            tagger = pickle.load(f)            
        
        if type(transformed_document) == str:
            document_pos_tag = tagger.tag(transformed_document.split())
            return document_pos_tag
        else:
            for document_part in transformed_document:
                documents_pos_tag.append(tagger.tag(document_part.split()))
            return documents_pos_tag   

def lemmatizer (document, english = False):
    transformed_document = lemmatizer_document_transformer (document) #aus einem File/List/String wird List mit Strings gemacht 
    total_lemmatized_document = [] #neuer List mit lemmatizierten Dokumenten
    
    if english: #für englisch
        wnl = WordNetLemmatizer()
        for document in transformed_document: # jedes Dokument ...
            document_tokens = document.split() # ...auf Wörter verteilen 
            lemmatized_document_part = "" #neuen lemmatizierter Dokument 
            for word in document_tokens: 
                
                if pos_tag(word_tokenize(word))[0][1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word, pos = "v") 
                elif pos_tag(word_tokenize(word))[0][1][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word, pos = "a")         
                else:         
                    lemmatized_document_part = lemmatized_document_part + " " + wnl.lemmatize(word)             
            total_lemmatized_document.append(lemmatized_document_part.strip())
        return total_lemmatized_document

    else:   #für deutsch
        lem = GermaLemma ()
        for document in transformed_document: 
            document_tokens = document.split()
            lemmatized_document_part = ""
            for word in document_tokens:
               
                if pos_tagging(word)[0][0][1] in ['VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VAFIN', 'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "V") 
                elif pos_tagging(word)[0][0][1] in ['ADJA', 'ADJD', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPOSAT', 'PWAT']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "ADJ")         
                elif pos_tagging(word)[0][0][1] in ['ADV', 'PAV', 'PAVREL', 'PTKA', 'PWAV', 'PWAVREL']:         
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "ADV")
                elif pos_tagging(word)[0][0][1] in ['NA', 'NE', 'NN']:
                    lemmatized_document_part = lemmatized_document_part + " " + lem.find_lemma(word, "N")
                else:
                    lemmatized_document_part = lemmatized_document_part + " " + word

            total_lemmatized_document.append(lemmatized_document_part.strip())

        return total_lemmatized_document
            
def stemming (document, english = False):
    """This is docstring"""   

    if english == True:
        stemmer = SnowballStemmer("english")
    else: 
        stemmer = SnowballStemmer("german")

    if type(document) == list:

        stemmed_documents = []

        for document_part in document:
            document_tokens = word_tokenize(document_part)
            stemmed_document = ""
            for word in document_tokens:
                word = stemmer.stem(word)
                stemmed_document = stemmed_document + " " + word
            stemmed_documents.append(stemmed_document.strip())

        return stemmed_documents

    
    elif isfile(document) :
        full_document = []    
        document = open(document, "r")
        for line in document:
            full_document.append(line)
        document.close()  
        
        stemmed_documents = []
        for document_part in full_document:
            document_tokens = word_tokenize(document_part)
            stemmed_document = ""
            for word in document_tokens:
                word = stemmer.stem(word)
                stemmed_document = stemmed_document + " " + word
            stemmed_documents.append(stemmed_document.strip())

        return stemmed_documents


    elif type(document) == str:
        document_tokens = word_tokenize(document)
        stemmed_document = ""
        for word in document_tokens:
            word = stemmer.stem(word)
            stemmed_document = stemmed_document + " " + word
        return stemmed_document.strip()

    elif type(document) == list:

        stemmed_documents = []

        for document_part in document:
            document_tokens = word_tokenize(document_part)
            stemmed_document = ""
            for word in document_tokens:
                word = stemmer.stem(word)
                stemmed_document = stemmed_document + " " + word
            stemmed_documents.append(stemmed_document.strip())

        return stemmed_documents



def lemmatizer_document_transformer (document): # aus einem file wird str gemacht 
    """This is docstring"""   
    if type (document) == list:
        return document 
    elif isfile(document):   
        file_document = [] 
        document = open(document, "r")
        for line in document:
            file_document.append(line)
        document.close()  
        return file_document
    elif type(document) == str:
        document_as_list = []
        document_as_list.append(document)
        return document_as_list # ein dokument in der Liste 