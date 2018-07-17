from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile

def stop_words (document, english = False, user_definded_stop_word_list = None):
    """This is docstring"""   
    
    if user_definded_stop_word_list != None:
        stop_word_list = user_definded_stop_word_list
    elif english == True:
        stop_word_list = set (stopwords.words ('english'))    
    else:
        stop_word_list = set (stopwords.words ('german'))

    vectorizer = CountVectorizer()
    tokens = vectorizer.build_analyzer()

    if type(document) == list:

        documents_without_stopwords = []

        for document_part in document:
            document_tokens = tokens(document_part)
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
            document_tokens = tokens(document_part)
            document_without_stopwords = ""
            for word in document_tokens:
                if word not in stop_word_list :
                    document_without_stopwords = document_without_stopwords + " " + word
            documents_without_stopwords.append(document_without_stopwords.strip())

        return documents_without_stopwords    
        
    

    elif type(document) == str:
        document_tokens = tokens(document)
        document_without_stopwords = ""
        for word in document_tokens:
            if word not in stop_word_list :
                document_without_stopwords = document_without_stopwords + " " + word
        return document_without_stopwords.strip()

    