from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile

def stemming (document, english = False):
    """This is docstring"""   

    if english == True:
        stemmer = SnowballStemmer("english")
    else: 
        stemmer = SnowballStemmer("german")

    vectorizer = CountVectorizer()
    tokens = vectorizer.build_analyzer()

    if type(document) == list:

        stemmed_documents = []

        for document_part in document:
            document_tokens = tokens(document_part)
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
            document_tokens = tokens(document_part)
            stemmed_document = ""
            for word in document_tokens:
                word = stemmer.stem(word)
                stemmed_document = stemmed_document + " " + word
            stemmed_documents.append(stemmed_document.strip())

        return stemmed_documents


    elif type(document) == str:
        document_tokens = tokens(document)
        stemmed_document = ""
        for word in document_tokens:
            word = stemmer.stem(word)
            stemmed_document = stemmed_document + " " + word
        return stemmed_document.strip()

    elif type(document) == list:

        stemmed_documents = []

        for document_part in document:
            document_tokens = tokens(document_part)
            stemmed_document = ""
            for word in document_tokens:
                word = stemmer.stem(word)
                stemmed_document = stemmed_document + " " + word
            stemmed_documents.append(stemmed_document.strip())

        return stemmed_documents


