from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words ( document = None, file = None, indexOfDocument = None):
   
    vectorizer = CountVectorizer()

    full_document = []
    if file != None :    
        file = open(file, "r")
        for line in file:
            full_document.append(line)
        file.close()  
    
    if document != None:
        full_document += document
    
    termDocumentMatrix = vectorizer.fit_transform(full_document).toarray()

    if indexOfDocument != None :
        return termDocumentMatrix[indexOfDocument]
    else: 
        return termDocumentMatrix
    
def bag_of_words_vocabulary(document = None, file = None):

    vectorizer = CountVectorizer()
    full_document = []
    if file != None :    
        file = open(file, "r")
        for line in file:
            full_document.append(line)
        file.close()  
    
    if document != None:
        full_document += document
    
    vectorizer.fit_transform(full_document).toarray()
    vocabulary = vectorizer.vocabulary_ 

    index_vocabulary = {}
    vocabulary_sorted = sorted(vocabulary.keys())
    for word in vocabulary_sorted:
        index_vocabulary[word] = vocabulary.get(word)
    return index_vocabulary
    
        

     
