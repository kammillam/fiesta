from sklearn.feature_extraction.text import TfidfTransformer
from fiesta.bag_of_words import bag_of_words


def tf (file = None, document = None, indexOfDocument = None, scaled = False): 
    
    termFrequency = bag_of_words (file = file, document=document, indexOfDocument=indexOfDocument)    
    if scaled == True:
        termFrequency = scale (termFrequency)
         
    return termFrequency

def idf (file = None, document = None, smooth = True):
    termFrequency = tf (file = file, document = document)
    transformer = TfidfTransformer(smooth_idf=smooth)
    
    transformer.fit_transform(termFrequency).toarray()
   
    return transformer.idf_

    
def tfidf (file = None, document = None, smooth = True, indexOfDocument = None):
    termFrequency = tf (file = file, document = document)
    if smooth == False:
        transformer = TfidfTransformer(smooth_idf=False)
    
    else:
        transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(termFrequency).toarray()
    
    if indexOfDocument != None:
   
        return tfidf[indexOfDocument]
   
    else: 
        return tfidf
    
def scale ( termFrequency  ) :

    if len(termFrequency) == 1:
        scaledTermFrequency = termFrequency / max (termFrequency)
        return scaledTermFrequency
    else:
        scaledTermFrequency = []
        for term in termFrequency:
            scaledTerm = term / max (term)
            scaledTermFrequency.append(scaledTerm)
        return scaledTermFrequency
    