from sklearn.feature_extraction.text import TfidfTransformer
from fiesta.bag_of_words import bag_of_words


def tf (document, indexOfDocument = None, scaled = False): 
    """This is docstring"""   
    termFrequency = bag_of_words ( document, indexOfDocument=indexOfDocument)    
    if scaled == True:
        termFrequency = scale (termFrequency)
         
    return termFrequency

def idf ( document , smooth = True):
    """This is docstring"""   
    termFrequency = tf (document )
    transformer = TfidfTransformer(smooth_idf=smooth)
    
    transformer.fit_transform(termFrequency).toarray()
   
    return transformer.idf_
    
def tfidf ( document, smooth = True, indexOfDocument = None):
    """This is docstring"""   
    termFrequency = tf ( document)
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
    """This is docstring"""   
    if len(termFrequency) == 1:
        scaledTermFrequency = termFrequency / max (termFrequency)
        return scaledTermFrequency
    else:
        scaledTermFrequency = []
        for term in termFrequency:
            scaledTerm = term / max (term)
            scaledTermFrequency.append(scaledTerm)
        return scaledTermFrequency