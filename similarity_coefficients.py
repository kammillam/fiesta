from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from fiesta.bag_of_words import bag_of_words

def simple_matching (document1, document2):

    if type(document1) == str and type(document2) == str:
        document = document_concatenation( document1, document2)
        document1 = document[0]
        document2 = document[1]
    
    simple_matching_score = 0    
    
    for   sample in document1:
        simple_matching_score += sample * document2[document1.index(sample)]
    

    return simple_matching_score 

def jaccard_similarity_coefficient ( document1, document2, normalize =True):
    
    if type(document1) == str and type(document2) == str:
        document = document_concatenation( document1, document2)
        jaccard_similarity = jaccard_similarity_score(document[0], document[1], normalize=normalize)

    else:
        jaccard_similarity = jaccard_similarity_score(document1, document2, normalize=normalize)
    
    return jaccard_similarity 
        
def cosine_coefficient(document1, document2):

    if type(document1) == str and type( document2) == str:
        document = document_concatenation(document1, document2)
        cosine_similarity_score = cosine_similarity([document[0],],[document[1],])
    else:
        cosine_similarity_score = cosine_similarity([document1,],[document2])

    return cosine_similarity_score[0][0]

def document_concatenation (document1, document2):
    document = []
    document.append(document1)
    document.append(document2)
    documents_bag_of_words = bag_of_words(document=document)
    return documents_bag_of_words    