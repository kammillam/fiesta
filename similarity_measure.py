from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from fiesta.bag_of_words import bag_of_words
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from os.path import isfile



def simple_matching_coefficient (document1, document2):
    """This is docstring"""   

    document1 = document_transformer(document1) # aus einem file wird str gemacht 
    document2 = document_transformer(document2)
    total_document = document_concatenation(document1,document2) #aus zwei Dokumenten wird gemeinsamer BoW gemacht 

    i = 0  # i ist Index von Word im Dokumentenvektor
    common_part = 0 # ist Zähler im Formel; das was gemeinsam bei beiden Dokumenten ist 
    while i < len(total_document[0]): 
        if total_document[0][i] != 0 and total_document[1][i] != 0: #falls beide Dokumente das Wort haben 
            common_part += 1
        i += 1
    simple_matching_coef = common_part / len (total_document[0])    # gemeinsames Teil durch Anzahl aller Atributten 
    return simple_matching_coef



def jaccard_similarity_coefficient ( document1, document2, normalize =True):
    """This is docstring"""   
   
    document1 = document_transformer(document1) # aus einem file wird str gemacht 
    document2 = document_transformer(document2)
    
    document = document_concatenation( document1, document2) #aus zwei Dokumenten wird gemeinsamer BoW gemacht 
    jaccard_similarity = jaccard_similarity_score(document[0], document[1], normalize=normalize)
    
    return jaccard_similarity 

def cosine_coefficient(document1, document2):
    """This is docstring"""   
    document1 = document_transformer(document1) # aus einem file wird str gemacht 
    document2 = document_transformer(document2)
    
    document = document_concatenation(document1, document2) #bag_of_words für zwei input-strings 
    cosine_similarity_score = cosine_similarity([document[0],],[document[1],])
    
    return cosine_similarity_score[0][0]

def dice_coefficient(document1, document2):
    """This is docstring"""   
    document1 = document_transformer(document1) # aus einem file wird str gemacht 
    document2 = document_transformer(document2)

    dice_coef = 0

    if document1 == document2: 
        return 1

    if not len(document1) or not len(document2):
        return 0

    total_document = document_concatenation(document1, document2) #aus zwei Dokumenten wird gemeinsamer BoW gemacht 

    common_part = 0 

    i = 0
    while i <  len(total_document[0]):
        common_part += min (total_document[0][i], total_document[1][i])
        i += 1
    
    len_document1 = len(document1.split())
    len_document2 = len(document2.split())

    dice_coef = 2*common_part /(len_document1 + len_document2)
    return dice_coef

def overlap_coefficient(document1, document2):
    """This is docstring""" 

    document1 = document_transformer(document1) # aus einem file wird str gemacht 
    document2 = document_transformer(document2)

    overlap_coef  = OverlapCoefficient().get_raw_score(document1.split(), document2.split())
    return overlap_coef

def summary_similarity_measures (document1, document2): #string mit allen Koeffizienten 
    """This is docstring"""   
    simple_matching_coef = simple_matching_coefficient(document1,document2)
    jaccard_similarity_coef = jaccard_similarity_coefficient (document1, document2)
    cosine_coef = cosine_coefficient(document1, document2)
    dice_coef = dice_coefficient(document1,document2)
    overlap_coef = overlap_coefficient (document1,document2)

    summary = "Simple Matching Coefficient: " + str(simple_matching_coef) + "\nJaccard similarity coefficient:" + str(jaccard_similarity_coef) + "\nCosine similarity coefficient: " + str(cosine_coef) + "\nDice’s coefficient: " + str(dice_coef) + "\nOverlap coefficient: " + str(overlap_coef)

    return summary

def document_concatenation (document1, document2): #aus zwei Dokumenten wird gemeinsamer BoW gemacht 
    """This is docstring"""   
    document = []
    document.append(document1)
    document.append(document2)
    documents_bag_of_words = bag_of_words(document=document)
    return documents_bag_of_words    

def document_transformer (document): # aus einem file wird str gemacht 
    """This is docstring"""   
    if type(document) == list:
        first_docuemnt = document[0]
        return first_docuemnt # nur erster Dokument in der Liste wird für den Vergleich genommen 
    elif isfile(document):    
        document = open(document, "r")
        for line in document:
            file_document = line
        document.close()  
        return file_document # nur erster Dokument im File wird für den Vergleich genommen
    elif type(document) == str:
        return document