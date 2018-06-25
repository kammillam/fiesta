# fiesta
Documentation 

fiesta.BagOfWords
fiesta.bag_of_words.bag_of_words (document = None, file = None, indexOfDocument = None) 
	
		Parameters: document():
				file():
				indexOfDocument(int): 

		Returns: vector representation for all documents or vector representation for selected document
		Return type: numpy.ndarray

fiesta.bag_of_words.bag_of_words_vocabulary (document = None, file = None):
	
		Parameters: document():
				file():

		Returns: an assignment of terms to characteristic indexes in form [word : index] 
		Return type: dict

fiesta.tfidf
fiesta.tfidf.tf (file = None, document = None, indexOfDocument = None, 
						scaled = False)
		Parameters: file():
				document():
				indexOfDocument():
				scaled(bool):

		Returns: document-term array with the number of term in each document 
		Return type: numpy.ndarray

fiesta.tfidf.idf (file = None, document = None, smooth = True)
		Parameters: file():
				document():
				smooth(bool):

		Returns: list with idf weight for euch term
		Return type: numpy.ndarray


tiesta.tfidf.tfidf (file = None, document = None, smooth = True,
 indexOfDocument =None)
		Parameters: file():
				document():
				smooth(bool):
				indexOfDocument(int):
		Returns: document-term array with tfidf weight for term in each document 
		Return type: numpy.ndarray


fiesta.similarity_coefficients

	fiesta.similarity_coefficients.jaccard_similarity_coefficient ( document1,           document2, normalize =True)
		
		Parameters: document1(str or list):
				document2(str or list):
				normalize(bool):
		Returns: Jaccard similarity coefficient
		Return type: numpy.float64

		fiesta.similarity_coefficients.cosine_coefficient(document1, document2)
		
		Parameters: document1(str or list):
				document2(str or list):

		Returns: cosine similarity coefficient
		Return type: numpy.float64
