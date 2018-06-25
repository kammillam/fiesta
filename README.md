# fiesta
## Documentation 

### fiesta.bag_of_words    

* _fiesta.bag_of_words.bag_of_words (document = None, file = None, indexOfDocument = None)_
	
		__Parameters:__ document():
				file():
				indexOfDocument(int): 

		__Returns:__ vector representation for all documents or vector representation for selected document
		__Return type:__ numpy.ndarray

* _fiesta.bag_of_words.bag_of_words_vocabulary (document = None, file = None)_
	
		__Parameters:__ document():
				file():

		__Returns:__ an assignment of terms to characteristic indexes in form [word : index] 
		__Return type:__ dict

#### fiesta.tfidf    

* _fiesta.tfidf.tf (file = None, document = None, indexOfDocument = None,_ 
						_scaled = False)_
		__Parameters:__ file():
				document():
				indexOfDocument():
				scaled(bool):

		__Returns:__ document-term array with the number of term in each document 
		__Return type:__ numpy.ndarray

* _fiesta.tfidf.idf (file = None, document = None, smooth = True)_
		__Parameters:__ file():
				document():
				smooth(bool):

		__Returns:__ list with idf weight for euch term
		__Return type:__ numpy.ndarray


* _tiesta.tfidf.tfidf (file = None, document = None, smooth = True,_
 _indexOfDocument =None)_    
		__Parameters:__ file():
				document():
				smooth(bool):
				indexOfDocument(int):
		__Returns:__ document-term array with tfidf weight for term in each document 
		__Return type:__ numpy.ndarray


#### fiesta.similarity_coefficients   

* _fiesta.similarity_coefficients.jaccard_similarity_coefficient ( document1, document2, normalize =True)_
		
		__Parameters:__ document1(str or list):
				document2(str or list):
				normalize(bool):
		__Returns:__ Jaccard similarity coefficient
		__Return type:__ numpy.float64

* _fiesta.similarity_coefficients.cosine_coefficient(document1, document2)_
		
		__Parameters:__ document1(str or list):
				document2(str or list):

		__Returns:__ cosine similarity coefficient
		__Return type:__ numpy.float64
