# fiesta
## Documentation 

### fiesta.bag_of_words    

* _fiesta.bag_of_words.bag_of_words (document = None, file = None, indexOfDocument = None)_
	
		**Parameters:** document():
				file():
				indexOfDocument(int): 

		**Returns:** vector representation for all documents or vector representation for selected document
		**Return type:** numpy.ndarray

* _fiesta.bag_of_words.bag_of_words_vocabulary (document = None, file = None)_
	
		**Parameters:** document():
				file():

		**Returns:** an assignment of terms to characteristic indexes in form [word : index] 
		**Return type:** dict

### fiesta.tfidf    

* _fiesta.tfidf.tf (file = None, document = None, indexOfDocument = None,_ 
						_scaled = False)_   

		**Parameters:** file():
				document():
				indexOfDocument():
				scaled(bool):

		**Returns:** document-term array with the number of term in each document 
		**Return type:** numpy.ndarray

* _fiesta.tfidf.idf (file = None, document = None, smooth = True)_

		**Parameters:** file():
				document():
				smooth(bool):

		**Returns:** list with idf weight for euch term
		**Return type:** numpy.ndarray


* _tiesta.tfidf.tfidf (file = None, document = None, smooth = True,_
 _indexOfDocument =None)_    

		**Parameters:** file():
				document():
				smooth(bool):
				indexOfDocument(int):
		**Returns:** document-term array with tfidf weight for term in each document 
		**Return type:** numpy.ndarray


### fiesta.similarity_coefficients   

* _fiesta.similarity_coefficients.jaccard_similarity_coefficient ( document1, document2, normalize =True)_
		
		**Parameters:** document1(str or list):
				document2(str or list):
				normalize(bool):
		**Returns:** Jaccard similarity coefficient
		**Return type:** numpy.float64

* _fiesta.similarity_coefficients.cosine_coefficient(document1, document2)_
		
		**Parameters:** document1(str or list):
				document2(str or list):

		**Returns:** cosine similarity coefficient
		**Return type:** numpy.float64
