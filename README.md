# fiesta
## Documentation 

### fiesta.bag_of_words    

* _fiesta.bag_of_words.bag_of_words (document, indexOfDocument = None)_
	
		Parameters: document(str, list of strings or file directory): 
				indexOfDocument(int):

		Returns: vector representation for all documents or vector representation for selected document
		Return type: numpy.ndarray

* _fiesta.bag_of_words.bag_of_words_vocabulary (document)_
	
		Parameters: document(str, list of strings or file directory):
				
		Returns: an assignment of terms to characteristic indexes in form [word : index] 
		Return type: dict

* _fiesta.bag_of_words.words_counting (document)_
	
		Parameters: document(str, list of strings or file directory):
				
		Returns: an assignment of terms to Anzahl von terms   in form [word : counter] 
		Return type: dict

* _fiesta.bag_of_words.document_transformer(document)_
	
		Parameters: document(str, list of strings or file directory)
				
		Returns:
		Return type: list



### fiesta.tfidf    

* _fiesta.tfidf.tf (document, indexOfDocument = None,  scaled = False)_
		
		Parameters: document (str, list or file directory):							 				indexOfDocument():
				scaled(bool):

		Returns: document-term array with the number of term in each document 
		Return type: numpy.ndarray

* _fiesta.tfidf.idf (document, smooth = True)_
		
		Parameters:	document(str, list or file directory):											smooth(bool):

		Returns: list with idf-weight for euch term
		Return type: numpy.ndarray


* _tiesta.tfidf.tfidf (document, smooth = True,  indexOfDocument =None)_

		Parameters: document (str, list or file directory):
				smooth(bool):
				indexOfDocument(int):
		Returns: document-term array with tfidf-weight for term in each document 
		Return type: numpy.ndarray

* _tiesta.tfidf.scale (termFrequency)_

		Parameters: (numpy.ndarray)

		Returns:
		Return type:



### fiesta.similarity_measure

* _fiesta.similarity_coefficients.simple_matching_coefficient( document1, document2)_
		
		Parameters: document1(str or list of strings or file directory ):
				document2(str or list of strings or file directory):
		
		Returns:
		Return type:

* _fiesta.similarity_coefficients.jaccard_similarity_coefficient ( document1, document2, normalize =True)_
		
		Parameters: document1(str or list of strings or file directory):
				document2(str or list of strings or file directory):
				normalize(bool):
				
		Returns: Jaccard similarity coefficient between document1 and document2
		Return type: numpy.float64

* _fiesta.similarity_coefficients.cosine_coefficient(document1, document2)_
		
		Parameters: document1(str or list of strings or file directory):
			document2(str or list of strings or file directory):

		Returns: cosine similarity coefficient between document1 and document2
		Return type: numpy.float64

* _fiesta.similarity_coefficients.dice_coefficient(document1, document2)_
		
		Parameters: document1(str or list of strings or file directory):
			document2(str or list of strings or file directory):

		Returns:
		Return type:

* _fiesta.similarity_coefficients.overlap_coefficient(document1, document2)_
		
		Parameters: document1(str or list of strings or file directory):
			document2(str or list of strings or file directory):

		Returns:
		Return type:

* _fiesta.similarity_coefficients.summary_similarity_measures(document1, document2)_
		
		Parameters: document1(str or list of strings or file directory):
			document2(str or list of strings or file directory):

		Returns:
		Return type:


* _fiesta.similarity_coefficients.document_concatenation(document1, document2)_
		
		Parameters: document1(str or list of strings or file directory):
			document2(str or list of strings or file directory):

		Returns:
		Return type:

* _fiesta.similarity_coefficients.document_transformer(document)_
		
		Parameters: document1(str or list of strings or file directory):
				
		Returns:
		Return type:

### fiesta.normalizing 


* _fiesta.normalizing.stop_words( document, english =False, user_definded_stop_word_list = None)_
		
		Parameters: document(str, list or file directory):
				english(bool):
				user_definded_stop_word_list:
		Returns: String or list of strings without stop words
		Return type: str or list 

* _fiesta.normalizing.pos_tagging( document, english = False)_
		
		Parameters: document(str, list or file directory):
				english(bool)
				
		Returns:
		Return type: str or list 


* _fiesta.normalizing.lemmatizer( document, english = False)_
		
		Parameters: document(str, list or file directory):
				english(bool)
				
		Returns: String or list of strings with stemmed words
		Return type: str or list 

* _fiesta.normalizing.stemming( document, english = False)_
		
		Parameters: document(str, list or file directory):
				english(bool)
				
		Returns: String or list of strings with stemmed words
		Return type: str or list 

* _fiesta.normalizing.lemmatizer_document_transformer( document)_
		
		Parameters: document(str, list or file directory):
							
		Returns: String or list of strings with stemmed words
		Return type: str or list 

### fiesta.ngrams

* _fiesta.ngrams.n_grams( document, n)_
		
		Parameters: document(str, list or file directory):
				n(int)
				
		Returns: list
		Return type: list 




