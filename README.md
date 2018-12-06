# fiesta
## Documentation 

### fiesta.automatic
#### fiesta.automatic.preparation 

		This module contains methods for automatic preprocessing of text, feature extraction and feature selection.

* _fiesta.automatic.preparation.extract_features (document_collection, language_of_documents = "en", preprocessing_only = False):
		
		This method preprocesses the documents and extracts features from the preprocessed text.

		Parameters
			document_collection (str, list or file directory): document collection to extract the features of
			language_of_documents (str): (default „en“) language of the document collection 
			preprocessing_only (bool): (default False) if True returns only preprocessed documents
		Returns: list of preprocessed documents or DataFrame with matrix of token counts, tf-idf representation of the document collection and preprocessed documents. 
		Return type: list or pandas.core.frame.DataFrame

* _fiesta.automatic.preparation.select_features (document_collection_cat1, document_collection_cat2 = None):
		
		This method select relevant features based on the feature_selection module.

		Parameters: 
			document_collection_cat1 (str, list or file directory): document collection of the first category
			document_collection_cat2 (str, list or file directory): (default None) document collection of the second category
		Returns: relevant features for two document collections selected with all five methods of the feature_selection module or relevant features for one document collection selected with two methods.
		Return type:  pandas.core.frame.Series  or pandas.core.frame.DataFrame

* _fiesta.automatic.preparation.extract_relevant_features (document_collection_cat1, document_collection_cat2 = None,  language_of_document_collection = ‚en‘):
		
		Firstly prepares this method the document collection, then it extracts features and at the end it selects relevant features.

		Parameters: 
			document_collection_cat1 (str, list or file directory): document collection of the first category	
			document_collection_cat2 (str, list or file directory): (default None) document collection of the second category
			language_of_document_collection (str): (default „en“) language of the document collection 

		Returns: relevant features for two preprocessed document collections selected with all five methods of the feature_selection module or relevant features for one preprocessed document collection selected with two methods.
		Return type: pandas.core.series.Series or pandas.DataFrame


### fiesta.preprocessing

#### fiesta.preprocessing.tokenization 

		This module divides a documents into individuals words or sequence of words
 		by splitting on the blank spaces.

* _fiesta.preprocessing.tokenization.tokenization (document_collection):
		
		This method divides a documents into individual words (strings) by splitting on the blank spaces.

		Parameters: 
			document_collection (str, list or file directory): document collection to be tokenized	
		Returns: list of divided documents into individual words
		Return type: list 


* _fiesta.preprocessing.tokenization.n_grams_tokenization (document_collection,  n)

		This method divides a documents into sequences of n words (strings) by splitting on the blank spaces.

		Parameters: 
			document_collection (str, list or file directory): document collection
			n (int): length of the sequence of words	
		Returns: list of divided documents into sequences of words
		Return type: list 

#### fiesta.preprocessing.normalizing 

		This module preprocesses a text before feature extraction.

* _fiesta.preprocessing.normalizing.stop_words (document_collection, language = "en", user_defined_stop_word_list = None, punctuation = True)

	This method removes stop words which do not contribute to any future 				operations.

		Parameters:
			document_collection (str, list or file directory): document collection where stop words are to be removed 
			language (str): (default „en“) if "en": a predefened set of english stop words will be used;  if "de": a predefened set of german stop words will be used
			user_defined_stop_word_list (list): (default None) a user defined set of stop words will be used
			punctuation (bool): (default True) special characters will be removed
		Returns: String or list of documents without stop words
		Return type: str or list 

* _fiesta.preprocessing.normalizing.pos_tagging ( document_collection, language = "en")
		
		This method determines the part of speech of each word.

		Parameters: 
			document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 				
		Returns:  list of assigned part-of-speech-tags for each word in form (term, part-of-speech-tag)
		Return type: list 

* _fiesta.preprocessing.normalizing.lemmatizer (document_collection, language = "en")
		
		This method reduces each word to their word stem or dictionary form.

		Parameters: 
			document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 	
		Returns: String or list of documents with lemmatized words
		Return type: str or list 

* _fiesta.preprocessing.normalizing.stemming (document_collection, language = "en")
		
		This method reduces each word to their word stem or root form.

		Parameters: 
			document_collection (str, list or file directory): document collection
			language (str): (default „en“) „en“ for english; „de“ for german: for which language the method is to be executed 		
		Returns: String or list of strings with stemmed words
		Return type: str or list 


### fiesta.feature_extraction

#### fiesta.feature_extraction.bag_of_words
This module enables a transformation of the document collection into vector space representation and the counting of occurrences of the respective word in the document collection.

* _fiesta.feature_extraction.bag_of_words.bag_of_words (document_collection, index_of_document = None, specific_word = None, array = False) 

	This method converts a collection of text documents to a matrix of token counts.

		Parameters: 
			document_collection (str, list of strings or file directory): document collection 
			index_of_document (int): (default None)  index of the document whose vector is to be returned
			specific_word (str): (default None) word whose vector is to be returned
			array (bool): (default False) True, if Bag of Words is needed for other method
		Returns: vector representation for all documents or vector representation for selected document or word
		Return type: pandas.core.frame.DataFrame  or pandas.core.series.Series or numpy.ndarray

* _fiesta.feature_extraction.bag_of_words.words_counting (document_collection, specific_word = None)
	
		This method  counts the number of words in the whole document collection.

		Parameters: 
			document_collection(str, list of strings or file directory): document collection 
			specific_word (str): (default None) word whose number is to be returned
		Returns: an assignment of terms to number of terms in all documents		Return type: pandas.core.series.Series

#### fiesta.feature_extraction.tfidf

	This module allows transformation of document collection into tf- and tfidf-		representation and calculation of idf-weights of each term in the document collection.

* _fiesta.feature_extraction.tfidf.term_frequency (document_collection, index_of_document = None, specific_word = None, scaled = False)
		
		This method calculates term frequency in each document of the document collection.

		Parameters: 
			document_collection (str, list of strings or file directory): document collection 
			index_of_document (int):  (default None) index of the document whose vector is to be returned
			specific_word (str): (default None) word whose vector is to be returned
			scaled (bool):  (default False)  True, if scaling relative to the frequency of words in the document is needed
		Returns: tf representation of the document collection or of selected document or word
		Return type: pandas.core.frame.DataFrame 
				 pandas.core.series.Series

* _fiesta.feature_extraction.tfidf.inverse_document_frequency (document_collection, smooth = True, specific_word = None)

		This method calculates idf-weights for each term in the document collection.

		Parameters:	
			document_collection (str, list or file directory): document collection	
			smooth (bool): (default True) add one to document frequencies and prevents zero divisions
			specific_word (str): (default None) word whose vector is to be returned
		Returns: idf-weights of each word in the document collection or of of selected word
		Return type: pandas.core.frame.DataFrame or pandas.core.series.Series

* _tiesta.feature_extraction.tfidf.tfidf (document_collection, smooth = True,  index_of_document =None)

		This method calculates tf-idf weights for each term in the document collection. 

		Parameters: 
			document_collection (str, list or file directory): document collection
			smooth (bool): (default True) add one to document frequencies and prevents zero divisions
			index_of_document (int): (default None) index of the document whose vector is to be returned
		Returns: tf-idf representation of the document collection
		Return type: pandas.core.frame.DataFrame 
				 pandas.core.series.Series

* _fiesta.feature_extraction.tfidf.scale  (termFrequency)

		This method scales tf-representation of the document collection relative to the frequency of words in the document.

		Parameters: 
			termFrequency (numpy.ndarray): tf representation of the document collection
		Returns: scaled term frequency representation relative to the frequency of words in the document
		Return type: list


### fiesta.feature_selection 

#### fiesta.feature_selection.selection
* _fiesta.feature_selection.selection.term_frequency_selection (document_collection_category1, document_collection_category2 = None,  specific_word = None,  list_size = 10)
		
		This method selects relevant features from one or two document collections based on the frequency of occurrence of the word.

		Parameters: 
			document_collection_category1 (str, list or file directory): document collection of the first category 
			document_collection_category2 (str, list or file directory): (default None) document collection of the second category
			specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size (int): (default 10) number of features to be returned
		Returns: most relevant features and their frequency in all documents
		Return type: pandas.core.series.Series

* _fiesta.feature_selection.selection.tfidf_selection (document_collection_category1, document_collection_category2 = None, specific_word = None,  list_size = 10)

		This method selects relevant features from one or two document collections based on the tf-idf value of the word.
		
		Parameters: 
			document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): (default None) document collection of the second category
			specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned 
			list_size (int): (default 10) number of features to be returned
		Returns: most relevant features and sum of their Tf-idf weights in all documents
		Return type: pandas.core.series.Series


* _fiesta.feature_selection.selection.information_gain (document_collection_category1, document_collection_category2, specific_word = None,  list_size = 10, visualize = False)
			
		This method selects relevant features from two document collections based on the information gain algorithm.

		Parameters: 
			document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
			specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned
			list_size (int): (default 10) number of features to be returned
			visualize (bool): (default False)   if True it represents the features graphically
		Returns: most relevant features and their their information gain values 
		Return type: pandas.core.series.Series 

* _fiesta.feature_selection.selection.chi_square (document_collection_category1, document_collection_category2, specific_word = None, list_size = 10, visualize = False)
		
		This method selects relevant features from two document collections based on the chi square test.

		Parameters: 
			document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
			specific_word (str): (default None) word whose relevance in the document collection(s) is to be returned
			list_size (int): (default 10) number of features to be returned
			visualize (bool): (default False)   if True it represents the features graphically
		Returns: most relevant features and their their chi square test values 
		Return type: pandas.core.series.Series 

* _fiesta.feature_selection.selection.latent_semantic_analysis (document_collection_category1, document_collection_category2, list_size = 10, visualize = False):

		This method selects relevant features from two document collections based on the singular-value decomposition.

		Parameters: 
			document_collection_category1 (str, list or file directory): document collection of the first category
			document_collection_category2 (str, list or file directory): document collection of the second category
			list_size (int): (default 10) number of features to be returned
			visualize (bool): (default False)   if True it represents the features graphically
		Returns: most relevant features and their their relevance values 
		Return type: pandas.core.frame.DataFrame 

### fiesta.transformers

#### fiesta.transformers.document_transformer
* _fiesta.transformers.document_transformer.document_transformer (document_collection)
	
		Convert string or file directory to an list. 

		Parameters: 
			document_collection (str, list of strings or file directory): document collection
		Returns: transformed string or file directory to an list of strings (documents)
		Return type: list



