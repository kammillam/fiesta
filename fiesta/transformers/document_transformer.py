from os.path import isfile

def document_transformer  (document):
    """Convert string or file directory to an list 
    Args:
        document(str, list of strings or file directory): document collection
    Returns:
        list: transformed string or file directory 
    """
    full_document = []

    if  type(document) == list:
        full_document = document
    elif isfile(document):    
        document = open(document, "r")
        full_document = document.read().split('\n')
        document.close()  

    elif type(document) == str:
        full_document.append(document)
    
    return full_document