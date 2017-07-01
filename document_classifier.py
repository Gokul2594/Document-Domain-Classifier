import os
import re
import nltk
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup             
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

 
"""

1. This code supports both windows and Linux platforms.
2. Install the required pakckages.
3. Dataset is assumed to be a csv with two columns "Data" and "Target"
4. Specify the path to train dataset
5. Run the code to classify your data.
6. If the data is too large use Cross-Validation to split train and test dataset


"""      

#Base Path
ROOT_PATH = os.path.abspath(__file__)

# Pickle files folder name
pickle_files = "pickle_files"

def pre_processing(raw_data):
    # Function to convert a raw data to a string of words
    # The input is a single string (a raw data), and 
    # the output is a single string (a preprocessed data)
    
    # 1. Remove HTML
    text = BeautifulSoup(raw_data).get_text() 
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
     
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

# function to train your data
def train():

    #Get the training data
    train = pd.read_csv("Path to your Train Data", header=0, delimiter="\t")

    #pre_processing the train data
    clead_data = pre_processing( train["Data"][0] )

    num_data = train["Data"].size
    clean_train = []

    print ("Cleaning and parsing the training data...\n")
    clean_train = []
    for i in range( 0, num_data ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%10 == 0 ):
            print ("Data %d of %d\n" % ( i+1, num_data ) )                                                                   
        clean_train.append( pre_processing( train["Data"][i] ))

    print ("Creating the bag of words...\n")    

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it 
    # appears in the training set

    print ("Training the Random Forest...")

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["Target"] )
    
    # used try to run commands in both Windows and Linux Platform
    try:
        if not os.path.exists(ROOT_PATH + "\\" + pickle_files "\\train.pickle"):
            try:
                os.makedirs(pickle_files)
            except:
                pass
    except:
        if not os.path.exists(ROOT_PATH + "/" + pickle_files "/train.pickle"):
            try:
                os.makedirs(pickle_files)
            except:
                pass

            # used try to run commands in both Windows and Linux Platform
            try:               
                save_vectorizer = open(ROOT_PATH + "\\" + pickle_files "\\vectorizer.pickle","wb")
                pickle.dump(vectorizer, save_vectorizer)
                save_vectorizer.close()
            except:
                save_vectorizer = open(ROOT_PATH + "/" + pickle_files "/vectorizer.pickle","wb")
                pickle.dump(vectorizer, save_vectorizer)
                save_vectorizer.close()
            try:
                save_classifier = open(ROOT_PATH + "\\" + pickle_files "\\train.pickle","wb")
                pickle.dump(train, save_classifier)
                save_classifier.close()
            except:
                save_vectorizer = open(ROOT_PATH + "/" + pickle_files "/train.pickle","wb")
                pickle.dump(vectorizer, save_vectorizer)
                save_vectorizer.close()

# Use this function to use test data as CSV and the output is saved to a dataset
def bulk_test():

    # used try to run commands in both Windows and Linux Platform
    try:
        if not os.path.exists(ROOT_PATH + "\\" + pickle_files "\\train.pickle"):
            train()
    except:
        if not os.path.exists(ROOT_PATH + "/" + pickle_files "/train.pickle"):
            train()

    # used try to run commands in both Windows and Linux Platform
    try:
        train_f = open(ROOT_PATH + "\\" + pickle_files "\\train.pickle","rb")
        train = pickle.load(train_f)
        vectorizer_f = open(ROOT_PATH + "\\" + pickle_files "\\vectorizer.pickle","rb")
        vectorizer = pickle.load(vectorizer_f)
    except:
        train_f = open(ROOT_PATH + "/" + pickle_files "/train.pickle","rb")
        train = pickle.load(train_f)
        vectorizer_f = open(ROOT_PATH + "/" + pickle_files "/vectorizer.pickle","rb")
        vectorizer = pickle.load(vectorizer_f)

    test = pd.read_csv("Path to your test dataset", header=0, delimiter="\t")

    # Create an empty list and append the clean data one by one
    num_data = len(test["Question"])
    clean_test = [] 

    print ("Cleaning and parsing the test data...\n")
    for i in range(0,num_data):
        if( (i+1) % 100 == 0 ):
            print ("questions %d of %d\n" % (i+1, num_data))
        clead_data = pre_processing( test["Question"][i] )
        clean_test.append( clead_data )

    #~ # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test)
    test_data_features = test_data_features.toarray()

    #~ # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)
    print(result)
     
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"Question":test["Question"], "Target":result} ) 

    # Use pandas to write the comma-separated output file
    output.to_csv( "Testing3_result", index=True, quoting=3 , escapechar='|')

    return result

# Use this function to use single test data 
def single_input_test():

    # used try to run commands in both Windows and Linux Platform
    try:
        if not os.path.exists(ROOT_PATH + "\\" + pickle_files "\\train.pickle"):
            train()
    except:
        if not os.path.exists(ROOT_PATH + "/" + pickle_files "/train.pickle"):
            train()
    
    # used try to run commands in both Windows and Linux Platform
    try:
        train_f = open(ROOT_PATH + "\\" + pickle_files "\\train.pickle","rb")
        train = pickle.load(train_f)
        vectorizer_f = open(ROOT_PATH + "\\" + pickle_files "\\vectorizer.pickle","rb")
        vectorizer = pickle.load(vectorizer_f)
    except:
        train_f = open(ROOT_PATH + "/" + pickle_files "/train.pickle","rb")
        train = pickle.load(train_f)
        vectorizer_f = open(ROOT_PATH + "/" + pickle_files "/vectorizer.pickle","rb")
        vectorizer = pickle.load(vectorizer_f)
            
    #~ # Read the test data
    test = input("Enter your test data: ")

    # Create an empty list and append the clean data one by one
    num_data = len(test["Question"])
    clean_test = [] 

    print ("Cleaning and parsing the test data...\n")
    for i in range(0,num_data):
        if( (i+1) % 100 == 0 ):
            print ("questions %d of %d\n" % (i+1, num_data))
        clead_data = pre_processing( test["Question"][i] )
        clean_test.append( clead_data )

    #~ # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test)
    test_data_features = test_data_features.toarray()

    #~ # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)
    print(result)

    return result


"""

 Un-comment anyone of the function depending on your requirement

"""

# bulk_test()
# single_input_test()

