import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def run(dataset):
    print(">>>>> SUPPORT VECTOR MACHINE <<<<<")
 
    np.random.shuffle(dataset)

    classes = dataset[:, 57] # Classification for each email in the dataset
    features = dataset[:, :54] # Features for each email in the dataset

    features = TD_IDF(features) 

    svm("linear", None, features, classes)
    svm("poly", 2, features, classes)
    svm("rbf", None, features, classes)
    # Normalization of dataset in order to exploit angle
    norms = np.sqrt(((features+1e-100)**2).sum(axis=1, keepdims=True))
    features_norm = np.where(norms > 0.0, features / norms, 0.)
    svm("linear", None, features_norm, classes)
    svm("poly", 2, features_norm, classes)
    svm("rbf", None, features_norm, classes)
    

def printResults(results, kernel):
    print("***********")
    print("Kernel: "+kernel)
    print("Minimum accuracy: "+str(results.min()))
    print("Maximum accuracy: "+str(results.max()))
    print("Mean accuracy: "+str(results.mean()))
    print("Variance: " + str(results.var()))
    print("***********")

def svm(kernel, degree, features, classes):
    # If we don't explicitly provide a degree, set this to be the default degree = 3 for the SVC method
    if (degree == None):
        degree = 2 #TODO: DOVREBBE ESSERE INUTILE IN RFB E LINEAR, NO?
    
    classifier = SVC(kernel=kernel, degree=degree) 
    #Evaluate a score by cross-validation Cit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    results = cross_val_score(classifier, features, classes, n_jobs=-1) #-1 = use all processors, 1 use only one processor (default)
    
    printResults(results, classifier.kernel)

#TF/IDF representation.
def TD_IDF(features):
    tf = features/100.0
    ndoc = features.shape[0]
    idf = np.log10(ndoc/(features != 0).sum(0))
    return tf*idf