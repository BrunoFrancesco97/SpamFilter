import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def run(dataset):
    np.random.shuffle(dataset)
    classes = dataset[:, 57] 
    features = dataset[:, :54] 
    features = TD_IDF(features) 
    svm("linear", 3, features, classes) #3 default degree  value in SVC, it's ignored in linear and rbf
    svm("poly", 2, features, classes)
    svm("rbf", 3, features, classes)
    # Normalization of dataset in order to exploit angle
    norms = np.sqrt(((features+1e-100)**2).sum(axis=1, keepdims=True))
    features_norm = np.where(norms > 0.0, features / norms, 0.)
    svm("linear", 3, features_norm, classes)
    svm("poly", 2, features_norm, classes)
    svm("rbf", 3, features_norm, classes)
    

def view(result, kernel):
    print("***********")
    print("Kernel: "+kernel)
    print("Minimum accuracy: "+str(result.min()))
    print("Maximum accuracy: "+str(result.max()))
    print("Accuracy mean: "+str(result.mean()))
    print("***********")

def svm(kernel, degree, features, classes):
    classifier = SVC(kernel=kernel, degree=degree) 
    result = cross_val_score(classifier, features, classes, n_jobs=-1) 
    view(result, classifier.kernel)

#TF/IDF representation.
def TD_IDF(features):
    tf = features/100.0
    ndoc = features.shape[0]
    idf = np.log10(ndoc/(features != 0).sum(0))
    return tf*idf