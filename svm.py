import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def run(dataset):
    np.random.shuffle(dataset)
    classes = dataset[:, -1]
    features = dataset[:, :54]
    features = tf_idf(features)
    svm("linear", 3, features, classes) #3 default degree  value in SVC, it's ignored in linear and rbf
    svm("poly", 2, features, classes)
    svm("rbf", 3, features, classes)
    features = normalize(features)
    svm("linear", 3, features, classes)
    svm("poly", 2, features, classes)
    svm("rbf", 3, features, classes)
    

def view(result, kernel):
    print("***********")
    print("Kernel: "+kernel)
    print("Minimum accuracy: "+str(result.min()))
    print("Maximum accuracy: "+str(result.max()))
    print("Accuracy mean: "+str(result.mean()))
    print("Accuracy variance: "+str(result.var()))
    print("***********")

def svm(kernel, degree, features, classes):
    classifier = SVC(kernel=kernel, degree=degree) 
    result = cross_val_score(classifier, features, classes, cv=10, n_jobs=-1) 
    view(result, classifier.kernel)

def tf_idf(features):
    ndoc = features.shape[0]
    idf = np.log10(ndoc/(features != 0).sum(0))
    return (features/100.0)*idf

def normalize(features):
    norms = np.sqrt((np.power(features+sys.float_info.epsilon,2)).sum(axis=1, keepdims=True))
    return np.where(norms > 0.0, features / norms, 0.)
