import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def run(data, k):
    np.random.shuffle(data)
    classes = data[:, -1] 
    features = data[:, :54] 
    knn(k, features, classes)

def knn(k, features, classes):
    classifier = KNeighborsClassifier(n_neighbors=k)
    results = cross_val_score(classifier, features, classes, cv=10, n_jobs=-1)
    view(results)

def view(results):
    print("***********")
    print("Minimum accuracy: "+str(results.min()))
    print("Maximum accuracy: "+str(results.max()))
    print("Accuracy mean: "+str(results.mean()))
    print("Accuracy variance: "+str(results.var()))
    print("***********")
