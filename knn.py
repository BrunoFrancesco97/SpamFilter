import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def run(data, k):
    np.random.shuffle(data)
    classes = data[:, 57] # Classification for each email in the dataset
    features = data[:, :54] # Features for each email in the dataset
    knn(k, features, classes)

def knn(k, features, classes):
    classifier = KNeighborsClassifier(n_neighbors=k)
    results = cross_val_score(classifier, features, classes, n_jobs=-1)
    view(results)

def view(results):
    print("***********")
    print("Minimum accuracy: "+str(results.min()))
    print("Maximum accuracy: "+str(results.max()))
    print("Accuracy mean: "+str(results.mean()))
    print("***********")