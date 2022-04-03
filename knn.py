from time import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def run(data, k_nearest):
    print(">>>>> K-NEAREST NEIGHBORS (k=" + str(k_nearest) +") <<<<<")
    
    np.random.shuffle(data)

    classes = data[:, 57] # Classification for each email in the dataset
    features = data[:, :54] # Features for each email in the dataset

    knn(k_nearest, features, classes)

    
    


def knn(k, features, classes):
    start = time()
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric="euclidean")
    results = cross_val_score(classifier, features, classes, n_jobs=-1)
    end = time()
    printResults(results, end-start)

def printResults(results, time):
    print("******************")
    print("Accuracy (minimum): %.3f%%" % (results.min() * 100))
    print("Accuracy (maximum): %.3f%%" % (results.max() * 100))
    print("Accuracy (mean): %.3f%%" % (results.mean() * 100))
    print("Variance: " + str(results.var()))
    print("\nTime elapsed: "+str(time))
    print("******************\n")