import numpy as np
import svm as svm
import knn as knn


filedata = "spambase/spambase.data"
file = open(filedata, "r")
dataset = np.loadtxt(file, delimiter = ",")

# Run SVM Classifier on dataset
# Will run through different kernels: linear, polynomial, RBF, linear angular, polynomial angular, RBF angular
svm.run(dataset)

# Run Naive Bayes Classifier on dataset with 10-fold cross validation
#k_folds = 10
#NB.run(dataset, k_folds)

# Run K-Nearest Neighbors Classifier on dataset with k=5
k_nearest = 5
knn.run(dataset, k_nearest)