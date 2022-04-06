import numpy as np
import svm as svm
import knn as knn


filedata = "spambase/spambase.data"
file = open(filedata, "r")
dataset = np.loadtxt(file, delimiter = ",")

print("SVM Classifier:")
svm.run(dataset)

# Run Naive Bayes Classifier on dataset with 10-fold cross validation
#k_folds = 10
#NB.run(dataset, k_folds)

print("K-Nearest Neighbors Classifier:")
knn.run(dataset, 5)