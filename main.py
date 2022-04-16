import numpy as np
import svm as svm
import knn as knn
import nb as nb

filedata = "spambase/spambase.data"
file = open(filedata, "r")
dataset = np.loadtxt(file, delimiter = ",")
print("SVM Classifier:")
svm.run(dataset)
print("Naive-Bayes Classifier:")
nb.nb(dataset)
print("K-Nearest Neighbors Classifier:")
knn.run(dataset, 5)
