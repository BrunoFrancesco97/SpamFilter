import numpy as np
import utils as u
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import math
import sys 



class Naive_Bayes_Classifier(BaseEstimator):
    def fit(self,X,Y):
        features = X.copy()
        classes = Y.copy()
        self.spam = []
        self.no_spam = []     
        for i in range(0,len(features)):
            if(classes[i] == 1):
                self.spam.append(features[i])
            if(classes[i] == 0):
                self.no_spam.append(features[i])   
        self.spam = np.array(self.spam)
        self.no_spam = np.array(self.no_spam)
        total_length = len(self.spam)+len(self.no_spam)
        self.spam_prob = len(self.spam)/total_length
        self.no_spam_prob = len(self.no_spam)/total_length
        self.spam_mean = self.spam.mean(axis = 0)
        self.no_spam_mean = self.no_spam.mean(axis = 0)
        self.spam_variance = self.spam.var(axis = 0)
        self.no_spam_variance = self.no_spam.var(axis = 0)
        
    
    def score(self, X, Y):
        spam_prod_internal = np.power((2 * math.pi * (self.spam_variance+sys.float_info.epsilon)),(-1/2)) * np.exp((-1./(2*self.spam_variance+sys.float_info.epsilon)) * np.power((X-self.spam_mean),2))
        no_spam_prod_internal = np.power((2 * math.pi * (self.no_spam_variance+sys.float_info.epsilon)),(-1/2)) * np.exp((-1./(2*self.no_spam_variance+sys.float_info.epsilon)) * np.power((X-self.no_spam_mean),2))
        spam_prod = spam_prod_internal.prod(axis=1)
        no_spam_prod = no_spam_prod_internal.prod(axis=1)
        spam_prob = spam_prod * self.spam_prob
        no_spam_prob = no_spam_prod * self.no_spam_prob
        labels = np.argmax([no_spam_prob,spam_prob], axis = 0)
        success = 0
        index = 0
        for i in labels:
            if i == Y[index]:        
                success = success + 1
            index = index + 1
        return (success / len(labels))
    
def run(dataset):
    np.random.shuffle(dataset)
    classifier = Naive_Bayes_Classifier()
    features = dataset[:, :54]
    classes = dataset[:, -1]
    features = u.tf_idf(features)
    results = cross_val_score(classifier, features, classes, cv=10, n_jobs=-1)
    view(results)

def view(result):
    print("***********")
    print("Minimum accuracy: "+str(result.min()))
    print("Maximum accuracy: "+str(result.max()))
    print("Accuracy mean: "+str(result.mean()))
    print("Accuracy variance: "+str(result.var()))
    print("***********")
