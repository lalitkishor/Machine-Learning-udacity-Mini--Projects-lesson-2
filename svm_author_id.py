#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###

#########################################################

from sklearn.svm import SVC
clf=SVC(kernel='rbf',C=10000.0) #Optimized C 
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print pred
#print pred[10]
#print pred[26]
#print pred[50]
y=0
z=0
for x in pred:
    if x==0:
        y=y+1
    else:
        z=z+1
if y>z:
    print "Sara"
if z>y:
    print "Chris"
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc
