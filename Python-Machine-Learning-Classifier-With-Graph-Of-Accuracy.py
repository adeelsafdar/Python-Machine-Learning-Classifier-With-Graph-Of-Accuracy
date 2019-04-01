from __future__ import division  # only for Python 2
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from warnings import simplefilter
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib.ticker import PercentFormatter

simplefilter(action='ignore', category=FutureWarning)

wine = datasets.load_wine()

features = wine.data
labels = wine.target

train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)

def learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf,name):
	# print("\n")
	# print(name);

	clf.fit(train_feats, train_labels)

	predictions = clf.predict(test_feats)
	# print("Predictions:", predictions)

	score = 0
	for i in range(len(predictions)):
	    if predictions[i] == test_labels[i]:
	        score += 1
	# print("Accuracy:", (score / len(predictions)) * 100, "%")

	# to_return = {'name':name, 'accuraccy' : round(((score / len(predictions)) * 100), 2)}
	# to_return = {name: round(((score / len(predictions)) * 100), 2)}
	to_return = round(((score / len(predictions)) * 100), 2)
	return to_return

clf1 = RandomForestClassifier()
clf2 = tree.DecisionTreeClassifier()
clf3 = KNeighborsClassifier()
clf4 = GaussianProcessClassifier()
clf5 = svm.SVC(kernel='linear')
clf6 = svm.SVC(gamma=2, C=1)
clf7 = MLPClassifier(alpha=1)
clf8 = AdaBoostClassifier()
clf9 = GaussianNB()
clf10 = QuadraticDiscriminantAnalysis()

percentages = ()

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf1,'RandomForestClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf2,'DecisionTreeClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf3,'KNeighborsClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf4,'GaussianProcessClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf5,'linear'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf6,'gamma'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf7,'MLPClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf8,'AdaBoostClassifier'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf9,'GaussianNB'))
percentages = tuple(percentages)

percentages = list(percentages)
percentages.insert(len(percentages), learn_classify_and_get_accuracy(train_feats,test_feats,train_labels,test_labels,clf10,'QuadraticDiscriminantAnalysis'))
percentages = tuple(percentages)

print(percentages)

n_groups = 10

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.5

rects1 = plt.bar(index, percentages, bar_width, color='g', 
    label='Classifier', alpha= 0.8)

plt.xlabel('Classifier')
plt.ylabel('Percentage')
plt.title('Classifiers Accuracy Percentage')
plt.xticks(index + (bar_width - 0.3), ('RFC','DTC','KN','GPC','Linear','Gamma','MPLC','ABC','GNB','QDA'))
plt.legend()

print(index + bar_width)

for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width(), 0.99*height,
            '%d' % int(height) + "%", ha='center', va='bottom')

plt.tight_layout()
plt.show()