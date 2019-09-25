from DataParse import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


#get the training features and labels from cvs file
dataset = Dataset().getTrainingDataAndLabels()
# features - training data = 0 to 250 records from data
features1 = dataset[0][:700]
# labels = training labels 0 to 250 records from data
labels = dataset[1][:700]
#features to test = 250 above records
testFeatures1 = dataset[0][700:]
#labels of test data to check accuracy later
testLabels = dataset[1][700:]

print len(testFeatures1)

print len(testLabels)
print "testlabels", testLabels
scalar  = StandardScaler()
scalar.fit(features1)
features = scalar.transform(features1)
testFeatures = scalar.transform(testFeatures1)


mlpClassifier = MLPClassifier(hidden_layer_sizes=(12,12,12), max_iter=1000, activation='tanh')
mlpClassifier.fit(features, labels)
print "for variation - hidden layer size: (12,12,12), max iteration : 3000, activation function : relu\n",
print "accurracy of training at:  ", mlpClassifier.score(features, labels) * 100, "\n"
# print "accurracy of testing at:  ", (accuracy_score(testLabels, prediction)) * 100 "\n"
prediction = mlpClassifier.predict(testFeatures)
print "accurracy of testing at:  ", (accuracy_score(testLabels, prediction)) * 100 ,"\n"
CM = confusion_matrix(testLabels, prediction)

TN = float(CM[0][0])
FN = float(CM[1][0])
TP = float(CM[1][1])
FP = float(CM[0][1])

print "TN", TN
print "FN", FN
print "TP", TP
print "FP", FP

sensitivity = TP / (TP + FN) * 100
specificity = TN / (TN + FP) * 100
pos_pred_val = TP / (TP + FP) * 100
neg_pred_val = TN / (TN + FN) * 100

print "sensitivity:", sensitivity, "\n"
print "specificity:", specificity, "\n"
print "positive prediction value:", pos_pred_val, "\n"
print "negative prediction value:", neg_pred_val, "\n"