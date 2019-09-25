from DataParse import Dataset
from sklearn import  svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Example(object):



    def run(self):
        svc_scores = []
        trainingData, trainingLabels = Dataset().getTrainingDataAndLabels()
        trainingLabelsForLinear = list(trainingLabels)

        for index, value in enumerate(trainingLabelsForLinear):
            if value > 0.0:
                trainingLabelsForLinear[index] = 1

        #features - training data = 0 to 700 records from data
        features = trainingData[:700]
        # labels = training labels 0 to 700 records from data
        labels = trainingLabels[:700]
        linearLabels = trainingLabelsForLinear[:700]

        # features to test = 700 above records
        testFeatures = trainingData[700:]

        # labels of test data to check accuracy later
        testLabels = trainingLabels[700:]
        testLinearlabels = trainingLabelsForLinear[700:]


        # #linear kernel
        svc_classifier = svm.SVC(C=2, kernel='linear')
        svc_classifier.fit(features, linearLabels)
        svc_scores.append(svc_classifier.score(testFeatures, testLinearlabels))

        prediction = svc_classifier.predict(testFeatures)
        print "------------- linear----------"
        print "Prediction for test data\n",prediction, "\n"
        print "Accuracy of prediction",(accuracy_score(testLinearlabels,prediction, normalize=True)), "\n"
        print classification_report(testLinearlabels, prediction), "\n"

        CM = confusion_matrix(testLinearlabels, prediction)

        TN = float(CM[0][0])
        FN = float(CM[1][0])
        TP = float(CM[1][1])
        FP = float(CM[0][1])

        print "TN:", TN, "\n"
        print "FN:", FN, "\n"
        print "TP:", TP, "\n"
        print "FP:", FP, "\n"

        sensitivity = TP / (TP + FN) * 100
        specificity = TN / (TN + FP) * 100
        pos_pred_val = TP / (TP + FP) * 100
        neg_pred_val = TN / (TN + FN) * 100

        print "sensitivity:", sensitivity, "\n"
        print "specificity:", specificity, "\n"
        print "positive prediction value:", pos_pred_val, "\n"
        print "negative prediction value:", neg_pred_val, "\n"

        print classification_report(testLinearlabels, prediction)

        # rbf
        svc_classifier = svm.SVC(C=10, kernel='rbf', gamma=1)
        svc_classifier.fit(features, labels)
        svc_scores.append(svc_classifier.score(testFeatures, testLabels))
        prediction = svc_classifier.predict(testFeatures)
        print "------------- RBF(Non Linear)----------"
        print "Prediction for test data\n",prediction, "\n"
        print "Accuracy of prediction", (accuracy_score(testLabels, prediction, normalize=True)), "\n"
        print classification_report(testLabels, prediction), "\n"

        CM = confusion_matrix(testLabels, prediction, [0,1,2,3,4])

        TN = float(CM[0][0])
        FN = float(CM[1][0])
        TP = float(CM[1][1])
        FP = float(CM[0][1])

        print "TN:", TN, "\n"
        print "FN:", FN, "\n"
        print "TP:", TP, "\n"
        print "FP:", FP, "\n"

        sensitivity = TP / (TP + FN) * 100
        specificity = TN / (TN + FP) * 100
        pos_pred_val = TP / (TP + FP) * 100
        neg_pred_val = TN / (TN + FN) * 100

        print "sensitivity:", sensitivity, "\n"
        print "specificity:", specificity, "\n"
        print "positive prediction value:", pos_pred_val, "\n"
        print "negative prediction value:", neg_pred_val, "\n"

        print classification_report(testLabels, prediction)







if __name__ == '__main__':
    Example().run()