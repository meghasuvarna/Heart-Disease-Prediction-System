from DataParse import Dataset
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

num = 700

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


variationsOfActRelu = [((12,12,12), 3000, 'relu'), ((12,12,12), 500, 'relu'), ((12,12,12,12,12), 2000, 'relu')]
variationsOfActTanh = [((12,12,12), 500, 'tanh'), ((12,12,12, 12, 12), 1000, 'tanh'), ((12,12,12), 1000, 'tanh')]
variationsOfActLogistic = [((12,12,12), 500, 'logistic'), ((12,12,12,12), 2000, 'logistic')]
variationOfBackPropogation = [((12,12,12), 500, 'identity'), ((12,12,12,12), 500, 'identity')]
allVariations = [variationsOfActRelu, variationsOfActTanh, variationOfBackPropogation, variationsOfActLogistic]

scalar  = StandardScaler()
scalar.fit(features1)
features = scalar.transform(features1)
testFeatures = scalar.transform(testFeatures1)

for variations in allVariations:
    _file = open("mlp_"+variations[0][2]+"_result.txt", 'w')
    for variation in variations:
        hiddenLayerSizes = variation[0]
        max_iter = variation[1]
        activationFun = variation[2]
        _file.write("hidden layers: {hl}, iterations: {itr}, activation fun: {fun} \n".format(hl=hiddenLayerSizes, itr=max_iter, fun=activationFun))
        print "for variation - hidden layer size:", hiddenLayerSizes, "max iteration : ", max_iter, "activation function : ", activationFun
        clf = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, max_iter=max_iter, activation=activationFun)
        clf.fit(features, labels)
        ## check accuracy
        _file.write("accurracy of training at: {score} \n".format(score = clf.score(features, labels) * 100))
        _file.write("accurracy of testing at: {score} \n".format(score = clf.score(testFeatures, testLabels) * 100))
        print "accurracy of training at:  ", clf.score(features, labels) * 100, "\n"
        print "accurracy of testing at:  ", clf.score(testFeatures, testLabels) * 100, "\n"
        prediction = clf.predict(testFeatures)
        _file.write("accuracy score: {score} \n".format(score = accuracy_score(testLabels, prediction) * 100))
        print "accuracy score", (accuracy_score(testLabels, prediction) * 100)
        _file.write(classification_report(testLabels, prediction)+"\n")
        print classification_report(testLabels, prediction)
    _file.close()

