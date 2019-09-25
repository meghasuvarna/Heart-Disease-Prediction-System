import csv

training_data_features = []
training_data_labels = []
with open("C:\Users\Megha Suvarna\Desktop\FAI_Project\Dataset\cleveland14.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        training_data_features.append(row[:-1])
        training_data_labels.append(row[-1])

with open("C:\Users\Megha Suvarna\Desktop\FAI_Project\Dataset\hungarian14r.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        training_data_features.append(row[:-1])
        training_data_labels.append(row[-1])


print "training data", training_data_features
print "training data labels", training_data_labels
print "len of traning data", len(training_data_features)
print "len of traning labels", len(training_data_labels)