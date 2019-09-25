import csv

training_data_features = []
training_data_labels = []

class Dataset:

    def getTrainingDataAndLabels(self):
        training_data_features = []
        training_data_labels = []

        with open("Dataset\cleveland14.csv") as csvfile:

            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                training_data_features.append(row[:-1])
                training_data_labels.append(row[-1])

        with open("Dataset\hungarian14r.csv") as csvfile:

            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                training_data_features.append(row[:-1])
                training_data_labels.append(row[-1])

        with open("Dataset\heart_disease_all14.csv") as csvfile:

            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                training_data_features.append(row[:-1])
                training_data_labels.append(row[-1])



        return training_data_features, training_data_labels

if __name__ == '__main__':
    Dataset().getTrainingDataAndLabels();
