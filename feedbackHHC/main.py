import pandas as pd
import util
from random_forest_classifier import Random_Forest_Classifier
from Feature_selection import Feature_selection

file_path = 'HH_Provider_Oct2023.csv'

if __name__ == "__main__":
    # df = pd.read_csv("Preprocessed_data.csv", dtype='float')
    # print(df.dtypes)
    #fs = Feature_selection(df)
    #fs.apply_feature_selection()
    for _ in range(1):
        df = pd.read_csv("Final_data.csv", dtype='float')
        x_train, labels_train, x_test, labels_test = util.split_train_and_test_data(df,
                                                                                    'Quality of patient care star rating',
                                                                                    test_size=0.25)
        # print(f"Size of training data: {x_train.shape}")
        # print(f"Size of training labels: {labels_train.shape}")
        # print(f"Size of testing data: {x_test.shape}")
        # print(f"Size of testing labels: {labels_test.shape}\n")
        model_rf = Random_Forest_Classifier(x_train, labels_train)
        model_rf.train()
        predicted_labels = model_rf.predict_instance(x_train)

        # actual_predicted = list(zip(labels_test, predicted_labels))
        # print(actual_predicted)
        print(f"Accuracy: {util.get_accuracy(labels_train, predicted_labels)}\n")
