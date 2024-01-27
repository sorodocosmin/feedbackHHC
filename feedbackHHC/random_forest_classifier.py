import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from hyperparameter_tunning import grid_search
from hyperparameter_tunning import random_search
import pandas as pd
import numpy as np


class Random_Forest_Classifier:
    def __init__(self, x_train, y_train):
        """
        Initialize a Random Forest Classifier
        :param x_train: the training data
        :param y_train: the training labels
        """
        self.__model = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, n_estimators=500)
        self.__x_train = x_train
        self.__y_train = y_train

    def train(self):
        """
        Train the model
        """
        self.__model.fit(self.__x_train, self.__y_train)

    def predict(self, test):
        """
        Predict the labels for the test data
        :param test: a list of test data
        :return: a list of predicted labels
        """
        return self.__model.predict(test)

    def cross_fold_validation(self, fold=100):
        """
        Apply cross fold validation
        :param fold: a number representing the number of folds (by default, 5)
        :return: the accuracy for cross fold validation
        """
        return cross_val_score(self.__model, self.__x_train, self.__y_train, cv=fold)


if __name__ == '__main__':
    pass
    # here, we will make a Hyperparameter tunning using Grid Search and Random Search for the Random Forest Classifier
    # model = RandomForestClassifier()
    # possible_parameters_rf = {
    #     "n_estimators": [10, 100, 200, 500, 750, 1_000],
    #     "criterion": ["gini", "entropy"],
    #     # the maximum depth of the tree is represented by the nr of features which is 12
    #     "max_depth": [None, 5, 7, 8, 9, 10],
    #     # Bootstrap means that instead of training on all the observations,
    #     # each tree of RF is trained on a subset of the observations
    #     "bootstrap": [True, False]
    # }

    # possible_parameters_rf = {
    #     'n_estimators': [50, 100, 300],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5, 10],
    #     'max_features': ['auto', 'sqrt', 'log2'],  # Adjust based on your preference
    #     'bootstrap': [True, False]
    # }
    #
    # df = pd.read_csv("Final_data.csv")
    # x_train, labels_train, x_test, labels_test = (
    #     util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #
    # start_time = time.time()
    # best_parameters, best_score = random_search(possible_parameters_rf, model, x_train, labels_train)
    # print(f"Best parameters: {best_parameters} ; Best score: {best_score}")
    # print(f"Time elapsed for random Search: {time.time() - start_time}")
    #
    # start_time = time.time()
    # best_parameters, best_score = grid_search(possible_parameters_rf, model, x_train, labels_train)
    # print(f"Best parameters: {best_parameters} ; Best score: {best_score}")
    # print(f"Time elapsed for grid Search: {time.time() - start_time}")



    # Now, we will train the model using the best parameters obtained from the hyperparameter tunning
    # df = pd.read_csv("Final_data.csv", dtype='float')
    # column_to_exclude = 'Quality of patient care star rating'
    #
    # # Find and print duplicate rows excluding the specified column
    # duplicate_rows = df[df.duplicated(keep=False, subset=df.columns.difference([column_to_exclude]))]
    # # set_lebels = set(duplicate_rows['Quality of patient care star rating'])
    # duplicate_sets = set()
    # for index, row in duplicate_rows.iterrows():
    #     # Get values of all columns except the excluded one
    #     values_except_excluded = row.drop(column_to_exclude).values
    #
    #     # Get the value of the excluded column
    #     value_excluded = row[column_to_exclude]
    #
    #     # Create a tuple and add it to the list
    #     duplicate_sets.add((tuple(values_except_excluded), value_excluded))
    #
    # for s in duplicate_sets:
    #     print(s)
    #
    #
    # Display the duplicate rows
    # print("Duplicate Rows (excluding", column_to_exclude, "):")
    # print(duplicate_rows)

    # accuracy_test = []
    # accuracy_train = []
    # time_elapsed = []
    # nr_tests = 10
    # for _ in range(nr_tests):
    #     x1_train, labels1_train, x1_test, labels1_test = (
    #         util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #     start_time = time.time()
    #
    #     model_rf = Random_Forest_Classifier(x1_train, labels1_train)
    #     model_rf.train()
    #     end_time = time.time()
    #     time_elapsed.append(end_time - start_time)
    #     print(f"Time elapsed for training the data: {end_time - start_time}")
    #     predicted_labels_train = model_rf.predict(x1_train)
    #     predicted_labels_test = model_rf.predict(x1_test)
    #     curr_acc_test = util.get_accuracy(labels1_test, predicted_labels_test)
    #     curr_acc_train = util.get_accuracy(labels1_train, predicted_labels_train)
    #     print(f"Accuracy testing: {curr_acc_test}")
    #     print(f"Accuracy training: {curr_acc_train}")
    #     accuracy_test.append(curr_acc_test)
    #     accuracy_train.append(curr_acc_train)
    #     print()
    #
    # print(f"Average accuracy testing: {sum(accuracy_test) / nr_tests}")
    # print(f"Average accuracy training: {sum(accuracy_train) / nr_tests}")
    # print(f"Average time elapsed: {sum(time_elapsed) / nr_tests}")

    # compute cross fold validation
    # x1_train, labels1_train, x1_test, labels1_test = (
    #     util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.20))
    #
    # model_rf = Random_Forest_Classifier(x1_train, labels1_train)
    # model_rf.train()
    # start_time = time.time()
    # cross_f_score = model_rf.cross_fold_validation()
    # print(f"Cross fold validation: {len(cross_f_score)}")
    # print(f"Cross fold validation: {np.mean(cross_f_score)}")  # by default, 100 folds
    # print(f"Time elapsed for cross fold validation: {time.time() - start_time}")

    # df = pd.read_csv("Final_data.csv")
    # # # df['Provider Name'] = df['Provider Name'].apply(eval)  # will transform that string to a list
    # # # df['Provider Name'] = df['Provider Name'].apply(np.array)  # will transform that string to a list
    # # print(df)
    # accuracy_test = []
    # accuracy_train = []
    # time_elapsed = []
    #
    # nr_tests = 10
    # for _ in range(nr_tests):
    #     x1_train, labels1_train, x1_test, labels1_test = (
    #         util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #     start_time = time.time()
    #
    #     model_rf = Random_Forest_Classifier(x1_train, labels1_train)
    #     model_rf.train()
    #     end_time = time.time()
    #     time_elapsed.append(end_time - start_time)
    #     print(f"Time elapsed for training the data: {end_time - start_time}")
    #     predicted_labels_train = model_rf.predict(x1_train)
    #     predicted_labels_test = model_rf.predict(x1_test)
    #     curr_acc_test = util.get_accuracy(labels1_test, predicted_labels_test)
    #     curr_acc_train = util.get_accuracy(labels1_train, predicted_labels_train)
    #     print(f"Accuracy testing: {curr_acc_test}")
    #     print(f"Accuracy training: {curr_acc_train}")
    #     accuracy_test.append(curr_acc_test)
    #     accuracy_train.append(curr_acc_train)
    #     print()
    #
    # print(f"Average accuracy testing: {sum(accuracy_test) / nr_tests}")
    # print(f"Average accuracy training: {sum(accuracy_train) / nr_tests}")
    # print(f"Average time elapsed: {sum(time_elapsed) / nr_tests}")



