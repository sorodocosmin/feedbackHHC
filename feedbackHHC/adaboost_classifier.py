import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time

from hyperparameter_tunning import grid_search
from hyperparameter_tunning import random_search

import util


class AdaBoost_Classifier:
    def __init__(self, x_train, y_train):
        """
        Initialize a AdaBoost Classifier
        :param x_train: the training data
        :param y_train: the training labels
        """
        self.__model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, criterion='gini'), n_estimators=500,
                                          learning_rate=2.0)
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
    # hyperparameters tunning for the AdaBoost classifier
    # model = AdaBoostClassifier()
    # possible_parameters_ab = {
    #     "n_estimators": [10, 25, 50, 100, 200, 250, 500],  # which is the number of iterations
    #     # learning rate is mutiplied with the weight of the distribution for each iteration
    #     "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    #     "estimator": [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2, criterion="gini"),
    #                   DecisionTreeClassifier(max_depth=2, criterion='entropy')]
    #
    # }
    # df = pd.read_csv("Final_data.csv")
    # x_train, labels_train, x_test, labels_test = (
    #     util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #
    # accuracy_test = []
    # accuracy_train = []
    # time_elapsed = []
    # nr_tests = 10
    # for _ in range(nr_tests):
    #     x1_train, labels1_train, x1_test, labels1_test = (
    #         util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #     start_time = time.time()
    #
    #     model_rf = AdaBoost_Classifier(x1_train, labels1_train)
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

    # Hyperparameter tunning

    # start_time = time.time()
    # best_parameters, best_score = grid_search(possible_parameters_ab, model, x_train, labels_train)
    # print(f"Best parameters: {best_parameters} ; Best score: {best_score}")
    # print(f"Time elapsed for grid Search: {time.time() - start_time}")
    #
    # start_time = time.time()
    # best_parameters, best_score = random_search(possible_parameters_ab, model, x_train, labels_train)
    # print(f"Best parameters: {best_parameters} ; Best score: {best_score}")
    # print(f"Time elapsed for random Search: {time.time() - start_time}")

    # compute cross fold validation
    # x1_train, labels1_train, x1_test, labels1_test = (
    #     util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
    #
    # model_ab = AdaBoost_Classifier(x1_train, labels1_train)
    # model_ab.train()
    # start_time = time.time()
    # cross_f_score = model_ab.cross_fold_validation()
    # print(f"Cross fold validation: {len(cross_f_score)}")
    # print(f"Cross fold validation: {np.mean(cross_f_score)}")  # by default, 100 folds
    # print(f"Time elapsed for cross fold validation: {time.time() - start_time}")

    df = pd.read_csv("Final_1_data.csv")
    # df['Provider Name'] = df['Provider Name'].apply(eval)  # will transform that string to a list
    # df['Provider Name'] = df['Provider Name'].apply(np.array)  # will transform that string to a list

    accuracy_test = []
    accuracy_train = []
    time_elapsed = []

    nr_tests = 10
    for _ in range(nr_tests):
        x1_train, labels1_train, x1_test, labels1_test = (
            util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))
        start_time = time.time()

        # scaler = StandardScaler()
        # x1_train = scaler.fit_transform(x1_train)
        # x1_test = scaler.transform(x1_test)

        model_rf = AdaBoost_Classifier(x1_train, labels1_train)
        model_rf.train()
        end_time = time.time()
        time_elapsed.append(end_time - start_time)
        print(f"Time elapsed for training the data: {end_time - start_time}")
        predicted_labels_train = model_rf.predict(x1_train)
        predicted_labels_test = model_rf.predict(x1_test)
        curr_acc_test = util.get_accuracy(labels1_test, predicted_labels_test)
        curr_acc_train = util.get_accuracy(labels1_train, predicted_labels_train)
        print(f"Accuracy testing: {curr_acc_test}")
        print(f"Accuracy training: {curr_acc_train}")
        accuracy_test.append(curr_acc_test)
        accuracy_train.append(curr_acc_train)
        print()

    print(f"Average accuracy testing: {sum(accuracy_test) / nr_tests}")
    print(f"Average accuracy training: {sum(accuracy_train) / nr_tests}")
    print(f"Average time elapsed: {sum(time_elapsed) / nr_tests}")

