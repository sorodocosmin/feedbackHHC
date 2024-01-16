from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import pandas as pd
import util
import time
import numpy as np
class NaiveBayesClassifier:
    def __init__(self, x_train, y_train):
        self.model = GaussianNB()
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        start_time = time.time()
        self.model.fit(self.x_train, self.y_train)
        training_time = time.time() - start_time
        return training_time

    def predict(self, x_test):
        return self.model.predict(x_test)

    def cross_fold_validation(self, folds=100):
        start_time = time.time()
        return cross_val_score(self.model, self.x_train, self.y_train, cv=folds, scoring='accuracy'), time.time() - start_time


if __name__ == "__main__":
    df = pd.read_csv("Final_data.csv", dtype='float')
    x_train, labels_train, x_test, labels_test = util.split_train_and_test_data(df,
                                                                                'Quality of patient care star rating',
                                                                                test_size=0.25)
    model_nb = NaiveBayesClassifier(x_train, labels_train)

    training_time = model_nb.train()

    predicted_labels_train_nn = model_nb.predict(x_train)
    accuracy_train = util.get_accuracy(labels_train, predicted_labels_train_nn)

    predicted_labels_test_nn = model_nb.predict(x_test)
    accuracy_test = util.get_accuracy(labels_test, predicted_labels_test_nn)

    cv_scores, cv_time = model_nb.cross_fold_validation(folds=100)

    with open("analyze/naive_bayes_stats.txt", "w") as file:
        file.write(f"Time elapsed for training: {training_time} seconds\n")
        file.write(f"Accuracy testing Bayes Naive: {accuracy_test}\n")
        file.write(f"Accuracy training Bayes Naive: {accuracy_train}\n")
        file.write(f"Cross-Validation Scores: {np.mean(cv_scores)}\n")
        file.write(f"Time elapsed for Cross-Validation : {cv_time} seconds\n")
