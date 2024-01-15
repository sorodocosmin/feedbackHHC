from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import util
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
import time

class NeuralNetworkClassifier:
    def __init__(self, x_train, y_train):
        self.__model = MLPClassifier(max_iter=500, activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=(100, 100), random_state=1)
        self.__x_train = x_train
        self.__y_train = y_train

    def train(self):
        self.__model.fit(self.__x_train, self.__y_train)

    def predict_instance(self, test):
        return self.__model.predict(test)
    
    def cross_fold_validation(self, fold=100):
        return cross_val_score(self.__model, self.__x_train, self.__y_train, cv=fold)



if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\Andrei\\OneDrive\\Desktop\\AI\\Project\\feedbackHHC\\feedbackHHC\\Final_data.csv", dtype='float')
    x_train, labels_train, x_test, labels_test = util.split_train_and_test_data(df,
                                                                                    'Quality of patient care star rating',
                                                                                    test_size=0.25)
    scaler = StandardScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)

    start_time = time.time()
    # Neural Network with Hyperparameter Tuning and Regularization
    model_nn = NeuralNetworkClassifier(x_train_normalized, labels_train)
    model_nn.train()
    print(f"Time elapsed for training: {time.time() - start_time}")
    predicted_labels_test_nn = model_nn.predict_instance(x_test_normalized)
    print(f"Neural Network Accuracy testing: {util.get_accuracy(labels_test, predicted_labels_test_nn)}\n")

    predicted_labels_train_nn = model_nn.predict_instance(x_train_normalized)
    print(f"Neural Network Accuracy train: {util.get_accuracy(labels_train, predicted_labels_train_nn)}\n")

    start_time = time.time()
    cross_f_score = model_nn.cross_fold_validation()
    print(f"Cross fold validation: {len(cross_f_score)}")
    print(f"Cross fold validation: {np.mean(cross_f_score)}")  # by default, 100 folds
    print(f"Time elapsed for cross fold validation: {time.time() - start_time}")

