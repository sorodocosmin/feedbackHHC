from random_forest_classifier import Random_Forest_Classifier
from adaboost_classifier import AdaBoost_Classifier
from naive_bayes_classifier import NaiveBayesClassifier
from neuronal_networks_classifier import NeuralNetworkClassifier
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import util


def main():
    # Load the data
    df = pd.read_csv("Final_data.csv")

    x1_train, labels1_train, x1_test, labels1_test = (
        util.split_train_and_test_data(df, 'Quality of patient care star rating', test_size=0.25))

    # Convert multiclass labels to binary labels (one-hot encoding)
    # ex [1, 3, 2, 3, 3, 0, 10] -> [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0], ...]
    unique_labels = util.get_classes(labels1_train)
    y_test_bin = label_binarize(labels1_test, classes=unique_labels)

    classifiers = [AdaBoost_Classifier(x1_train, labels1_train), Random_Forest_Classifier(x1_train, labels1_train),
                   NaiveBayesClassifier(x1_train, labels1_train), NeuralNetworkClassifier(x1_train, labels1_train)]
    models_name = ["AdaBoost", "RandomForest", "NaiveBayes", "NeuralNetwork"]

    # Train the models
    for cls in classifiers:
        cls.train()

    y_score_all = [clf.predict(x1_test) for clf in classifiers]
    # transform the labels to one-hot encoding
    y_score_all = [label_binarize(y, classes=unique_labels) for y in y_score_all]

    for i in unique_labels:
        plt.figure()

        for y_score, model_name in zip(y_score_all, models_name):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {unique_labels[i]}')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    main()
