from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from random_forest_classifier import Random_Forest_Classifier
import pandas as pd


def get_accuracy(actual_labels, predicted_labels):
    """
    Get the accuracy of the model
    :param actual_labels: a list of the actual labels
    :param predicted_labels: a list of the predicted labels
    :return: the accuracy of the model as a float number, between 0 and 1
    """
    return accuracy_score(actual_labels, predicted_labels)


def split_train_and_test_data(df, name_output_column, test_size=0.2):
    """
    Split the data into training and testing data
    :param df: a pandas dataframe
    :param name_output_column: the name of the output column
    :param test_size: a float number (<1) representing the percentage of the test data
    :return: a tuple containing the training and testing data, in this format:
    (x_train, y_train, x_test, y_test), where these are np.ndarray
    and y_train[0] is the output of the first instance in the training data (x_train[0])
    """
    datas_without_output = df.drop(name_output_column, axis=1)  # axis=1 means column
    output = df[name_output_column].astype('category').cat.codes  # convert the float numbers to numeric values
    # ex : 0.5 -> 0, 1.5 -> 1, 2.0 -> 2, ..etc
    x_train, x_test, y_train, y_test = train_test_split(datas_without_output.values, output.values, test_size=test_size)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # example of how to use the functions
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1],
        'Target': [0.5, 1.5, 2.0, 3.245, 0.5]
    }

    df1 = pd.DataFrame(data)

    x1_train, y1_train, x1_test, y1_test = split_train_and_test_data(df1, 'Target', test_size=0.3)
    # print(f"Training data: \n{x1_train}\n")
    # print(f"Training labels: \n{y1_train}\n")
    # print(f"Testing data: \n{x1_test}\n")
    # print(f"Testing labels: \n{y1_test}\n")

    for lab in x1_train:
        print(lab)
    model_rf = Random_Forest_Classifier(x1_train, y1_train)
    model_rf.train()

    predicted_labels1 = model_rf.predict_instance(x1_test)
    print(f"Predicted labels: {predicted_labels1}\n")
    print(f"Accuracy: {get_accuracy(y1_test, predicted_labels1)}\n")

