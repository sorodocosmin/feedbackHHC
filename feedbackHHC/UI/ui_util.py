from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import pandas as pd


def give_all_datas(df, name_output_column):
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
    original_values = df[name_output_column]
    output = df[name_output_column].astype('category').cat.codes  # convert the float numbers to numeric values
    # ex : 0.5 -> 0, 1.0 -> 1, 1.5-> 2, 2.0 -> 3, 2.5 -> 4, 3.0 -> 5, 3.245 ->6, ..etc
    # scaler = StandardScaler()
    # datas_without_output = scaler.fit_transform(datas_without_output)

    dict_class_original_value = {}

    # print("Original: ", original_values)
    # print("Output: ", output)

    for i in range(len(output)):
        dict_class_original_value[output[i]] = original_values[i]

    # print("Dict: ", dict_class_original_value)

    return datas_without_output, output, dict_class_original_value


def get_classes(list_labels):
    """
    Get the classes from the labels
    :param list_labels: a list of the labels
    :return: a list in sorted order of labels
    """
    set_labels = set()
    for label in list_labels:
        set_labels.add(label)

    return sorted(list(set_labels))


