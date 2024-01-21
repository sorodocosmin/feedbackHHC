import streamlit as st
import pandas as pd

import sys

sys.path.append('../')

import ui_util
import adaboost_classifier as ab
import naive_bayes_classifier as nb
import neuronal_networks_classifier as nn
import random_forest_classifier as rf


@st.cache_data
def initialize_classifiers():
    print("Initialize classifiers")

    df = pd.read_csv('../Final_data.csv')
    x_train, labels_train, dict_classes_original = (
        ui_util.give_all_datas(df, 'Quality of patient care star rating'))

    m_ab = ab.AdaBoost_Classifier(x_train, labels_train)
    m_ab.train()

    m_nb = nb.NaiveBayesClassifier(x_train, labels_train)
    m_nb.train()

    m_nn = nn.NeuralNetworkClassifier(x_train, labels_train)
    m_nn.train()

    m_rf = rf.Random_Forest_Classifier(x_train, labels_train)
    m_rf.train()

    return m_ab, m_nb, m_nn, m_rf, dict_classes_original


@st.cache_data
def get_column_names():
    """
    Get the column names from Final_data.csv, except the column 'Quality of patient care star rating'
    :return: list of column names
    """
    df = pd.read_csv('../Final_data.csv')
    df = df.drop(['Quality of patient care star rating'], axis=1)
    return list(df.columns)


model_ab, model_nb, model_nn, model_rf, dict_class_original = initialize_classifiers()
df_col_names = get_column_names()


# # Function to estimate quality using a pre-implemented classification algorithm
def estimate_quality(data, classifier):
    prediction = classifier.predict(data)
    print("Prediction: ", prediction)
    return prediction


def get_explanation(name):
    expl = ""
    if "DTC" in name:
        expl += "(DTC = Discharge to community)"

    if "DTC Denominator" in name:
        expl += "(Denominator = Number of eligile stays for DTC measure)"

    if "Risk-Standardized Rate" in name:
        expl += "(Risk-Standardized Rate = The rate of discharges of high-risk patients)"

    return expl


def get_user_input():
    """
    Creates several input fields for the user to input data
    :return: a dictionary containing the user input for each field,
    """
    global df_col_names

    user_input = dict()
    for name in df_col_names:
        insert_newline()
        # User input
        if name.startswith('Offers'):  # only two options for user input
            user_input[name] = st.checkbox(name + '?' + ' \n' + ' Check the box if yes.')

        elif name.startswith('How often'):  # float number from 0 to 100
            user_input[name] = st.number_input(name + '? \n' + ' Enter a percent from 0.0 to 100.',
                                               min_value=0.0, max_value=100.0, step=0.1)
        elif name.startswith('Changes'):  # number from 0 to 2
            user_input[name] = st.number_input(name + '? \n' + ' Enter a number from 0.0 to 1.',
                                               min_value=0.0, max_value=2.0, step=0.1)
        elif name.endswith('Rate'):  # number from 0 to 100
            user_input[name] = st.number_input(name + get_explanation(name) + '? Enter a percent from 0.0 to 100.',
                                               min_value=0.0, max_value=100.0, step=0.1)
        else:  # number from 0
            user_input[name] = st.number_input(name + get_explanation(name) + '? \n' + ' Enter a positive number.',
                                               min_value=0.0, step=0.1)

    insert_newline()
    algorithm = st.selectbox("Select the classification algorithm",
                             ("Random Forest Classifier", "Neural Networks Classifier",
                              "AdaBoost Classifier", "Naive Bayes Classifier"))

    return user_input, algorithm


def select_classifier(algorithm):
    """
    Select the classifier based on the algorithm name
    :param algorithm: the name of the algorithm
    :return: the classifier
    """
    global model_ab, model_nb, model_nn, model_rf

    if algorithm == "AdaBoost Classifier":
        return model_ab
    elif algorithm == "Naive Bayes Classifier":
        return model_nb
    elif algorithm == "Neural Networks Classifier":
        return model_nn
    elif algorithm == "Random Forest Classifier":
        return model_rf
    else:
        return None


def get_user_data(user_input):
    """
    Convert the user input to a pandas dataframe
    :param user_input: a dictionary containing the user input
    :return: a pandas dataframe containing the user input
    """
    global df_col_names

    user_data = list()
    for name in df_col_names:
        if type(user_input[name]) is bool:
            if user_input[name] is True:
                user_data.append(1.0)
            else:
                user_data.append(0.0)
        else:
            user_data.append(float(user_input[name]))

    df_data_test = pd.DataFrame([user_data], columns=df_col_names)

    return df_data_test


def insert_newline(times=1):
    """
    Insert a new line
    :param times: the number of new lines to insert
    :return:
    """
    for _ in range(times):
        st.markdown("<br>", unsafe_allow_html=True)


def get_opacities(quality):
    """
    Get the opacities for the stars
    :param quality: the quality value
    :return: a list of opacities
    """
    opacities = []
    colors = []
    for i in range(1, 6):
        if i <= quality:
            opacities.append(1)
            colors.append("gold")
        else:
            r = quality - i
            if r <= -1:
                r = 0.5
                colors.append("grey")
            else:
                r = 0.5
                colors.append("gold")
            opacities.append(r)

    return opacities, colors

# Streamlit app
def main_page():
    """
    Main page of the app
    :return:
    """
    global model_ab, model_nb, dict_class_original, df_col_names

    st.title("Quality Estimator")

    # insert an explanation of how this estimator works and how to use it
    st.warning("This is a quality estimator for HealInsight.\n"
"It uses a machine learning algorithm to estimate the quality of a HHC based on the user input.\n"
"The user input is a set of features that describe the best the service.\n")


    st.warning("""Then a classification will estimated the quality of the HHC.
                The classification algorithms are trained on a public dataset containing information about HHCs.
                The dataset is available at 'https://data.cms.gov/provider-data/dataset/6jpm-sxkc'.""")
    insert_newline(2)


# User input
    st.markdown(
        """
            <div style="border: 2px solid #86b6f6; padding: 10px;">
                <p>Next you'll have to provide some information about the HHC service you are testing.</p>
            </div>
        """,
        unsafe_allow_html=True
    )

    user_input, algorithm = get_user_input()

    classifier = select_classifier(algorithm)


    insert_newline(3)
    # Button to trigger quality estimation
    if st.button("Estimate Quality"):
        test_data = get_user_data(user_input)

        # Estimate quality
        prediction = estimate_quality(test_data, classifier)[0]

        # Get the original quality value
        quality = dict_class_original[prediction]
        print(f"Quality: {quality}")

        st.header("Quality Estimation Result:")
        st.write(f"The estimated quality is: {quality}")

        num_stars = int(round(quality))  # Assuming quality_prediction is between 0 and 1
        star_size = 100  # Adjust the size of the stars

        opacities, colors = get_opacities(quality)

        # Displaying stars with specified color and opacity
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; font-size: {star_size}px;">
                <span style="color: {colors[0]}; opacity: {opacities[0]}; ">★</span>
                <span style="color: {colors[1]}; opacity: {opacities[1]};">★</span>
                <span style="color: {colors[2]}; opacity: {opacities[2]};">★</span>
                <span style="color: {colors[3]}; opacity: {opacities[3]};">★</span>
                <span style="color: {colors[4]}; opacity: {opacities[4]};">★</span>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main_page()
