import streamlit as st
import pandas as pd

def behind_the_scenes_page():
    # Page title



    st.title("Project Report: Enhancing Patient Care Quality Prediction in Home Health Agencies")

    # Table of Contents
    st.markdown("## Table of Contents")
    chapters = ["Introduction", "Data Preprocessing", "Feature Selection", "Machine Learning Models"]

    # Create links for each chapter
    for chapter in chapters:
        st.markdown(f"- [{chapter}](#{chapter.lower().replace(' ', '-')})")

    st.markdown("---")
    # Introduction
    introduction_part()

    st.markdown("---")

    # Data Preprocessing
    preprocessing_part()

    st.markdown("---")

    # Feature Selection
    feature_selection_part()

    st.markdown("---")

    # Model Evaluation
    machine_learning_models_part()
    st.markdown("---")


def introduction_part():
    st.markdown("<a name='introduction'></a>", unsafe_allow_html=True)
    st.markdown("## Introduction")
    st.write("""
    Healthcare quality prediction is a critical task in ensuring optimal patient outcomes.
    In this report, we present a meticulous exploration of our predictive modeling approach applied to home health agencies.
    The journey unfolds from **data preprocessing**, **feature selection**, to the deployment and evaluation of advanced **machine learning models**.
    """)

def preprocessing_part():
    st.markdown("<a name='data-preprocessing'></a>", unsafe_allow_html=True)
    st.markdown("## Data Preprocessing")
    st.write("""
        In the realm of data preprocessing, meticulous attention to detail transforms raw datasets into a refined and
        structured form, setting the stage for effective machine learning model training. 
        This crucial phase involves a series of operations aimed at refining and organizing the data, ensuring that it
        aligns with the requirements of machine learning algorithms.

        Let's delve into some specific preprocessing steps applied to our dataset:
        """)

    st.subheader("Geospatial Enhancement: Transforming Addresses into Coordinates")
    st.write("""
        To bring a geographical perspective into our dataset, we utilized the Google Maps API to convert textual addresses,
        such as `State`, `ZIP Code`, `City/Town`, and `Address`, into specific latitude and longitude coordinates.
        Giving each location a unique set of geographical coordinates, it will make it easier for our machine learning model
        to understand the spatial relationships between different places.

        **Example:** AK	ANCHORAGE 27001 4001 DALE STREET, SUITE 101 ➡️ 61.1839967   -149.8175083	

        """)

    st.subheader("Transforming Dates info Numerical Values")
    st.write("""
        To facilitate the interpretation of dates, we transformed them into numerical values.
        The `Date Certified` column was transformed into the number of days since the agency was certified.
        This transformation provides a numerical measure that reflects how long ago the certification was acquired.
        The rationale behind this approach is to assign larger values to agencies that obtained their certification earlier,
        as it is assumed that agencies with longer certification histories may have accumulated more experience and expertise.
        """)

    st.subheader("Transforming Text into Vectors")
    st.write("""
    
    In order to incorporate the textual information from the `Provider Name` column into our machine learning model,
    a technique called **Doc2Vec** was applied (Doc2Vec is a neural network-based approach that learns the distributed
    representation of documents. It is an unsupervised learning technique that maps each document to a fixed-length vector
    in a high-dimensional space) This method converts text into numerical vectors, allowing the model to process and analyze textual data effectively.
    You can learn more about Doc2Vec [here](https://radimrehurek.com/gensim/models/doc2vec.html).
    
    *Procedure:*
    - Each entry in the `Provider Name` column was treated as a separate document.
    - The trained model was used to infer vectors for each document in the "Provider Name" column.
    - Result Integration:
        * The inferred vectors were incorporated into the dataset as new columns labeled as `Provider_Name_1`, `Provider_Name_2`,
    and so forth. 
        * The original `Provider Name` column was dropped.
    
    """)

    st.subheader("Label Encoding")
    st.write("""
    To optimize model performance and facilitate streamlined data interpretation, we strategically applied label encoding to specific columns associated with performance categorization, offered services, and type of ownership.

    **Performance Categorization:**
    Achieve a numeric representation of performance categorizations for enhanced model understanding.
    *Columns Affected:* `DTC Performance Categorization`, `PPR Performance Categorization`, `PPH Performance Categorization`.
    
    *Mapping:*
    - "Worse Than National Rate" → 0
    - "Same As National Rate" → 1
    - "Better Than National Rate" → 2
    - "Not Available" → 3
    - "-" → 4.

    **Offered Services:**
    Quantify the availability of various services in a numeric manner.
    *Columns Affected:* `Offers Nursing Care Services`, `Offers Physical Therapy Services`, `Offers Occupational Therapy Services`, 
    `Offers Speech Pathology Services`, `Offers Medical Social Services`, `Offers Home Health Aide Services`.
    
    *Mapping:*
    - "Yes" → 1
    - "No" → 0

    **Type of Ownership:**
    *Objective:* Convert diverse ownership types into a numerical representation.
    *Columns Affected:* `Type of Ownership`.
    
    *Mapping:*
    - "PROPRIETARY" → 0
    - "VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION" → 1
    - GOVERNMENT - STATE/COUNTY" → 2
    - ...etc.
    """)

    st.subheader("Removing Irrelevant Features")
    st.write("""
    *Columns Removed:*
    1. Columns with `Footnote` in their name: These columns were excluded from the dataset because they consistently
    contained either "-" or the text "The number of patients is too small to report."
    Since these footnotes provide little meaningful information, retaining them would only introduce noise into the model.
    
    *Impact on Model:*
    Introducing __noise__ in the context of machine learning refers to including irrelevant or uninformative data in the dataset,
    which can negatively impact the model's ability to learn meaningful patterns.
    __Noise__ can "distract" the model from recognizing important features, leading to overfitting or decreased predictive performance.
    In this case, as columns with footnotes contain repetitive and non-informative content, they were retained.

    2. `Telephone Number` and `CMS Certification Number`: These columns were strategically eliminated as they contained
    unique identifiers for each record. Including them in the predictive modeling process could lead to _overfitting_,
    where the model memorizes the training data instead of learning general patterns.
    
    *Impact on Model:*
    Removing these columns ensures that the model focuses on relevant features, reducing the risk of overfitting and enhancing its ability to generalize to new data.
    """)

    st.subheader("Removing Rows without Ratings (labels)")
    st.write("""
    
    To maintain the integrity of our dataset and facilitate effective model training,
    a crucial step involves removing rows where the target variable, the `Quality of patient care star rating` is missing.
    The absence of this essential label could hinder the model's ability to learn and make accurate predictions.
    
    This quality check revealed that initially, we had **11739** rows.
    However, after removing instances with missing quality ratings, we were left with a refined dataset comprising **7852** rows.
    
    """)

    st.subheader("Removing Outliers")
    st.write("""
    Outliers, data points significantly deviating from the general trend, can impact the performance of machine learning models.
    To systematically detect and remove outliers, we employed the **Interquartile Range (IQR)** method.
    *Interquartile Range (IQR) method :*
    - Calculate the first quartile (Q1) and the third quartile (Q3) for the specific column.
    - Determine the Interquartile Range (IQR) as the difference between Q3 and Q1.
    - *Lower Bound* = Q1 - 1.5 &times; IQR
    - *Upper Bound* = Q3 + 1.5 &times; IQR
        * The choice of **1.5** is a rule of thumb that has been widely adopted in statistics.
        It provides a balance between detecting potential outliers and avoiding the exclusion of too many data points that may still be valid.
    - Remove all data points outside the lower and upper bounds.
    
    Now, we will illustrate through a graph how the numbers of rows reduced after removing outliers for each column.
    
    """)

    with st.expander("Legend - Column Names to Numbers"):
        st.write("How often the home health team began their patients' care in a timely manner ⟶ Col 1")
        st.write(
            "How often the home health team determined whether patients received a flu shot for the current flu season ⟶ Col 2")
        st.write("How often patients got better at walking or moving around ⟶ Col 3")
        st.write("How often patients got better at getting in and out of bed ⟶ Col 4")
        st.write("How often patients got better at bathing ⟶ Col 5")
        st.write("How often patients' breathing improved ⟶ Col 6")
        st.write("How often patients got better at taking their drugs correctly by mouth ⟶ Col 7")
        st.write("How often home health patients had to be admitted to the hospital ⟶ Col 8")
        st.write(
            "How often patients receiving home health care needed urgent, unplanned care in the ER without being admitted ⟶ Col 9")
        st.write("Changes in skin integrity post-acute care: pressure ulcer/injury ⟶ Col 10")
        st.write("How often physician-recommended actions to address medication issues were completely timely ⟶ Col 11")
        st.write("Percent of Residents Experiencing One or More Falls with Major Injury ⟶ Col 12")
        st.write(
            "Application of Percent of Long Term Care Hospital Patients with an Admission and Discharge Functional Assessment and a Care Plan that Addresses Function ⟶ Col 13")
        st.write("DTC Numerator ⟶ Col 14")
        st.write("DTC Denominator ⟶ Col 15")
        st.write("DTC Observed Rate ⟶ Col 16")
        st.write("DTC Risk-Standardized Rate ⟶ Col 17")
        st.write("DTC Risk-Standardized Rate (Lower Limit) ⟶ Col 18")
        st.write("DTC Risk-Standardized Rate (Upper Limit) ⟶ Col 19")
        st.write("PPR Numerator ⟶ Col 20")
        st.write("PPR Denominator ⟶ Col 21")
        st.write("PPR Observed Rate ⟶ Col 22")
        st.write("PPR Risk-Standardized Rate ⟶ Col 23")
        st.write("PPR Risk-Standardized Rate (Lower Limit) ⟶ Col 24")
        st.write("PPR Risk-Standardized Rate (Upper Limit) ⟶ Col 25")
        st.write("PPH Numerator ⟶ Col 26")
        st.write("PPH Denominator ⟶ Col 27")
        st.write("PPH Observed Rate ⟶ Col 28")
        st.write("PPH Risk-Standardized Rate ⟶ Col 29")
        st.write("PPH Risk-Standardized Rate (Lower Limit) ⟶ Col 30")
        st.write("PPH Risk-Standardized Rate (Upper Limit) ⟶ Col 31")
        st.write("How much Medicare spends on an episode of care at this agency, compared to Medicare spending across all agencies nationally ⟶ Col 32")
        st.write("No. of episodes to calc how much Medicare spends per episode of care at agency, compared to spending at all agencies (national) ⟶ Col 33")

    st.image("plot_outlier.png", width=700)

    st.subheader("Handling Missing Values")
    st.write("""
    
    These missing entries can hinder the performance of our models, and various strategies exist for addressing them.
    One common approach is to replace missing values with the mean of their respective columns.

    **Why Impute with Column Mean?**
    - Imputing missing values with the mean of the column provides a way to maintain the overall statistical characteristics of the data,
    ensuring that the central tendency of the original distribution is conserved. This strategy is effective when
    the absence of values is unrelated to any specific pattern or factor, making it a suitable choice for scenarios
    where data is missing at random
    """)


def feature_selection_part():
    st.markdown("<a name='feature-selection'></a>", unsafe_allow_html=True)
    st.markdown("## Feature Selection")
    st.write("""
    
    Feature selection is a crucial step in refining datasets for machine learning models.
    In this section, we delve into the methodology of feature selection based on correlation matrices,
    particularly using the **Pearson correlation coefficient**.
    
    **The Pearson correlation coefficient**, often denoted as *r*, is a statistical measure that quantifies the
    linear relationship between two variables. Specifically, it assesses how well a change in one variable predicts a
    proportional change in another. The coefficient ranges from `-1 to 1`, signifying the strength and direction of the correlation:
    - **Positive Correlation (r > 0):** As one variable increases, the other tends to increase proportionally.
    - **Negative Correlation (r < 0):** As one variable increases, the other tends to decrease proportionally.
    - **No Correlation (r = 0):** There is no discernible linear relationship between the variables.
    
    You can learn more about the Pearson correlation coefficient [here](https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/).
    
    """)

    st.subheader("Removing Features Weakly Correlated with the Output:")
    st.write("""
    To ensure that our model focuses on the most relevant features, we removed columns with a weak correlation with the output variable.
    Specifically, we removed columns with a correlation coefficient less than 0.2.(the absolute value of the correlation coefficient, of course)
    (This threshold was chosen based on the correlation matrix heatmap, which revealed that the majority of features and 
    also by analyzing the performance of the ML models with different thresholds)
    """)

    st.subheader("Removing Features with High Correlation:")
    st.write("""
    Now, we calculate the correlation matrix for the remaining features and remove features that are highly correlated with each other.
    (with a correlation coefficient greater than 0.8). To keep things clear and avoid repetition,
    we decided to keep only one column from each group of similar ones.
    
    **Example:** if `col1`, `col2` and `col3` are highly correlated( in our case, have a greater coefficient than 0.8),
    we will keep only one of them.
    
    Having unique and independent features makes our dataset cleaner and helps the model make better sense of the data.\
    """)
    st.write("""
    **Finnally**, after applying all the above steps (**Preprocessing** and **Feature Selection**) we started with
    [this dataset](https://github.com/sorodocosmin/feedbackHHC/blob/main/feedbackHHC/HH_Provider_Oct2023.csv)
    and ended up with [this one](). Which is nice and ready to be used for training the ML models.
    """)


def machine_learning_models_part():
    st.markdown("<a name='machine-learning-models'></a>", unsafe_allow_html=True)
    st.markdown("## Machine Learning Models")

    st.write("""

        In our quest to predict and understand patient care quality, we employed several machine learning algorithms including:
        `AdaBoost`, `RandomForest`, `Neural Network`, and `Bayes Naive`.

        Additionally, for `AdaBoost`, `RandomForest`, and `Neural Network`, we conducted hyperparameter tuning to enhance their performance. 
        Now, let's explore the concept of **hyperparameter tuning**.
        """)

    st.subheader("A Dive into Hyperparameter Tuning")
    st.write("""
        Hyperparameter tuning is the process of selecting the optimal values for a machine learning model’s hyperparameters.
        Hyperparameters are settings that control the learning process of the model, such as the learning rate, the 
        number of neurons in a neural network,etc. The goal of hyperparameter tuning is to find the values that lead
        to the best performance on a given task.
        
        What are Hyperparameters?
        - In the context of machine learning, hyperparameters are configuration variables that are set **before**
        the training process of a model begins. They control the learning process itself, rather than being learned
        from the data. Hyperparameters are often used to tune the performance of a model, and they can have a significant
        impact on the model’s accuracy and generalization.
        
        """)

    st.subheader("GridSearch and RandomSearch:")
    st.write("""
    
    Two prevalent methods for hyperparameter tuning are *Grid Search* and *Random Search*. Both techniques systematically
    explore different combinations of hyperparameter values, allowing us to discover the configuration that yields the
    best performance on a given task.
    
    - **Grid Search:**
    This method entails defining a grid of hyperparameter values to be evaluated exhaustively.
    The model is trained and validated for each combination in the grid, helping identify the set of hyperparameters
    that optimize performance.
    - **Random Search:**
     In contrast, Random Search randomly samples hyperparameter combinations from predefined ranges.
    While not as exhaustive as Grid Search, Random Search often requires fewer evaluations and can be more
    efficient in finding good hyperparameter values.
    """)
    


