import streamlit as st

st.set_page_config(
    page_title="HealInsight",
    page_icon=":hospital:",
    layout="wide"
)

from streamlit_option_menu import option_menu
import json
import database_handler_forUI as db
from profile_page import profile_page
from estimate_quality import main_page
import pandas as pd
import altair as alt
import numpy as np

from behind_the_scenes import behind_the_scenes_page

#Layout

 

st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def search(keyword):
    connection = db.DatabaseHandler.create_connection(
        db.Config.DATABASE_NAME,
        db.Config.DATABASE_USER,
        db.Config.DATABASE_PASSWORD,
        db.Config.DATABASE_HOST,
        db.Config.DATABASE_PORT
    )
    cursor = connection.cursor()

    keyword_lower = keyword.lower()

    cursor.execute(f"SELECT state, provider_name, address, city_town, type_of_ownership, quality_of_patient_care_star_rating, cms_certification_number FROM homecare WHERE LOWER(address) LIKE '%{keyword_lower}%' OR LOWER(zip_code) LIKE '%{keyword_lower}%' OR LOWER(city_town) LIKE '%{keyword_lower}%' OR LOWER(state) LIKE '%{keyword_lower}%'")
    results = cursor.fetchall()
    return results

def search_by_provider_name(keyword):

    connection = db.DatabaseHandler.create_connection(
        db.Config.DATABASE_NAME,
        db.Config.DATABASE_USER,
        db.Config.DATABASE_PASSWORD,
        db.Config.DATABASE_HOST,
        db.Config.DATABASE_PORT
    )
    cursor = connection.cursor()

    keyword_lower = keyword.lower()

    cursor.execute(f"SELECT state, provider_name, address, city_town, type_of_ownership, quality_of_patient_care_star_rating, cms_certification_number FROM homecare WHERE LOWER(provider_name) LIKE '%{keyword_lower}%'")
    results = cursor.fetchall()
    return results


def display_results_page(results, page_number, results_per_page):
    start_idx = (page_number - 1) * results_per_page
    end_idx = start_idx + results_per_page
    for result in results[start_idx:end_idx]:
        key = f"view_profile_{result[6]}"
        custom_write(result)
        if st.button(f"View Profile for {result[1]}", key=key):
            profile_page(result[6])

def custom_write(result):
    if len(result) >= 7:
        quality_rating = result[5]
        if quality_rating is not None:
            quality_rating_stars = '⭐' * int(quality_rating)
        else:
            quality_rating_stars = 'N/A'

        html_code = f"""
        <style>
            .result-container:hover {{
                background-color: #B4D4FF;
                cursor: pointer;
            }}
        </style>
        <div class="result-container" style='padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin: 15px 0; transition: background-color 0.3s;'>
            <h2 style='margin-bottom: 10px;'>{result[1]}</h2>
            <p><b>State:</b> {result[0]}</p>
            <p><b>City/Town:</b> {result[3]}</p>
            <p><b>Address:</b> {result[2]}</p>
            <p><b>Type of Ownership:</b> {result[4]}</p>
            <p><b>Quality Rating:</b> {quality_rating_stars}</p>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    else:
        st.warning("Invalid result format. Unable to display.")

def create_adaboost_chart():
        st.markdown("The hyperparameters used for AdaBoostClassifier are the following:")
        code = '''
        possible_parameters_ab = {
                "n_estimators": [10, 25, 50, 100, 200, 250, 500],  # which is the number of iterations
                # learning rate is mutiplied with the weight of the distribution for each iteration
                "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0r],
                "estimator": [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2, criterion="gini"),
                              DecisionTreeClassifier(max_depth=2, criterion='entropy')]

            }
        '''


        st.code(code, language='python')
        data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Time(in seconds)": [77.70032358169556, 858.1552698612213]
        }

        chart_data = pd.DataFrame(data)
        chart1 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
            x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
            color=alt.Color('Algorithms:N', legend=None)
        ).properties(width=500, height=400, title="Time took for hyperparameter tuning (AdaBoostClassifier)")

        st.altair_chart(chart1, use_container_width=True)

        data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Best Score": [0.5379636937647988, 0.692583935401615]
        }

        chart_data = pd.DataFrame(data)
        chart2 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
            x=alt.X('Best Score:Q', axis=alt.Axis(title='Best Score')),
            color=alt.Color('Algorithms:N', legend=None)
        ).properties(width=500, height=400, title="Best Score for hyperparameters tunning (AdaBoostClassifier)")

        st.altair_chart(chart2, use_container_width=True)

        data = {
            "Accuracy": ["Before", "After"],
            "Score": [0.366006600660066, 0.7138613861386138]
        }

        chart_data = pd.DataFrame(data)
        chart3 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Accuracy:N', axis=alt.Axis(title='Accuracy')),
            x=alt.X('Score:Q', axis=alt.Axis(title='Score')),
            color=alt.Color('Accuracy:N', legend=None)
        ).properties(width=500, height=400, title="Accuracy testing before and after hyperparameter tuning (AdaBoostClassifier)")

        max_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='red', strokeWidth=2).encode(y='y')
        final_chart3 = chart3 + max_line
        st.altair_chart(final_chart3, use_container_width=True)

        data = {
            "When": ["Before", "After"],
            "Time in seconds": [0.358627033233642, 4.953497314453125]
        }

        chart_data = pd.DataFrame(data)
        chart4 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('When:N', axis=alt.Axis(title='')),
            x=alt.X('Time in seconds:Q', axis=alt.Axis(title='Time in seconds')),
            color=alt.Color('When:N', legend=None)
        ).properties(width=500, height=400, title="Time took for hyperparameter tuning (AdaBoostClassifier)")

        st.altair_chart(chart4, use_container_width=True)

        return chart1, chart2, final_chart3, chart4

def create_comparation_chart():
        data = {
            "Models": ["AdaBoost", "RandomForest", "NaiveBayes", "NeuronalNetwork"],
            "Score": [0.7138613861386138, 0.7608910891089109, 0.6671617161716171,  0.8353135313531354]
        }

        chart_data = pd.DataFrame(data)
        chart1 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Models:N', axis=alt.Axis(title='Models')),
            x=alt.X('Score:Q', axis=alt.Axis(title='Score')),
            color=alt.Color('Models:N', legend=None)
        ).properties(width=500, height=400, title="Accuracy testing for all models")

        max_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='red', strokeWidth=2).encode(y='y')
        final_chart1 = chart1 + max_line
        st.altair_chart(final_chart1, use_container_width=True)

        data = {
            "Models": ["AdaBoost", "RandomForest", "NaiveBayes", "NeuronalNetwork"],
            "Time(in seconds)": [4.953497314453125, 4.612986588478089,  0.017272686958312987, 14.307233572006226]
        }

        chart_data = pd.DataFrame(data)
        chart2 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Models:N', axis=alt.Axis(title='Models')),
            x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
            color=alt.Color('Models:N', legend=None)
        ).properties(width=500, height=400, title="Time took for training all models")

        st.altair_chart(chart2, use_container_width=True)

        data = {
            "Models": ["AdaBoost", "RandomForest", "NaiveBayes", "NeuronalNetwork"],
            "Time(in seconds)": [503.62843465805054, 541.2269668579102,  0.5620238780975342, 433.14596819877625]
        }

        chart_data = pd.DataFrame(data)
        chart3 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Models:N', axis=alt.Axis(title='Models')),
            x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
            color=alt.Color('Models:N', legend=None)
        ).properties(width=500, height=400, title="Time for Cross-Validation (Fold=100) for all models")

        st.altair_chart(chart3, use_container_width=True)

        data = {
            "Models": ["AdaBoost", "RandomForest", "NaiveBayes", "NeuronalNetwork"],
            "Score": [0.5522222222222222, 0.762263157894737, 0.6579532163742691, 0.8265204678362573]
        }

        chart_data = pd.DataFrame(data)
        chart4 = alt.Chart(chart_data).mark_bar().encode(
            y=alt.Y('Models:N', axis=alt.Axis(title='Models')),
            x=alt.X('Score:Q', axis=alt.Axis(title='Score')),
            color=alt.Color('Models:N', legend=None)
        ).properties(width=500, height=400, title="Accuracy at Cross-Validation (Fold=100) for all models")

        max_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='red', strokeWidth=2).encode(y='y')
        final_chart4 = chart4 + max_line
        st.altair_chart(final_chart4, use_container_width=True)

        return final_chart1, chart2, chart3, final_chart4

def create_randomforest_chart():
    st.markdown("The hyperparameters used for RandomForestClassifier are the following:")
    code = '''
            possible_parameters_rf = {
            "n_estimators": [10, 100, 200, 500, 750, 1_000],
            "criterion": ["gini", "entropy"],
            # the maximum depth of the tree is represented by the nr of features which is 12
            "max_depth": [None, 5, 7, 8, 9, 10],
            # Bootstrap means that instead of training on all the observations,
            # each tree of RF is trained on a subset of the observations
            "bootstrap": [True, False]
        }
        '''
    st.code(code, language='python')
    data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Time(in seconds)": [141.7917718887329, 2126.8367805480957]
        }

    chart_data = pd.DataFrame(data)
    chart1 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Time took for hyperparameter tuning (RandomForestClassifier)")

    st.altair_chart(chart1, use_container_width=True)

    data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Best Score": [0.7612925748284864, 0.7673517090644162]
        }

    chart_data = pd.DataFrame(data)
    chart2 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Best Score:Q', axis=alt.Axis(title='Best Score')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Best Score for hyperparameters tuning (RandomForestClassifier)")

    st.altair_chart(chart2, use_container_width=True)

    data = {
            "Algorithms": ["After", "Before"],
            "Accuracy": [0.7608910891089109, 0.7315181518151814]
        }

    chart_data = pd.DataFrame(data)
    chart3 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Accuracy:Q', axis=alt.Axis(title='Accuracy')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Accuracy testing before and after hyperparameter tuning (RandomForestClassifier)")

    st.altair_chart(chart3, use_container_width=True)

    data = {
            "Algorithms": ["After", "Before"],
            "Time(in seconds)": [4.612986588478089, 0.7774333000183106]
        }

    chart_data = pd.DataFrame(data)
    chart4 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Time took on training before and after hyperparameter tuning (RandomForestClassifier)")

    st.altair_chart(chart4, use_container_width=True)

    return chart1, chart2, chart3, chart4


def create_neuralnetwork_chart():
    data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Time(in seconds)": [139.5538055896759, 1297.5675098896027]
        }

    chart_data = pd.DataFrame(data)
    chart1 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Time took for hyperparameter tuning (Neuronal Network)")

    st.altair_chart(chart1, use_container_width=True)

    data = {
            "Algorithms": ["RandomSearch", "GridSearch"],
            "Best Score": [0.849826968611499, 0.8614048934490924]
        }

    chart_data = pd.DataFrame(data)
    chart2 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Best Score:Q', axis=alt.Axis(title='Best Score')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Best Score for hyperparameters tuning (Neuronal Network)")

    st.altair_chart(chart2, use_container_width=True)

    data = {
            "Algorithms": ["After", "Before"],
            "Accuracy": [0.8592409240924093, 0.8353135313531354]
        }

    chart_data = pd.DataFrame(data)
    chart3 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Accuracy:Q', axis=alt.Axis(title='Accuracy')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Accuracy testing before and after hyperparameter tuning (Neuronal Network)")

    st.altair_chart(chart3, use_container_width=True)

    data = {
            "Algorithms": ["After", "Before"],
            "Time(in seconds)": [14.307233572006226, 4.309966945648194]
        }

    chart_data = pd.DataFrame(data)
    chart4 = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Algorithms:N', axis=alt.Axis(title='Algorithms')),
        x=alt.X('Time(in seconds):Q', axis=alt.Axis(title='Time (in seconds)')),
        color=alt.Color('Algorithms:N', legend=None)
    ).properties(width=500, height=400, title="Time took on training before and after hyperparameter tuning (Neuronal Network)")

    st.altair_chart(chart4, use_container_width=True)

    return chart1, chart2, chart3, chart4

def create_collapsible_chart(title, create_chart_function):
    with st.expander(title):
        create_chart_function()

#Options Menu
with st.sidebar:
    selected = option_menu('HealInsight', ["Home", 'Search', 'Predict','Behind the Scenes', 'About'],
        icons=['play-btn','search','','star', 'info-circle'],menu_icon='hospital', default_index=0)

#Home Page
if selected == "Home":
    st.title('Welcome to HealInsight')
    st.subheader('*A new tool designed to revolutionize your search for the perfect home health services!*')

    st.divider()

    #Use Cases
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('Here you can:')
            st.markdown(
                """
                <div style="font-size:22px">
                - <i> Discover and select home health services</i><br><br>
                - <i> Predict ratings for home health services</i><br><br>
                - <i> Uncover the app's behind-the-scenes</i><br><br>
                - <i> Interested in just exploring and learning?</i><br><br>
                </div>
                """,
                unsafe_allow_html=True
            )
        gif_path = "https://i.pinimg.com/originals/ea/7f/2d/ea7f2dd47969349da148ea0b4ec56815.gif"
        with col2:
            st.markdown(f'<img src="{gif_path}" alt="gif" width="500">', unsafe_allow_html=True)
    st.divider()

    st.header('Tutorial Video')
    # video_file = open('Similo_Tutorial3_compressed.mp4', 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)
    
#Search Page
if selected == 'Search':
    st.title("Search page")

    keyword1 = st.text_input("Street, ZIP code, city, or state")
    st.text("OR")
    keyword2 = st.text_input("Provider name")

    if st.button("Search"):
        if (keyword1 == "" and keyword2 == ""):
            st.warning("Please enter a search term")
        elif (keyword1 != "" and keyword2 != ""):
            st.warning("Please enter only one search term")
        if (keyword1 == ""):
            results = search_by_provider_name(keyword2)
        else:
            results = search(keyword1)
        st.session_state.results = results
        st.session_state.page_number = 1

    results = st.session_state.get("results", [])
    if results:
        results_per_page = 5
        num_results = len(results)

        display_results_page(results, st.session_state.page_number, results_per_page)

        max_pages = max((num_results - 1) // results_per_page + 1, 1)
        page_number = st.number_input("Page", min_value=1, max_value=max_pages, value=st.session_state.page_number,
                                      step=1)
        st.session_state.page_number = int(page_number)
        st.info(f"Showing {num_results} results")
    else:
        max_pages = 1
        page_number = 1
        st.session_state.page_number = page_number
        st.warning("No results found")


if selected == 'Predict':
    main_page()



if selected == 'Behind the Scenes':

    behind_the_scenes_page()
    create_collapsible_chart("Chart 1 - RandomForestClassifier", create_randomforest_chart)
    create_collapsible_chart("Chart 2 - AdaBoostClassifier", create_adaboost_chart)
    create_collapsible_chart("Chart 3 - NeuronalNetwork", create_neuralnetwork_chart)
    create_collapsible_chart("Chart 4 - Comparations", create_comparation_chart)

if selected == 'About':
    st.title('About the Project')

    st.markdown(
        "<p style='font-size: 22px;'>Our team embarked on a comprehensive Artificial Intelligence project, delving into the realm of healthcare to analyze patient sentiments and opinions regarding the treatment provided by home healthcare service agencies.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 22px;'>The primary focus of our investigation was to assess the quality of patient care offered by these agencies. To conduct this analysis, we leveraged a rich dataset available at <a href='https://data.cms.gov/provider-data/dataset/6jpm-sxkc' target='_blank'>CMS Provider Data</a>. This dataset encompasses a wealth of information pertaining to agencies that provide home care services, with a specific emphasis on patient care quality as the target variable.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 22px;'>Throughout the project, our team aimed to contribute valuable insights to the healthcare domain, leveraging advanced Artificial Intelligence techniques for a more nuanced understanding of patient experiences and healthcare service quality.</p>",
        unsafe_allow_html=True
    )

    st.divider()
    st.title('Project Team')

    with st.container():
        col1, col2 = st.columns(2)
        col1.markdown("<li style='font-size: 24px;'>Bodnar Alina-Florina</li>", unsafe_allow_html=True)
        col1.markdown("<li style='font-size: 24px;'>Galan Andrei-Iulian</li>", unsafe_allow_html=True)
        col1.markdown("<li style='font-size: 24px;'>Ignat Gabriel-Andrei</li>", unsafe_allow_html=True)
        col1.markdown("<li style='font-size: 24px;'>Șorodoc Tudor-Cosmin</li>", unsafe_allow_html=True)
        col1.markdown("<li style='font-size: 24px;'>Ungurean Ana-Maria</li>", unsafe_allow_html=True)
        col1.markdown("</ul>", unsafe_allow_html=True)

        col2.markdown("<li style='font-size: 24px;'><strong>Group:</strong> 3A4</li>", unsafe_allow_html=True)
        col2.markdown("<li style='font-size: 24px;'><strong>Faculty:</strong> Faculty of Computer Science</li>",unsafe_allow_html=True)
    st.divider()







