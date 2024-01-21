import streamlit as st
from streamlit_option_menu import option_menu
import json
import database_handler_forUI as db
from profile_page import profile_page
from estimate_quality import main_page

#Layout
st.set_page_config(
    page_title="HealInsight",
    page_icon=":hospital:",
    layout="wide"
)

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
    st.markdown("Behind the Scenes")



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







