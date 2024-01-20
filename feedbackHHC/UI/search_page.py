import streamlit as st
import database_handler_forUI as db


def search(keyword):
    # Obținerea datelor din baza de date
    connection = db.DatabaseHandler.create_connection(
        db.Config.DATABASE_NAME,
        db.Config.DATABASE_USER,
        db.Config.DATABASE_PASSWORD,
        db.Config.DATABASE_HOST,
        db.Config.DATABASE_PORT
    )
    cursor = connection.cursor()

    # Căutarea în baza de date
    cursor.execute(f"SELECT state, provider_name, address, city_town, type_of_ownership, quality_of_patient_care_star_rating FROM homecare WHERE address LIKE '%{keyword}%' OR zip_code LIKE '%{keyword}%' OR city_town LIKE '%{keyword}%' OR state LIKE '%{keyword}%'")

    # Obținerea rezultatelor
    results = cursor.fetchall()
    return results

def search_by_provider_name(keyword):
    # Obținerea datelor din baza de date
    connection = db.DatabaseHandler.create_connection(
        db.Config.DATABASE_NAME,
        db.Config.DATABASE_USER,
        db.Config.DATABASE_PASSWORD,
        db.Config.DATABASE_HOST,
        db.Config.DATABASE_PORT
    )
    cursor = connection.cursor()

    keyword_lower = keyword.lower()

    cursor.execute(f"SELECT state, provider_name, address, city_town, type_of_ownership, quality_of_patient_care_star_rating FROM homecare WHERE LOWER(provider_name) LIKE '%{keyword_lower}%'")

    results = cursor.fetchall()
    return results


def display_results_page(results, page_number, results_per_page):
    start_idx = (page_number - 1) * results_per_page
    end_idx = start_idx + results_per_page
    for result in results[start_idx:end_idx]:
        custom_write(result)

def main():
    st.title("Search page")

    keyword1 = st.text_input("Street, ZIP code, city, or state")
    st.text("OR")
    keyword2 = st.text_input("Provider name")


    if st.button("Search"):
        if(keyword1 == "" and keyword2 == ""):
            st.warning("Please enter a search term")
            return
        elif(keyword1 != "" and keyword2 != ""):
            st.warning("Please enter only one search term")
            return
        if(keyword1 == ""):
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
        page_number = st.number_input("Page", min_value=1, max_value=max_pages, value=st.session_state.page_number, step=1)
        st.session_state.page_number = int(page_number)
        st.info(f"Showing {num_results} results")
    else:
        max_pages = 1
        page_number = 1
        st.session_state.page_number = page_number
        st.warning("No results found")


def custom_write(result):
    if len(result) >= 6:
        quality_rating = result[5]
        if quality_rating is not None:
            quality_rating_stars = '⭐' * int(quality_rating)
        else:
            quality_rating_stars = 'N/A'

        html_code = f"""
        <style>
            .result-container:hover {{
                background-color: #f5f5f5;
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

if __name__ == "__main__":
    main()
