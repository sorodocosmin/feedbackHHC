import streamlit as st
import psycopg2
import pandas as pd

def get_provider_data(cms_certification_number):
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="123456",
        database="HomeCare"
    )
    cursor = conn.cursor()

    query = f"SELECT * FROM homecare WHERE cms_certification_number = '{cms_certification_number}' LIMIT 1"
    cursor.execute(query)
    provider_data = cursor.fetchone()
    conn.close()

    return provider_data

def display_icon(value):
    return "✅" if value else "❌"

def display_percentage(data, index):
    return f"{data[index]} %" if data[index] is not None else "Not Available"

def profile_page(selected_profile):
    if selected_profile:
        # Afișați pagina de profil pentru furnizorul selectat
        provider_name = selected_profile
        provider_data = get_provider_data(provider_name)

        if provider_data:
            st.markdown(f"# <span style='font-size:60px'>{provider_data[4]}</span>", unsafe_allow_html=True)

            latitude = float(provider_data[0]) if provider_data[0] is not None else None
            longitude = float(provider_data[1]) if provider_data[1] is not None else None

            rating = provider_data[17]
            if rating is not None:
                stars = "⭐" * int(rating)
            else:
                rating = "Not Available"
                stars = ""
            st.markdown(f"<span style='font-size:25px;'>**Quality of Patient Care Rating:** {rating} ({stars})</span>",
                        unsafe_allow_html=True)

            html_code = f"""
                <style>
                    .result-container {{
                        display: flex;
                        justify-content: space-between;
                        padding: 15px;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        margin: 15px 0;
                        transition: background-color 0.3s;
                    }}
                    .location-info {{
                        flex: 1;
                        margin-right: 10px;
                    }}
                    .contact-info {{
                        flex: 1;
                        margin-left: 10px;
                    }}
                    .details-info {{
                        flex: 1;
                        margin-left: 10px;
                    }}
                    .result-container:hover {{
                        background-color: #f5f5f5;
                        cursor: pointer;
                    }}
                    .location-info p,
                    .contact-info p,
                    .details-info p {{
                        font-size: 20px; 
                    }}
                </style>
                <div class="result-container">
                    <div class="location-info">
                        <h2>{"Location Information"}</h2>
                        <p><b>State:</b> {provider_data[2]}</p>
                        <p><b>City/Town:</b> {provider_data[6]}</p>
                        <p><b>Address:</b> {provider_data[5]}</p>
                        <p><b>Zip Code:</b> {provider_data[7]}</p>
                    </div>
                    <div class="contact-info">
                        <h2>{"Contact Information"}</h2>
                        <p><b>Telephone Number:</b> {provider_data[8]}</p>
                        <h2>{"Details"}</h2>
                        <p><b>Type of Ownership:</b> {provider_data[9]}</p>
                        <p><b>Certification Date:</b> {provider_data[16]}</p>
                    </div>
                </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)
            html_code = """<div style="font-size: larger; font-style: italic; font-weight:bold; text-align: center;">
                See map below for location of provider
            </div>"""
            st.markdown(html_code, unsafe_allow_html=True)
            df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(df, color='#FF6868', size=20)

            st.markdown("## Services Provided")
            services_table = pd.DataFrame({
                'Service': ['Nursing care', 'Physical therapy', 'Occupational therapy', 'Speech therapy', 'Medical social services', 'Home health aide'],
                'Provided': [display_icon(provider_data[11]), display_icon(provider_data[12]),
                             display_icon(provider_data[13]), display_icon(provider_data[14]),
                             display_icon(provider_data[15]), display_icon(provider_data[16])]
            })
            st.table(services_table)

            st.markdown("## Managing Daily Activities")
            activities_table = pd.DataFrame({
                'Activity': ['Walking or moving around', 'Getting in and out of bed', 'Bathing', 'Breathing improvement'],
                'Percentage': [display_percentage(provider_data, 23), display_percentage(provider_data, 25),
                               display_percentage(provider_data, 27), display_percentage(provider_data, 29)]
            })
            st.table(activities_table)

            st.markdown("## Preventing Harm")
            harm_table = pd.DataFrame({
                'Prevention': ['Timely start of care', 'Flu shot determination', 'Correct drug administration',
                               'Hospital admissions', 'ER care without admission',
                               'Timely medication issue actions'],
                'Percentage': [display_percentage(provider_data, 19), display_percentage(provider_data, 21),
                               display_percentage(provider_data, 31), display_percentage(provider_data, 33),
                               display_percentage(provider_data, 35), display_percentage(provider_data, 39)]
            })
            st.table(harm_table)

            table_data = {
                "Agency": ["This Agency", "National Average"],
                "Medicare Spending": [display_percentage(provider_data, 35), 1.00]
            }

            st.markdown("## Payment and Value of Care")
            st.table(table_data)

        else:
            st.warning(f"No information found for provider: {provider_name}")


if __name__ == "__main__":
    query_params = st.query_params()
    selected_profile = query_params.get("selected_profile", [None])[0]
    profile_page(selected_profile)
