import psycopg2
import csv
from psycopg2 import sql

class Config:
    """
    A class used to store the configuration of the database.

    Attributes:
    -----------
        DATABASE_NAME: the name of the database
        DATABASE_USER: the user of the database
        DATABASE_PASSWORD: the password of the database
        DATABASE_HOST: the host of the database
        DATABASE_PORT: the port of the database
    """
    DATABASE_NAME = 'HomeCare'
    DATABASE_USER = 'postgres'
    DATABASE_PASSWORD = '123456'
    DATABASE_HOST = 'localhost'
    DATABASE_PORT = '5432'

class DatabaseHandler:
    @staticmethod
    def create_connection(db_name, db_user, db_password, db_host, db_port):
        connection = None

        try:
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port
            )
            print("Connection to PostgreSQL DB successful")
        except Exception as e:
            print(f"The error '{e}' occurred")
        
        return connection
    
    @staticmethod
    def create_database(connection):
        cursor = connection.cursor()

        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS homecare(
                            latitude DECIMAL(9, 7),
                            longitude DECIMAL(10, 7),
                            state VARCHAR(2),
                            cms_certification_number INTEGER,
                            provider_name VARCHAR(355),
                            address VARCHAR(355),
                            city_town VARCHAR(355),
                            zip_code VARCHAR(10),
                            telephone_number VARCHAR(15),
                            type_of_ownership VARCHAR(355),
                            offers_nursing_care BOOLEAN,
                            offers_physical_therapy BOOLEAN,
                            offers_occupational_therapy BOOLEAN,
                            offers_speech_pathology BOOLEAN,
                            offers_medical_social_services BOOLEAN,
                            offers_home_health_aide_services BOOLEAN,
                            certification_date DATE,
                            quality_of_patient_care_star_rating DECIMAL(3, 1),
                            footnote_for_quality_of_patient_care_star_rating VARCHAR(355),
                            how_often_care_begins_timely DECIMAL(4, 1),
                            footnote_for_how_often_care_begins_timely VARCHAR(355),
                            how_often_flu_shot_determined DECIMAL(4, 1),
                            footnote_for_how_often_flu_shot_determined VARCHAR(355),
                            how_often_patients_better_walking DECIMAL(4, 1),
                            footnote_for_how_often_patients_better_walking VARCHAR(355),
                            how_often_patients_better_bed DECIMAL(4, 1),
                            footnote_for_how_often_patients_better_bed VARCHAR(355),
                            how_often_patients_better_bathing DECIMAL(4, 1),
                            footnote_for_how_often_patients_better_bathing VARCHAR(355),
                            how_often_patients_breathing_improved DECIMAL(4, 1),
                            footnote_for_how_often_patients_breathing_improved VARCHAR(355),
                            how_often_patients_better_drugs_by_mouth DECIMAL(4, 1),
                            footnote_for_how_often_patients_better_drugs_by_mouth VARCHAR(355),
                            how_often_admitted_to_hospital DECIMAL(4, 1),
                            footnote_for_how_often_admitted_to_hospital VARCHAR(355),
                            how_often_urgent_care_needed DECIMAL(4, 1),
                            footnote_for_how_often_urgent_care_needed VARCHAR(355),
                            changes_in_skin_integrity DECIMAL(4, 1),
                            footnote_changes_in_skin_integrity VARCHAR(355),
                            how_often_actions_timely DECIMAL(4, 1),
                            footnote_for_how_often_actions_timely VARCHAR(355),
                            percent_of_residents_falls_with_major_injury DECIMAL(5, 2),
                            footnote_for_percent_of_residents_falls_with_major_injury VARCHAR(355),
                            application_of_percent_of_long_term_care DECIMAL(5, 2),
                            footnote_for_application_of_percent_of_long_term_care VARCHAR(355),
                            dtc_numerator INTEGER,
                            dtc_denominator INTEGER,
                            dtc_observed_rate DECIMAL(5, 2),
                            dtc_risk_standardized_rate DECIMAL(5, 2),
                            dtc_risk_standardized_rate_lower_limit DECIMAL(5, 2),
                            dtc_risk_standardized_rate_upper_limit DECIMAL(5, 2),
                            dtc_performance_categorization VARCHAR(355),
                            footnote_for_dtc_risk_standardized_rate VARCHAR(355),
                            ppr_numerator INTEGER,
                            ppr_denominator INTEGER,
                            ppr_observed_rate DECIMAL(5, 2),
                            ppr_risk_standardized_rate DECIMAL(5, 2),
                            ppr_risk_standardized_rate_lower_limit DECIMAL(5, 2),
                            ppr_risk_standardized_rate_upper_limit DECIMAL(5, 2),
                            ppr_performance_categorization VARCHAR(355),
                            footnote_for_ppr_risk_standardized_rate VARCHAR(355),
                            pph_numerator INTEGER,
                            pph_denominator INTEGER,
                            pph_observed_rate DECIMAL(5, 2),
                            pph_risk_standardized_rate DECIMAL(5, 2),
                            pph_risk_standardized_rate_lower_limit DECIMAL(5, 2),
                            pph_risk_standardized_rate_upper_limit DECIMAL(5, 2),
                            pph_performance_categorization VARCHAR(355),
                            footnote_for_pph_risk_standardized_rate VARCHAR(355),
                            medicare_spending_episode_care DECIMAL(10, 2),
                            footnote_for_medicare_spending_episode_care VARCHAR(355),
                            episodes_to_calculate_medicare_spending INTEGER
            )''')
            connection.commit()
            cursor.close()

            print("Database created successfully")

        except Exception as e:
            print(f"The error '{e}' occurred")
    @staticmethod
    def clean_numeric_value(value):
    # Remove commas from numeric values
        if(value == '-'):
            return None
        return value.replace(',', '') if value else None

    @staticmethod
    def insert_data(connection):
        cursor = connection.cursor()
    
        with open('C:\\Users\\Andrei\\OneDrive\\Desktop\\AI\\Project\\feedbackHHC\\feedbackHHC\\HH_Provider_Oct2023.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            
            for row in reader:
                cleaned_row = [DatabaseHandler.clean_numeric_value(value) for value in row]

                placeholders = ', '.join(['%s' for _ in cleaned_row])
                
                insert_query = sql.SQL("INSERT INTO {} VALUES ({})").format(
                    sql.Identifier('homecare'),
                    sql.SQL(placeholders)
                )
                
                cursor.execute(insert_query, cleaned_row)
                
            connection.commit()
            cursor.close()



if __name__ == "__main__":
    connection = DatabaseHandler.create_connection(
        Config.DATABASE_NAME,
        Config.DATABASE_USER,
        Config.DATABASE_PASSWORD,
        Config.DATABASE_HOST,
        Config.DATABASE_PORT
    )
    DatabaseHandler.create_database(connection)
    DatabaseHandler.insert_data(connection)