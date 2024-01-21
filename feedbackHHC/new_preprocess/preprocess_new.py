from cmath import nan
from datetime import datetime
import json

import numpy as np
import pandas as pd
import googlemaps

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize  # You might need to install nltk: pip install nltk



class NewPreprocess:
    def __init__(self, df):
        self.__df = df
        self.__results_google_api = {}
        self.__processed_columns = ['Provider Name', 'Latitude', 'Longitude']

    def get_df(self):
        return self.__df

    def get_geo_coords_with_google(self, address):
        # Geocoding request
        api_key = 'secret ; )'
        gmaps = googlemaps.Client(key=api_key)

        geocode_result = gmaps.geocode(address)

        if geocode_result:
            self.__results_google_api[address] = geocode_result
            location = geocode_result[0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            print(f"Coordinates for '{address}': ({latitude}, {longitude})")
            return latitude, longitude
        else:
            print(f"No coordinates found for '{address}'")
            return None, None

    # def get_coordinates(self, address):
    #     base_url = "https://nominatim.openstreetmap.org/search"
    #     params = {'q': address, 'format': 'json'}
    #
    #     retries = 3
    #     delay_seconds = 1
    #
    #     while retries > 0:
    #         response = requests.get(base_url, params=params)
    #         data = response.json()
    #
    #         if data:
    #             return float(data[0]['lat']), float(data[0]['lon'])
    #         else:
    #             retries -= 1
    #             sleep(delay_seconds)  # Adding a delay before retrying
    #
    #     return None, None

    def apply_coordinates(self):
        """
        Apply the function get_geo_coords_with_google to the data from HH_Provider_Oct2023.csv
        eliminating the columns State, ZIP Code, City/Town, Address
        and writing the result to HH_Provider_With_Coordinates_With_Google_API.csv
        :return: None
        """
        df = pd.read_csv("../HH_Provider_Oct2023.csv", dtype='str')

        columns_addr = ['State', 'ZIP Code', 'City/Town', 'Address']
        new_df = df.drop(columns=columns_addr)
        # addresses = df[columns_addr].apply(lambda x: " ".join(x), axis=1) .tolist()

        new_df['Latitude'], new_df['Longitude'] = zip(
            *df[columns_addr].apply(lambda x: self.get_geo_coords_with_google(" ".join(x)), axis=1))

        new_df.to_csv("HH_Provider_With_Coordinates_With_Google_API.csv", index=False)

        with open('results_google_API.json', 'w') as json_file:
            json.dump(self.__results_google_api, json_file)

    def apply_remove_rows_that_dont_have_rating(self):
        """
        Remove the rows (from the df) that don't have a Quality of patient care star rating
        :return: None
        """
        # drop the rows that contain '-' on the column Quality of patient care star rating
        self.__df.drop(self.__df[self.__df['Quality of patient care star rating'] == '-'].index, inplace=True)
        self.__df = self.__df.reset_index(drop=True)

        self.__processed_columns.append('Quality of patient care star rating')

        # for index, row in df.iterrows():
        #     print(f"Index: {index}, Row: {row['Quality of patient care star rating']}")

    def apply_nr_of_days(self):
        """
        Transform the column Certification Date from the dataset to number of days since the certification date until now
        :return: None
        """

        self.__df['Certification Date'] = pd.to_datetime(self.__df['Certification Date'], format='%m/%d/%Y')
        self.__df['Certification Date'] = (datetime.now() - self.__df['Certification Date']).dt.days

        # self.__processed_columns.append('Certification Date')

        # for row in df[['Certification Date', 'Days until now']].itertuples():
        #     print(row)

    def apply_remove_columns_footnote(self):
        """
        Remove the columns that contain 'Footnote' in their name
        :return: None
        """
        columns_to_drop = []
        for column in self.__df.columns:
            if 'Footnote' in column:
                columns_to_drop.append(column)

        # for column in columns_to_drop:
        #     print(f"Removing column '{column}'")

        self.__df.drop(columns=columns_to_drop, inplace=True)

    def apply_remove_columns_that_have_unique_values(self):
        """
        Remove the columns CMS Certification Number (CCN) and Telephone Number
        as they have unique values for each row
        and they are not relevant for the classification
        :return: None
        """
        columns_to_drop = ['CMS Certification Number (CCN)', 'Telephone Number']
        self.__df.drop(columns=columns_to_drop, inplace=True)

    def apply_label_encoding_for_performance_categorization(self):
        """
        Transform the column Performance categorisation to a numeric value
        worse -> 0
        same -> 1
        better -> 2
        not available -> 3
        - -> 4
        :return: None
        """
        mapping = {'Worse Than National Rate': 0,
                   'Same As National Rate': 1,
                   'Better Than National Rate': 2,
                   'Not Available': 3,
                   '-': 4
                   }

        columns_to_encode = ['DTC Performance Categorization', 'PPR Performance Categorization',
                             'PPH Performance Categorization']

        self.__df[columns_to_encode] = self.__df[columns_to_encode].apply(lambda x: x.map(mapping))

        for column in columns_to_encode:
            self.__processed_columns.append(column)

        # for row in df[columns_to_encode].itertuples():
        #     index = row[0]
        #     values = row[1:]
        #     print(f"Index: {index}, Values: {values}")

    def apply_label_encoding_for_offers_services(self):
        """
        Transform the column Offers ... Sevices to a numeric value
        Yes -> 1
        No -> 0
        :return: the transformed dataframe
        """

        # print(f"SHAPPA {self.__df.shape}")
        mapping = {'Yes': 1,
                   'No': 0,
                   '-': 0  # if the value is '-' then it means that the service is not offered as seen in the
                   # https://www.medicare.gov/care-compare/details/home-health/037806?city=Tempe&state=AZ&zipcode=85288
                   }

        columns_to_encode = ['Offers Nursing Care Services', 'Offers Physical Therapy Services',
                             'Offers Occupational Therapy Services', 'Offers Speech Pathology Services',
                             'Offers Medical Social Services', 'Offers Home Health Aide Services']

        self.__df[columns_to_encode] = self.__df[columns_to_encode].apply(lambda x: x.map(mapping))

        for column in columns_to_encode:
            self.__processed_columns.append(column)

        # columns_to_encode.append('Provider Name')
        # for row in df[columns_to_encode].itertuples():
        #     index = row[0]
        #     values = row[1:]
        #     print(f"Index: {index}, Values: {values}")

    def apply_label_encoding_for_type_of_ownership(self):
        """
        Transform the column Type of Ownership to a numeric value
        Proprietary -> 0
        Government - State/County -> 1
        ...etc
        :return: None
        """

        name_type_column = 'Type of Ownership'

        # Convert the column to categorical and get the codes
        self.__df[name_type_column] = self.__df[name_type_column].astype('category').cat.codes

        self.__processed_columns.append(name_type_column)

        # for i, row in enumerate(self.__df[name_type_column]):
        #     print(f"Index: {i}, Values: {row}")

    def transform_to_numeric(self):
        """
        Transform the columns to numeric values
        ex: 1,234 cannot be converted to numeric value by default, so we need to remove the comma
        :return: None
        """
        for column in self.__df.columns:
            if column not in self.__processed_columns and self.__df[column].dtype == 'O':
                # check if the column is object\
                # Convert to string, remove commas, and convert back to numeric
                self.__df[column] = pd.to_numeric(self.__df[column].astype(str).str.replace(',', ''),
                                                  errors='coerce')
            # check if type is int
            elif pd.api.types.is_integer_dtype(self.__df[column]):
                self.__df[column] = self.__df[column].astype('float')

    def apply_remove_outliers_and_replace_unknown_with_mean(self):
        """
        Remove the outliers from the dataset and replace the unknown values with the mean
        :return: None
        """

        unprocessed_columns = []
        for column in self.__df.columns:
            if column not in self.__processed_columns:
                unprocessed_columns.append(column)

        for column in unprocessed_columns:
            print(f"Processing column '{column}'")
            print(f"Shape before: {self.__df.shape}")
            self.remove_outliers_and_replace_unknown_with_mean(column)
            print(f"Shape after: {self.__df.shape}")

    def remove_outliers_and_replace_unknown_with_mean(self, column):
        column_values = self.__df[column]

        q1 = column_values.quantile(0.25)
        q3 = column_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # print(f"Lower bound: {lower_bound}")
        # print(f"Upper bound: {upper_bound}")

        outlier_index_rows = np.where((column_values < lower_bound) | (column_values > upper_bound))[0]

        if len(outlier_index_rows) > 0:
            self.__df = self.__df.drop(self.__df.index[outlier_index_rows])

        column_values = self.__df[column]

        mean = np.nanmean(column_values)
        for index, row in self.__df.iterrows():
            if pd.isna(row[column]):
                self.__df.loc[index, column] = mean.round(5)

    def apply_doc_to_vec_for_provider_name(self):
        """
        Apply doc2vec for the column Provider Name
        :return: None
        """

        documents = []
        for i, text in enumerate(self.__df['Provider Name']):
            if pd.isnull(text):
                text = " "

            documents.append(TaggedDocument(words=word_tokenize(text.lower()), tags=[str(i)]))

        vector_size = 10  # adjust as needed
        window = 2
        min_count = 1
        epochs = 20
        model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=-1, epochs=epochs)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        # Infer vectors for the 'Provider Name' column
        inferred_vectors = []

        for tagged_doc in documents:
            # Infer the vector for each document
            inferred_vector = model.infer_vector(tagged_doc.words)

            # Append the inferred vector to the list
            inferred_vectors.append(list(inferred_vector))

        inferred_df = pd.DataFrame(inferred_vectors,
                                   columns=[f'Provider_Name_{i + 1}' for i in range(len(inferred_vectors[0]))])

        self.__df.drop(columns=['Provider Name'], inplace=True)

        self.__df = pd.concat([self.__df, inferred_df], axis=1)

def main():
    # apply preprocessing
    # apply_coordinates()
    df = pd.read_csv("HH_Provider_With_Coordinates_With_Google_API.csv")
    preprocess = NewPreprocess(df)
    preprocess.apply_remove_columns_footnote()
    preprocess.apply_remove_columns_that_have_unique_values()
    preprocess.apply_remove_rows_that_dont_have_rating()
    preprocess.apply_label_encoding_for_performance_categorization()
    preprocess.apply_label_encoding_for_offers_services()
    preprocess.apply_label_encoding_for_type_of_ownership()
    preprocess.apply_nr_of_days()
    preprocess.apply_doc_to_vec_for_provider_name()
    preprocess.transform_to_numeric()
    preprocess.apply_remove_outliers_and_replace_unknown_with_mean()
    df = preprocess.get_df()

    df.to_csv("Datas_After_Preprocessing.csv", index=False)


if __name__ == '__main__':
    main()
