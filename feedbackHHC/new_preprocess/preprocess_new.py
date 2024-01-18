from time import sleep

import pandas as pd
import requests
import json


import googlemaps

import random

# Replace 'YOUR_API_KEY' with the API key you obtained
api_key = 'secret ; )'
gmaps = googlemaps.Client(key=api_key)




results = {}

def get_geo_coords_with_google(address):
    # Geocoding request
    global results

    geocode_result = gmaps.geocode(address)

    if geocode_result:
        results[address] = geocode_result
        location = geocode_result[0]['geometry']['location']
        latitude = location['lat']
        longitude = location['lng']
        print(f"Coordinates for '{address}': ({latitude}, {longitude})")
        return latitude, longitude
    else:
        print(f"No coordinates found for '{address}'")
        return None, None


# def get_coordinates(address):
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


df = pd.read_csv("../HH_Provider_Oct2023.csv", dtype='str')


columns_addr = ['State', 'ZIP Code', 'City/Town', 'Address']
new_df = df.drop(columns=columns_addr)
addresses = df[columns_addr].apply(lambda x: " ".join(x), axis=1) .tolist()


new_df['Latitude'], new_df['Longitude'] = zip(*df[columns_addr].apply(lambda x: get_geo_coords_with_google(" ".join(x)), axis=1))

new_df.to_csv("HH_Provider_With_Coordinates_With_Google_API.csv", index=False)

# with open('results_google_API.json', 'w') as json_file:
#     json.dump(results, json_file)


#
# text_data = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
# text_vectors = tfidf_vectorizer.fit_transform(text_data)
#
# print(text_vectors)

# print(df)
