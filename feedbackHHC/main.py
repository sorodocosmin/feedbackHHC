import pandas as pd

from Feature_selection import Feature_selection

file_path = 'HH_Provider_Oct2023.csv'

if __name__ == "__main__":
    df = pd.read_csv("Preprocessed_data.csv", dtype='float')
    # print(df.dtypes)
    fs = Feature_selection(df)
    fs.apply_feature_selection()
