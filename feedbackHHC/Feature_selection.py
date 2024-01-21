import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Feature_selection:
    def __init__(self, df):
        self.df = df

    def apply_feature_selection(self):
        """
        Apply a feature section using a correlation matrix, matrix which was obtained using the Pearson correlation
        It also saves the new dataset to a csv file
        :return: None
        """
        corr_matrix = self.df.corr()

        plt.figure(figsize=(60, 60))
        sns.heatmap(corr_matrix, annot=True)
        plt.show()

        # Correlation with output variable
        corr_with_target = corr_matrix["Quality of patient care star rating"]

        # threshold
        threshold = 0.2

        # select features with abs(corr) >= threshold
        relevant_features = corr_with_target[abs(corr_with_target) >= threshold]
        print(f"Relevant features : {relevant_features}")

        relevant_features = relevant_features.drop("Quality of patient care star rating")

        # print(len(relevant_features))

        # correlation matrix for relevant features
        corr_matrix_relevant_features = self.df[relevant_features.index].corr()

        threshold_dependent_features = 0.8

        # delete the features that are dependent: have a correlation coefficient >= threshold
        # create a list of columns to drop
        columns_to_drop = []
        for i in range(len(corr_matrix_relevant_features.columns)):
            for j in range(i):
                if abs(corr_matrix_relevant_features.iloc[i, j]) >= threshold_dependent_features:
                    colname = corr_matrix_relevant_features.columns[i]
                    if colname not in columns_to_drop:
                        columns_to_drop.append(colname)

        # drop the columns based on the list of column names
        independent_features = relevant_features.index.drop(columns_to_drop)

        # save the new dataset to a csv file
        # add target column to the independent features
        independent_features = independent_features.append(pd.Index(["Quality of patient care star rating"]))

        # save the new dataset to a csv file
        self.df[independent_features].to_csv("Final_1_data.csv", index=False)


if __name__ == '__main__':
    df = pd.read_csv("new_preprocess/Datas_After_Preprocessing.csv")
    feature_selection = Feature_selection(df)
    feature_selection.apply_feature_selection()
