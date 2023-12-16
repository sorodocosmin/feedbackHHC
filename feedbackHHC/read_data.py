import pandas as pd
import numpy as np

class ReadData:
    def __init__(self, text_file_name):
        self.__df = pd.read_csv(text_file_name)
        self.__dict_types = {}

    def elim_cols(self, list_nr_cols_to_be_eliminated):
        """
        eliminate columns from the dataframe
        creates a dictionary with the type of each column in the following format:
        {0 : "float", 1:"bool", 2:"string"}
        :param list_nr_cols_to_be_eliminated:
        :return:
        """

        self.__df = self.__df.drop(self.__df.columns[list_nr_cols_to_be_eliminated], axis=1)
        # axis = 1 refers to the columns

        # go through rows until you find one with no '-' in it
        for index, row in self.__df.iterrows():
            if '-' not in row:
                if '-' not in row.values:
                    for col, elem in enumerate(row.values):
                        # check if elem contains yes or no
                        if self.elem_is_bool(elem):
                            self.__dict_types[col] = "bool"
                        elif self.elem_is_float(elem):
                            self.__dict_types[col] = "float"
                        else:
                            self.__dict_types[col] = "string"
                break

    def elim_rows(self):
        """
        for every row that has an '-' in the cols of type bool, we eliminate that row
        :return:
        """
        for index, row in self.__df.iterrows():
            for col, elem in enumerate(row.values):
                if self.__dict_types[col] == "bool":
                    if elem == '-':
                        self.__df = self.__df.drop(index)
                        break

    def elim_outliers_and_replace_with_mean(self):
        """
        for every row that on a column of type float is outlier, we remove it
        for every row that on a column of type float is '-', we replace it with the mean of that column
        :return:
        """
        for column, type_col in self.__dict_types.items():
            if type_col == "float":
                print(f"-------------------------------\ncolumn {column} is float")
                self.remove_outlier_for_col(column)
                print(f"\n---------------------------------")

    def remove_outlier_for_col(self, column):
        column_values = self.__df.iloc[:, column]
        column_values = pd.to_numeric(column_values.replace('-', np.nan), errors='coerce')
        Q1 = column_values.quantile(0.25)
        Q3 = column_values.quantile(0.75)
        IQR = Q3 - Q1
        # threshold = 1.5
        # Why 1.5?: The choice of 1.5 is somewhat arbitrary but is based on the assumption that the data follows an
        # approximately normal distribution. In a normal distribution, about 99.3% of the data lies within
        # 3 standard deviations of the mean.

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(lower_bound)
        print(upper_bound)
        print(self.__df.shape)
        outlier_index_rows = np.where((column_values < lower_bound) | (column_values > upper_bound))[0]

        print(outlier_index_rows)

        df_no_outliers_actual_values = self.__df.drop(index=outlier_index_rows)

        #print(self.__df)

    @staticmethod
    def elem_is_float(string):
        try:
            # in our dataset , is used as a separator for thousands
            float(string.replace(',', ''))
            return True
        except ValueError:
            return False

    @staticmethod
    def elem_is_bool(string):
        string = string.lower()
        if "yes" in string or "no" in string:
            return True
        return False

    def print_cols_names(self):
        print(self.__df.columns.values)

    def get_dataframe(self):
        return self.__df



