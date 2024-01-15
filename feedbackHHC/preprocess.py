import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self):
        self.empty_percentage, self.irrelevant_attributes, \
            self.input_file, self.preprocess_file = read_xml_file()
        self.unprocessed_data = pd.read_csv(self.input_file)
        self.preprocessed_data = self.unprocessed_data
        self.column_types = {}

    def preprocess(self):
        self.drop_irrelevant_attributes(self.irrelevant_attributes)
        self.drop_almost_empty_columns()
        self.column_types = self.assign_type_to_columns(self.preprocessed_data)
        self.preprocessed_data = self.transform_to_numeric(self.preprocessed_data)

        # drop rows that have empty values
        self.drop_rows()
        self.drop_outliers_and_replace_with_mean()

        # save the preprocessed data to a csv file
        self.preprocessed_data.to_csv(self.preprocess_file, index=False)

    def drop_irrelevant_attributes(self, irrelevant_attributes):
        # for every column in the dataset check if it is irrelevant
        for column in self.preprocessed_data.columns:
            if is_irrelevant_column(column, irrelevant_attributes):
                self.preprocessed_data.drop(column, axis=1, inplace=True)

    def drop_almost_empty_columns(self):
        # for every column count the empty('-') values on each row
        count_empty = {}
        for index, row in self.preprocessed_data.iterrows():
            for column in self.preprocessed_data.columns:
                if column not in count_empty:
                    count_empty[column] = 0
                if row[column] == '-':
                    count_empty[column] += 1

        # drop the columns that have more than empty_percentange empty values
        total_rows = len(self.preprocessed_data.index)
        for column in count_empty:
            if count_empty[column] > self.empty_percentage * total_rows:
                self.preprocessed_data.drop(column, axis=1, inplace=True)

    @staticmethod
    def assign_type_to_columns(data):
        column_types = {}
        for index, row in data.iterrows():
            if '-' not in row.values:
                for column in data.columns:
                    if is_bool_value(row[column]):
                        column_types[column] = 'bool'
                    elif is_float_value(row[column]):
                        column_types[column] = 'float'
                    else:
                        column_types[column] = 'string'
        return column_types

    @staticmethod
    def transform_to_numeric(data):
        for index, row in data.iterrows():
            for column in data.columns:
                if is_bool_value(row[column]):
                    row[column] = bool_to_int(row[column])
                elif is_float_value(row[column]):
                    row[column] = str_to_float(row[column])
        return data

    def drop_rows(self):
        for index, row in self.preprocessed_data.iterrows():
            for column in self.preprocessed_data.columns:
                if self.column_types[column] == 'bool':
                    if row[column] == '-':
                        self.preprocessed_data.drop(index, inplace=True)
                        break

    def drop_outliers_and_replace_with_mean(self):
        for column in self.column_types:
            if self.column_types[column] == 'float':
                self.remove_outlier_for_col(column)

    def remove_outlier_for_col(self, column):
        column_values = self.preprocessed_data[column]
        column_values = pd.to_numeric(column_values, errors='coerce')

        q1 = column_values.quantile(0.25)
        q3 = column_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_index_rows = np.where((column_values < lower_bound) | (column_values > upper_bound))[0]

        if len(outlier_index_rows) > 0:
            self.preprocessed_data = self.preprocessed_data.drop(self.preprocessed_data.index[outlier_index_rows])

        column_values = self.preprocessed_data[column]
        column_values = pd.to_numeric(column_values, errors='coerce')

        mean = np.nanmean(column_values)

        for index, row in self.preprocessed_data.iterrows():
            if row[column] == '-':
                row[column] = mean.round(3)


def is_float_value(value):
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    elif isinstance(value, (int, float)):
        return True
    return False


def is_bool_value(value):
    if isinstance(value, str):
        value = value.lower()
        if "yes" in value or "no" in value:
            return True
    return False


def bool_to_int(param):
    if is_bool_value(param):
        if param.lower() == 'yes':
            return 1
        else:
            return 0


def str_to_float(value):
    if is_float_value(value):
        if isinstance(value, str):
            return float(value.replace(",", ""))
        elif isinstance(value, (int, float)):
            return float(value)
    return None


def is_irrelevant_column(column, irrelevant_attributes):
    for attribute in irrelevant_attributes:
        if column.lower().startswith(attribute.lower()):
            return True
    return False


def read_xml_file():
    # read the xml file
    tree = ET.parse('preprocess.xml')
    root = tree.getroot()

    empty_percentage = float(root.find('empty_percentange').text)

    input_file = root.find('input_file').text
    preprocess_file = root.find('preprocess_file').text

    irrelevant_attributes = []
    for attribute in root.find('irrelevantAttributes'):
        irrelevant_attributes.append(attribute.text)

    return empty_percentage, irrelevant_attributes, input_file, preprocess_file


if __name__ == '__main__':
    pp = Preprocess()
    pp.preprocess()
    print(pp.column_types)
