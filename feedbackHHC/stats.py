import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from preprocess import Preprocess


class Stats:
    def __init__(self):
        pp = Preprocess()
        self.data = pd.read_csv(pp.preprocess_file)
        self.__dict_types = pp.asign_type_to_columns(self.data)
        self.dict_mean_median = self.find_mean_and_median()

    def find_mean_and_median(self):
        dict_mean_median = {}
        for column, type_col in self.__dict_types.items():
            if type_col == "float":
                column_values = self.data[column]
                column_values = pd.to_numeric(column_values, errors='coerce')

                mean = np.nanmean(column_values)
                median = np.nanmedian(column_values)
                dict_mean_median[column] = [mean, median]
        return dict_mean_median

    def plot_mean_and_median(self):
        dict_mean_median = self.find_mean_and_median()

        columns = list(dict_mean_median.keys())
        means = [values[0] for values in dict_mean_median.values()]
        medians = [values[1] for values in dict_mean_median.values()]

        bar_width = 0.45
        space_between_bars = 0.2
        index = np.arange(len(columns))

        bars_mean = plt.bar(index, means, bar_width, label='Mean', color='blue')
        bars_median = plt.bar(index + bar_width, medians, bar_width, label='Median', color='orange')

        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.title('Mean and Median for Numeric Columns')
        plt.xticks(index + bar_width + space_between_bars, columns, rotation='vertical')
        # Rotate x-axis labels vertically
        plt.legend()

        # Display values on top of each bar
        self.autolabel(bars_mean, means)
        self.autolabel(bars_median, medians)

        plt.savefig('mean_median.png', dpi=300)

        plt.show()

    @staticmethod
    def autolabel(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{value:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                color='black',
                rotation='vertical'
            )


def read_xml_file():
    # read the xml file
    tree = ET.parse('preprocess.xml')
    root = tree.getroot()
    preprocess_file = root.find('preprocess_file').text
    return preprocess_file


if __name__ == "__main__":
    stats = Stats()
    stats.plot_mean_and_median()
