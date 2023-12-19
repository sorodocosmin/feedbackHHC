import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class Feature_selection:
    def __init__(self, df):
        self.df = df

    def apply_feature_selection(self):
        cor = self.df.corr()

        plt.figure(figsize=(40, 40))
        sns.heatmap(cor, annot=True)

        plt.show()



