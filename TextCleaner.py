import re
import pandas as pd
from nltk.corpus import stopwords
import pymorphy2


class CleanText:
    """
    Clean text and create new columns with new data
    """

    @staticmethod
    def clean_text(text):
        """
        Clean text

        :param text: text to clean
        :return: clean text
        """

        text = str(text).lower()
        return text

    @staticmethod
    def prepare_text(df):
        """
        Create new column in DataFrame with clean text(using function 'clean_text')

        :param df: DataFrame which consist of the text to be cleaned
        :return: DataFrame with new column of cleaned text
        """
        k = 0
        for x in df.keywords:
            df.keywords[k] = ' '.join(x)
            k += 1

        df['keywords_clean'] = df.apply(lambda x: CleanText.clean_text(x['keywords']), axis=1)

        return df

    @staticmethod
    def clean_category(df):
        """
        Replace categories with numbers

        :param df: DataFrame with categories
        :return: DataFrame with new column of replaced categories
        """

        categories = {}
        for key, value in enumerate(df['category'].unique()):
            categories[value] = key

        df['category_code'] = df['category'].map(categories)

        total_categories = len(df['category'].unique())
        print('Total categories: {}'.format(total_categories))

        return df

    @staticmethod
    def prepare_df(df, pickle):
        """
        Create new DataFrame with cleaned text and replaced categories (using functions 'clean_category' and 'prepare_text')

        :param df: Default DataFrame
        :param pickle: Way to dataframe.pkl
        :return: New DataFrame with clean text and replaced categories
        """

        if 'category_code' not in df.columns:
            df = CleanText.clean_category(df)
        c_text = CleanText.prepare_text(df)
        df.to_pickle(pickle)
        df_new = pd.read_pickle(pickle)

        return df_new

