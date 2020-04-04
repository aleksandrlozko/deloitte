import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from test import uniq

class PrepareText:
    """
    Prepare text for model
    """

    @staticmethod
    def tokenizer(descriptions):
        """
        Vectorize a text corpus, by turning each text into either a sequence of integers

        :param descriptions: Clean text
        :return: tokenizer - text tokenization utility class, textSequences - Converts a text to a sequence of words (or tokens)
        """

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        return tokenizer, textSequences

    @staticmethod
    def max_count(keywords, tokenizer):
        """
        Count words in the text

        :param descriptions: Clean text
        :param tokenizer: Text tokenization utility class
        :return: total_unique_words - total unique words, maxSequenceLength - max amount of words in the longest text
        """

        max_words = 0
        for desc in keywords.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

        total_unique_words = len(tokenizer.word_counts)
        print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

        maxSequenceLength = max_words

        return total_unique_words, maxSequenceLength

    @staticmethod
    def num_classes(y_train):
        """
        Count categories

        :param y_train: Training labels
        :param y_test: Test labels
        :return: encoder - encode target labels, num_classes - amount of classes
        """

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        #y_test = encoder.transform(y_test)

        num_classes = np.max(y_train) + 1

        print('Количество категорий для классификации: {}'.format(num_classes))

        return encoder, num_classes

    @staticmethod
    def load_data_from_arrays(df, train_test_split=0.8):
        """
        Divide data into training and test sets

        :param train_test_split: Training set size
        :return: Data divided into training and test
        """
        accepted_function_uniq = uniq(df.accepted_function)
        rejected_function_uniq = uniq(df.rejected_function)
        accepted_product_uniq = uniq(df.accepted_product)
        rejected_product_uniq = uniq(df.rejected_product)

        total_uniq = list()
        for n in accepted_function_uniq:
            if n not in total_uniq:
                total_uniq.append(n)
        for n in rejected_function_uniq:
            if n not in total_uniq:
                total_uniq.append(n)
        for n in accepted_product_uniq:
            if n not in total_uniq:
                total_uniq.append(n)
        for n in rejected_product_uniq:
            if n not in total_uniq:
                total_uniq.append(n)

        # total_uniq.remove('accounting')
        # total_uniq.remove('bookeeping')
        # total_uniq.remove('auditing')

        data_size = len(df.id)
        test_size = int(data_size - round(data_size * train_test_split))

        y_train = df.target[test_size:]
        y_test = df.target[:test_size]

        l = 12411

        spisok_train = list()
        for x in total_uniq:
            print(df[str(x)])
            print(df[str(x)].tolist())
            spisok_train.append(df[str(x)].tolist()[:l])

        spisok_test = list()
        for x in total_uniq:
            spisok_test.append(df[str(x)].tolist()[l:-1])


        return spisok_train, y_train, spisok_test, y_test, total_uniq

    @staticmethod
    def transform_sets(vocab_size, descriptions, X_train, X_test, y_train, y_test, maxSequenceLength, num_classes):
        """
        Transform the data for training and testing in the format we need

        :param vocab_size: 10% of total unique words
        :param descriptions: Clean text
        :param X_train: Training data set
        :param X_test: Test data set
        :param y_train: Training labels
        :param y_test: Test labels
        :param maxSequenceLength: Max amount of words in the longest text
        :param num_classes: Number of classes
        :return: transformed data
        """

        print('Преобразуем описания заявок в векторы чисел...')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(descriptions)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
        X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

        print('Размерность X_train:', X_train.shape)
        print('Размерность X_test:', X_test.shape)

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

        return X_train, X_test, y_train, y_test

