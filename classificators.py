import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from TextCleaner import CleanText
from TextPreparer import PrepareText


train = pd.read_pickle('train_df.pkl')
test = pd.read_pickle('test_df.pkl')

# train = CleanText.clean_category(train)
train = CleanText.prepare_text(train)
print(train)
tokenizer, textSequences = PrepareText.tokenizer(train.keywords_clean)

#X_train, y_train, X_test, y_test = PrepareText.load_data_from_arrays(descriptions, categories, train_test_split=0.8)


total_unique_words, maxSequenceLength = PrepareText.max_count(train.keywords_clean, tokenizer)
# vocab_size = round(total_unique_words / 10)

#num_words=vocab_size
tokenizer.fit_on_texts(train.keywords_clean)

X_train = tokenizer.texts_to_sequences(train.keywords_clean)
#X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)


classifier = RandomForestClassifier()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

report = classification_report(Y_test, y_pred)
print(report)

matrix = confusion_matrix(Y_test, y_pred)
print(matrix)


classifier = SVC()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

report = classification_report(Y_test, y_pred)
print(report)

matrix = confusion_matrix(Y_test, y_pred)
print(matrix)


classifier = LogisticRegression()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

report = classification_report(Y_test, y_pred)
print(report)

matrix = confusion_matrix(Y_test, y_pred)
print(matrix)

