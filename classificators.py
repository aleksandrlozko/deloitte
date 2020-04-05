import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from TextCleaner import CleanText
from TextPreparer import PrepareText
from test import function
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from test_test import test_function, test_keywords


df_test = pd.read_pickle('test_df.pkl')
print(df_test)
train = pd.read_pickle('train_df.pkl')
test = pd.read_pickle('test_df.pkl')

spisok_test, spisok_train, total_uniq = function(train)
# train = CleanText.clean_category(train)
train = CleanText.prepare_text(train)

test_keywords1 = test_keywords(df_test)
spisok_real_test = test_function(df_test, total_uniq)

# df_Test = pd.DataFrame(spisok_real_test, columns=total_uniq)


tokenizer, textSequences = PrepareText.tokenizer(train.keywords_clean)

y_train, y_test = PrepareText.load_data_from_arrays(train, train_test_split=0.8)

# Y_train = []
# for y in y_train:
#     Y_train.append([y])
#
# Y_test = []
# for y in y_train:
#     Y_test.append([y])

total_unique_words, maxSequenceLength = PrepareText.max_count(train.keywords_clean, tokenizer)
# vocab_size = round(total_unique_words / 10)

#num_words=vocab_size
tokenizer.fit_on_texts(train.keywords_clean)

train_keywords = train.keywords_clean[:11436]
test_keywords = train.keywords_clean[11436:]

XK = tokenizer.texts_to_sequences(test_keywords1)
X_train = tokenizer.texts_to_sequences(train_keywords)
X_test = tokenizer.texts_to_sequences(test_keywords)

XK = sequence.pad_sequences(XK, maxlen=maxSequenceLength)
X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)


X_testX = []
for x in X_train:
    X_testX.append(sum(x))

normalized_XX = preprocessing.normalize([X_testX])



X_train1 = []
for x in X_train:
    X_train1.append(sum(x))

normalized_X = preprocessing.normalize([X_train1])

X_test1 = []
for x in X_test:
    X_test1.append(sum(x))


normalized_X1 = preprocessing.normalize([X_test1])

spisok_train=spisok_train + [normalized_X[0]]
spisok_test=spisok_test + [normalized_X1[0][:2064]]
spisok_final = spisok_real_test +[normalized_XX[0]]
print(len(spisok_real_test[0]))
print(len(normalized_XX[0]))
print(len([normalized_XX[0]]))
total_uniq=total_uniq + ['keywords']
def create_dict(spisok_train, total_uniq):
    dict_train = dict()
    b = 0
    for x in total_uniq:
        dict_train[str(x)] = spisok_train[b]
        b +=1
    return dict_train

def create_dict1(spisok_test, total_uniq):
    dict_test = dict()
    b = 0
    for x in total_uniq:
        dict_test[str(x)] = spisok_test[b]
        b +=1
    return dict_test

def create_dict2(spisok_final, total_uniq):
    dict_test = dict()
    b = 0
    for x in total_uniq:
        dict_test[str(x)] = spisok_final[b]
        b +=1
    return dict_test


print(len(spisok_train), len(total_uniq))
print(len(spisok_test), len(total_uniq))
main = create_dict(spisok_train,total_uniq)
main1 = create_dict1(spisok_test,total_uniq)
main2=create_dict2(spisok_final, total_uniq)
print(len(main2))
print(len(spisok_final))
print(len(spisok_final[0]),len(spisok_final[1]),len(spisok_final[2]),len(spisok_final[3]),len(spisok_final[4]),len(spisok_final[5]))
print(len(total_uniq))

df_train = pd.DataFrame(main)
df_test = pd.DataFrame(main1)
df_Test=pd.DataFrame(main2)

classifier = RandomForestClassifier(criterion='gini', max_depth=21, n_estimators=1)

classifier.fit(df_train, y_train)
y_pred = classifier.predict(df_Test)
print(y_pred)
# report = classification_report(y_test[:2064], y_pred)
# print(report)
#
# matrix = confusion_matrix(y_test[:2064], y_pred)
# print(matrix)
qw=pd.DataFrame({'id':df_test.id,
                 'target':y_pred})
qw.to_csv('submission1.csv')


classifier = SVC()

classifier.fit(df_train, y_train)
y_pred = classifier.predict(df_Test)
print(y_pred)
qw=pd.DataFrame({'id':df_test['id'],
                 'target':y_pred})
qw.to_csv('submission2.csv')
# report = classification_report(y_test[:2064], y_pred)
# print(report)
#
# matrix = confusion_matrix(y_test[:2064], y_pred)
# print(matrix)


classifier = LogisticRegression(multi_class='ovr', penalty='none', solver='sag')

classifier.fit(df_train, y_train)
y_pred = classifier.predict(df_Test)
print(y_pred)
qw=pd.DataFrame({'id':df_test.id,
                 'target':y_pred})
qw.to_csv('submission3.csv')
# report = classification_report(y_test[:2064], y_pred)
# print(report)
#
# matrix = confusion_matrix(y_test[:2064], y_pred)
# print(matrix)