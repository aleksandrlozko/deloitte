import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_pickle('train_df.pkl')

train['text_str'] =  train['text'].apply(lambda x: ' '.join(x))

train['keywords_str'] = train['keywords'].apply(lambda x: ' '.join(x))

train.drop(columns = ['id','html','text','keywords'], inplace = True)

train_df = pd.DataFrame()
train_df["target"] = train["target"]
train_df["text"] = (
    + train["keywords_str"]
    + train["accepted_function"]
    + train["rejected_function"]
    + train["accepted_product"]
    + train["rejected_product"]
)
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True,
    min_df=5,
    max_features=100000)
word_vectorizer.fit_transform(train_df['text'])

print(word_vectorizer.get_feature_names())