import re
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd


df = pd.read_pickle('train_df.pkl')

def clean_text(text):
    text = str(text).lower()
    return text

k=0
for x in df.keywords:
    df.keywords[k] = ', '.join(x)
    k+=1

df['keywords_clean'] = df.apply(lambda x: clean_text(x.keywords), axis=1)
print(df.keywords_clean[0])