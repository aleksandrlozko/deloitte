import re

import pandas as pd
import pymorphy2

df = pd.read_pickle('train_df.pkl')
print(df.keywords[0])

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[.,:;_%©№?*,!@#$%^()\d]|[+=]|[\[][]]|[/]|"|\s{2,}|-', ' ', text)
    text = re.sub('forwarding\(', ' ', text)
    ma = pymorphy2.MorphAnalyzer()
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())

    return text

k=0

for x in df.keywords:
    df.keywords[k] = ', '.join(x)
    k+=1
print(df.keywords)

df['keywords_clean'] = df.apply(lambda x: clean_text(df.keywords))
print(df.keywords_clean)