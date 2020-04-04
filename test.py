import pandas as pd
from collections import Counter

df = pd.read_pickle('train_df.pkl')
# print(test.head())
# print(test.columns)
# print(len(test))
# n=15
# print(test.text[n])
# print(test.accepted_function[n])
# print(test.rejected_function[n])
# print(test.accepted_product[n])
# print(test.rejected_product[n])
# print(test.target[n])



def uniq(category):
    category_one = list()

    for x in category:
        for y in x.split(','):
            category_one.append(y.strip().lower())

    df_one = pd.DataFrame({'unique':category_one})
    df_one_uniq = df_one['unique'].unique()

    return df_one_uniq

def uniq_two(category1, category2):
    first = list(uniq(category1))
    second = list(uniq(category2))

    union = list((Counter(first) & Counter(second)).elements())
    union = pd.DataFrame({'unique':union})

    len_union = len(union.unique.unique())
    value_union = union.unique.unique()

    return len_union, value_union

lenn, union = uniq_two(df.accepted_function, df.rejected_function)
accepted_function_uniq = uniq(df.accepted_function)
rejected_function_uniq = uniq(df.rejected_function)
print(len(accepted_function_uniq), len(rejected_function_uniq))

k=0
for i in df.accepted_function:
    i = i.lower().split(', ')
    for h in i:
        if h in union:
            i.remove(h)
    df.accepted_function[k] = ','.join(i)
    k+=1

k=0
for i in df.rejected_function:
    i = i.lower().split(', ')
    for h in i:
        if h in union:
            i.remove(h)
    df.rejected_function[k] = ','.join(i)
    k+=1

accepted_function_uniq = uniq(df.accepted_function)
rejected_function_uniq = uniq(df.rejected_function)
print(accepted_function_uniq, rejected_function_uniq)
print(len(accepted_function_uniq), len(rejected_function_uniq))

# print(df.accepted_function)
# print(df.rejected_function)

