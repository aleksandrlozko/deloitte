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

def function(df):
    lenn, union = uniq_two(df.accepted_function, df.rejected_function)
    lenn1, union1 = uniq_two(df.accepted_product, df.rejected_product)
    lenn2, union2 = uniq_two(df.accepted_product, df.rejected_function)
    lenn3, union3 = uniq_two(df.accepted_function, df.rejected_product)


    k=0
    for i in df.accepted_function:
        i = i.lower().strip().split(',')
        for h in i:
            if h in union or h in union3:
                i.remove(h)
        df.accepted_function[k] = ','.join(i)
        k+=1

    k=0
    for i in df.rejected_function:
        i = i.lower().strip().split(',')
        for h in i:
            if h in union or h in union2:
                i.remove(h)
        df.rejected_function[k] = ','.join(i)
        k+=1

    k=0
    for i in df.accepted_product:
        i = i.lower().strip().split(',')
        for h in i:
            if h in union1 or h in union2:
                i.remove(h)
        df.accepted_product[k] = ','.join(i)
        k+=1

    k=0
    for i in df.rejected_product:
        i = i.lower().strip().split(',')
        for h in i:
            if h in union1 or h in union3:
                i.remove(h)
        df.rejected_product[k] = ','.join(i)
        k+=1

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


    for el in total_uniq:
        df[str(el)] = 0

    m=1
    for i in df.accepted_function:
        for el in i.split(','):
            el = str(el).strip().lower()
            df[str(el)][m] = 1
        m += 1

    l = 12411

    spisok_train = list()
    for x in total_uniq:
        spisok_train.append(df[str(x)].tolist()[:l])

    spisok_test = list()
    for x in total_uniq:
        spisok_test.append(df[str(x)].tolist()[l:-1])

    return spisok_test, spisok_train, total_uniq
# df_train = pd.DataFrame(spisok, columns=total_uniq)
# print(accepted_function_uniq, rejected_function_uniq)

# print(len(accepted_function_uniq), len(rejected_function_uniq))
# print(len(accepted_product_uniq), len(rejected_product_uniq))

# print(df.accepted_function)
# print(df.rejected_function)