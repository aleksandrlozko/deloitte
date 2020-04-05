import pandas as pd


df_test = pd.read_pickle('test_df.pkl')
print(len(df_test))
def test_function(df_test, total_uniq):
    for el in total_uniq:
        df_test[str(el)] = 0

    m = 1
    for i in df_test.accepted_function:
        for el in i.split(','):
            el = str(el).strip().lower()
            df_test[str(el)][m] = 1
        m += 1

    spisok_real_test = list()
    for x in total_uniq:
        spisok_real_test.append(df_test[str(x)].tolist())

    return spisok_real_test


def test_keywords(df_test):

    test_keywords = list()
    for x in df_test['keywords']:
        # test_real_keywords = list()
        if len(x) != 0:
            x = ' '.join(x).lower().strip().split(' ')
            test_keywords.append(x)
        else:
            test_keywords.append(x)
        #     test_real_keywords.append(x)
        # else:
        #     for y in x:
        #         y = y.lower().strip()
        #         test_real_keywords.append(y)

        # test_keywords.append(test_real_keywords)
    print(test_keywords)
    df_test_real_keywords = pd.DataFrame()
    df_test_real_keywords['keywords'] = test_keywords
    return test_keywords

print(test_keywords(df_test))